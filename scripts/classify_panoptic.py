import argparse
import time
from pathlib import Path
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.utils.data as data
import umsgpack
import fire
from tqdm import tqdm
from PIL import Image
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries
from torch import distributed

import seamseg.models as models
from seamseg.algos.detection import PredictionGenerator as BbxPredictionGenerator, DetectionLoss, \
    ProposalMatcher
from seamseg.algos.fpn import InstanceSegAlgoFPN, RPNAlgoFPN
from seamseg.algos.instance_seg import PredictionGenerator as MskPredictionGenerator, InstanceSegLoss
from seamseg.algos.rpn import AnchorMatcher, ProposalGenerator, RPNLoss
from seamseg.algos.semantic_seg import SemanticSegAlgo, SemanticSegLoss
from seamseg.config import load_config, DEFAULTS as DEFAULT_CONFIGS
from seamseg.data import ISSTestDataset, ISSTestTransform, iss_collate_fn
from seamseg.data.sampler import DistributedARBatchSampler
from seamseg.models.panoptic import PanopticNet
from seamseg.modules.fpn import FPN, FPNBody
from seamseg.modules.heads import FPNMaskHead, RPNHead, FPNSemanticHeadDeeplab
from seamseg.utils import logging
from seamseg.utils.meters import AverageMeter
from seamseg.utils.misc import config_to_string, norm_act_from_config
from seamseg.utils.panoptic import PanopticPreprocessing
from seamseg.utils.parallel import DistributedDataParallel
from seamseg.utils.snapshot import resume_from_snapshot

from csv_dataset import ISSTestCSVDatatset

def binary_classification(input_csv_path,
                          output_csv_path,
                          log_dir,
                          lbl_pixel_min=1,
                          transport_infra='marking--crosswalk-zebra',
                          **kwargs):

    model, meta, config, test_kwargs = _init_pretrained_and_logging(log_dir, **kwargs)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    test_dataloader = _make_dataloader(input_csv_path, config, rank, world_size)

    transport_infra_id = meta['categories'].index(transport_infra)
    def label_function(raw_pred):
        sem_pred, _,_,_,_ = raw_pred
        lbls, lbl_counts = np.unique(sem_pred.cpu(), return_counts=True)
        lbl2count = dict(zip(lbls, lbl_counts))
        transport_infra_npxl = lbl2count.get(transport_infra_id)
        bin_lbl = 1 if transport_infra_npxl and transport_infra_npxl > lbl_pixel_min else 0

    labels = _test(model,
                   test_dataloader,
                   label_function=label_function,
                   **test_kwargs)
    imgs_df = pd.read_csv(input_csv_path, index_col=0)
    imgs_df[f'bin-lbl_{transport_infra}'] = labels

def _init_pretrained_and_logging(log_dir,
                                 local_rank,
                                 config_path='../config.ini',
                                 model_path='../seamseg_r50_vistas.tar',
                                 meta_path='../metadata.bin',
                                 score_threshold=.5,
                                 iou_threshold=.5,
                                 min_area=4096):
    # Initialize multi-processing
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = local_rank, torch.device(local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Initialize logging
    if rank == 0:
        logging.init(log_dir, "test")

    # Load configuration
    config = _make_config(config_path)
    meta = _load_meta(meta_path)

    # Create model
    model = _make_model(config, meta["num_thing"], meta["num_stuff"])
    # Load snapshot
    _log_debug("Loading snapshot from %s", model_path)
    resume_from_snapshot(model, model_path, ["body", "rpn_head", "roi_head", "sem_head"])

    # Init GPU stuff
    torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
    model = DistributedDataParallel(model.cuda(device), device_ids=[device_id], output_device=device_id)

    # Panoptic processing parameters
    panoptic_preprocessing = PanopticPreprocessing(score_threshold, iou_threshold, min_area)

    test_kwargs = {
        'make_panoptic': panoptic_preprocessing,
        'num_stuff': meta['num_stuff'],
        'log_interval': config["general"].getint("log_interval"),
        'device': device,
        'summary': None
    }
    return model, meta, config, test_kwargs


def _log_debug(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().debug(msg, *args, **kwargs)


def _log_info(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().info(msg, *args, **kwargs)


def _make_config(config):
    _log_debug("Loading configuration from %s", config)

    conf = load_config(config, DEFAULT_CONFIGS["panoptic"])

    _log_debug("\n%s", config_to_string(conf))
    return conf


def _make_dataloader(input_csv_path, config, rank, world_size):
    config = config["dataloader"]
    _log_debug("Creating dataloaders for dataset in %s", input_csv_path)

    # Validation dataloader
    test_tf = ISSTestTransform(config.getint("shortest_size"),
                               config.getstruct("rgb_mean"),
                               config.getstruct("rgb_std"))
    test_db = ISSTestCSVDatatset(input_csv_path, test_tf)
    test_sampler = DistributedARBatchSampler(test_db, config.getint("val_batch_size"), world_size, rank, False)
    test_dl = data.DataLoader(test_db,
                              batch_sampler=test_sampler,
                              collate_fn=iss_collate_fn,
                              pin_memory=True,
                              num_workers=config.getint("num_workers"))

    return test_dl


def _load_meta(meta_file):
    with open(meta_file, "rb") as fid:
        data = umsgpack.load(fid, encoding="utf-8")
        meta = data["meta"]
    return meta


def _make_model(config, num_thing, num_stuff):
    body_config = config["body"]
    fpn_config = config["fpn"]
    rpn_config = config["rpn"]
    roi_config = config["roi"]
    sem_config = config["sem"]
    classes = {"total": num_thing + num_stuff, "stuff": num_stuff, "thing": num_thing}

    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)

    # Create backbone
    _log_debug("Creating backbone model %s", body_config["body"])
    body_fn = models.__dict__["net_" + body_config["body"]]
    body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}
    body = body_fn(norm_act=norm_act_static, **body_params)

    body_channels = body_config.getstruct("out_channels")

    # Create FPN
    fpn_inputs = fpn_config.getstruct("inputs")
    fpn = FPN([body_channels[inp] for inp in fpn_inputs],
              fpn_config.getint("out_channels"),
              fpn_config.getint("extra_scales"),
              norm_act_static,
              fpn_config["interpolation"])
    body = FPNBody(body, fpn, fpn_inputs)

    # Create RPN
    proposal_generator = ProposalGenerator(rpn_config.getfloat("nms_threshold"),
                                           rpn_config.getint("num_pre_nms_train"),
                                           rpn_config.getint("num_post_nms_train"),
                                           rpn_config.getint("num_pre_nms_val"),
                                           rpn_config.getint("num_post_nms_val"),
                                           rpn_config.getint("min_size"))
    anchor_matcher = AnchorMatcher(rpn_config.getint("num_samples"),
                                   rpn_config.getfloat("pos_ratio"),
                                   rpn_config.getfloat("pos_threshold"),
                                   rpn_config.getfloat("neg_threshold"),
                                   rpn_config.getfloat("void_threshold"))
    rpn_loss = RPNLoss(rpn_config.getfloat("sigma"))
    rpn_algo = RPNAlgoFPN(
        proposal_generator, anchor_matcher, rpn_loss,
        rpn_config.getint("anchor_scale"), rpn_config.getstruct("anchor_ratios"),
        fpn_config.getstruct("out_strides"), rpn_config.getint("fpn_min_level"), rpn_config.getint("fpn_levels"))
    rpn_head = RPNHead(
        fpn_config.getint("out_channels"), len(rpn_config.getstruct("anchor_ratios")), 1,
        rpn_config.getint("hidden_channels"), norm_act_dynamic)

    # Create instance segmentation network
    bbx_prediction_generator = BbxPredictionGenerator(roi_config.getfloat("nms_threshold"),
                                                      roi_config.getfloat("score_threshold"),
                                                      roi_config.getint("max_predictions"))
    msk_prediction_generator = MskPredictionGenerator()
    roi_size = roi_config.getstruct("roi_size")
    proposal_matcher = ProposalMatcher(classes,
                                       roi_config.getint("num_samples"),
                                       roi_config.getfloat("pos_ratio"),
                                       roi_config.getfloat("pos_threshold"),
                                       roi_config.getfloat("neg_threshold_hi"),
                                       roi_config.getfloat("neg_threshold_lo"),
                                       roi_config.getfloat("void_threshold"))
    bbx_loss = DetectionLoss(roi_config.getfloat("sigma"))
    msk_loss = InstanceSegLoss()
    lbl_roi_size = tuple(s * 2 for s in roi_size)
    roi_algo = InstanceSegAlgoFPN(
        bbx_prediction_generator, msk_prediction_generator, proposal_matcher, bbx_loss, msk_loss, classes,
        roi_config.getstruct("bbx_reg_weights"), roi_config.getint("fpn_canonical_scale"),
        roi_config.getint("fpn_canonical_level"), roi_size, roi_config.getint("fpn_min_level"),
        roi_config.getint("fpn_levels"), lbl_roi_size, roi_config.getboolean("void_is_background"))
    roi_head = FPNMaskHead(fpn_config.getint("out_channels"), classes, roi_size, norm_act=norm_act_dynamic)

    # Create semantic segmentation network
    sem_loss = SemanticSegLoss(ohem=sem_config.getfloat("ohem"))
    sem_algo = SemanticSegAlgo(sem_loss, classes["total"])
    sem_head = FPNSemanticHeadDeeplab(fpn_config.getint("out_channels"),
                                      sem_config.getint("fpn_min_level"),
                                      sem_config.getint("fpn_levels"),
                                      classes["total"],
                                      pooling_size=sem_config.getstruct("pooling_size"),
                                      norm_act=norm_act_static)

    # Create final network
    return PanopticNet(body, rpn_head, roi_head, sem_head, rpn_algo, roi_algo, sem_algo, classes)


def _test(model, dataloader, label_function=lambda x: x, **varargs):
    model.eval()
    dataloader.batch_sampler.set_epoch(0)

    data_time_meter = AverageMeter(())
    batch_time_meter = AverageMeter(())

    make_panoptic = varargs["make_panoptic"]
    num_stuff = varargs["num_stuff"]

    data_time = time.time()
    labels = []
    for it, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            # Extract data
            img = batch["img"].cuda(device=varargs["device"], non_blocking=True)

            data_time_meter.update(torch.tensor(time.time() - data_time))

            batch_time = time.time()

            # Run network
            _, pred, _ = model(img=img, do_loss=False, do_prediction=True)

            # Update meters
            batch_time_meter.update(torch.tensor(time.time() - batch_time))

            for i, (sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred) in enumerate(zip(
                    pred["sem_pred"], pred["bbx_pred"], pred["cls_pred"], pred["obj_pred"], pred["msk_pred"])):
                # Compute panoptic output
                panoptic_pred = make_panoptic(sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, num_stuff)

                # Save prediction
                raw_pred = (sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred)
                labels.append(label_function(raw_pred))

            # Log batch
            if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
                logging.iteration(
                    None, "val", 0, 1, 1,
                    it + 1, len(dataloader),
                    OrderedDict([
                        ("data_time", data_time_meter),
                        ("batch_time", batch_time_meter)
                    ])
                )

            data_time = time.time()
    return labels


if __name__ == "__main__":
    fire.Fire(binary_classification)
