# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
from collections import OrderedDict
import torch
import numpy as np
import os, json, cv2, random
import matplotlib
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
# dist.init_process_group('gloo', init_method='file:///temp/somefile', rank=0, world_size=1)


import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator


# ########################################## ########################################## #
class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def resume_or_load(self, resume=True):
        if not isinstance(self.checkpointer, AdetCheckpointer):
            # support loading a few other backbones
            self.checkpointer = AdetCheckpointer(
                self.model,
                self.cfg.OUTPUT_DIR,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
        super().resume_or_load(resume=resume)

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # if evaluator_type == "pascal_voc":
        #     return PascalVOCDetectionEvaluator(dataset_name)
        # if evaluator_type == "lvis":
        #     return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


# # ########################################################################################## #
def plain_register_dataset():
    
    from detectron2.data.datasets import register_coco_instances
    
    register_coco_instances("particle_saa_demo_coco_train", {},
                            "../../Datasets/particle_saa_demo_coco/annotations/instances_train.json",
                            "../../Datasets/particle_saa_demo_coco/images/train")
    register_coco_instances("particle_saa_demo_coco_val", {},
                            "../../Datasets/particle_saa_demo_coco/annotations/instances_val.json",
                            "../../Datasets/particle_saa_demo_coco/images/val")

     
    # register_coco_instances("particle_saa_demo_coco_train", {},
    #                         "../../Datasets/particle_saa_demo_coco_modified/annotations/instances_train.json",
    #                         "../../Datasets/particle_saa_demo_coco_modified/images/train")
    # register_coco_instances("particle_saa_demo_coco_val", {},
    #                         "../../Datasets/particle_saa_demo_coco_modified/annotations/instances_val.json",
    #                         "../../Datasets/particle_saa_demo_coco_modified/images/val")

    
    CLASS_NAMES = ['black', 'red']
    MetadataCatalog.get("particle_saa_demo_coco_train").set(thing_classes=CLASS_NAMES)
    MetadataCatalog.get("particle_saa_demo_coco_val").set(thing_classes=CLASS_NAMES)

    
    DatasetCatalog.get("particle_saa_demo_coco_train")
    DatasetCatalog.get("particle_saa_demo_coco_val")


# ################################################################################## #

# particle_demo_coco_metadata_train = MetadataCatalog.get("particle_demo_coco_train")
# dataset_dicts_train = DatasetCatalog.get("particle_demo_coco_train")
# for d in random.sample(dataset_dicts_train, 2):  
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=particle_demo_coco_metadata_train, scale=1)
#     out = visualizer.draw_dataset_dict(d)
#     # plt.imshow(out.get_image()[:, :, ::-1])
#     plt.figure(figsize=(12,12))
#     plt.imshow(out.get_image())
#     plt.show()



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    args.config_file = "../configs/BlendMask/R_50_3x.yaml"  
    
    cfg.merge_from_file(args.config_file)  
    cfg.merge_from_list(args.opts)  

    
    # ################################################ Datasets ################################################ #
    cfg.DATASETS.TRAIN = ("particle_saa_demo_coco_train",)  
    cfg.DATASETS.TEST = ("particle_saa_demo_coco_val",)


    # ################################################ Input ################################################ #

    cfg.INPUT.CROP.ENABLED = True  #  use Data Augmentation RandomCrop()
    cfg.INPUT.HFLIP_TRAIN = True  #  use Data Augmentation RandomFlip()

    cfg.INPUT.MAX_SIZE_TRAIN = 1024  
    cfg.INPUT.MIN_SIZE_TRAIN = (1024, 2048)  
    cfg.INPUT.MAX_SIZE_TEST = 1024  
    cfg.INPUT.MIN_SIZE_TEST = 1024  
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'  #  use Data Augmentation Resize_by_range
    
  
    cfg.TEST.DETECTIONS_PER_IMAGE = 500

    # ################################################ Model ################################################ #
    # cfg.MODEL.WEIGHTS = "../model/Blendmask_R_101_3x.pth"  
    
    cfg.MODEL.WEIGHTS = "../model/Blendmask_R_101_dcni3_5x.pth"  
    # cfg.MODEL.WEIGHTS = "../model/Blendmask_R_50_3x.pth"  
    # cfg.MODEL.WEIGHTS = "../model/Blendmask_X_101_32x8d_dcni2_5x.pth"  

    # cfg.MODEL.BASIS_MODULE.LOSS_ON = False 
    cfg.MODEL.RETINANET.NUM_CLASSES = 2  
    cfg.OUTPUT_DIR = "./output/"

    # ################################################ SOLVER ################################################ #
    cfg.SOLVER.IMS_PER_BATCH = 16
    ITERS_IN_ONE_EPOCH = int(672 / cfg.SOLVER.IMS_PER_BATCH)  
    MaxEpoch = 100*3  
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * MaxEpoch) - 1   

    # ################################################ LR Schedule ################################################ #
   
    cfg.SOLVER.BASE_LR = 0.0025*2  # 0.005
    
    cfg.SOLVER.MOMENTUM = 0.9
   
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0



    # 1 -- WarmupMultiStepLR (warmup+Step)
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"  
    cfg.SOLVER.GAMMA = 0.1  

    cfg.SOLVER.STEPS = (ITERS_IN_ONE_EPOCH*(MaxEpoch-15), ITERS_IN_ONE_EPOCH*(MaxEpoch-5))  # 672/16*(50)=2100, 672/16*(75)=3150

    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000  

    cfg.SOLVER.WARMUP_ITERS = ITERS_IN_ONE_EPOCH * 4  # 672/16*(4)=168
    cfg.SOLVER.WARMUP_METHOD = "linear"

    # # 2 -- WarmupCosineLR (no config param, warmup+Cosine)
    # cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"

    # # 3 -- WarmupConstantLR (warmup+Constant)
    # cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    # cfg.SOLVER.GAMMA = 0.1  
    # cfg.SOLVER.STEPS = (ITERS_IN_ONE_EPOCH*200, )

    # cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000  # default:0.001  # meaningless!!!

    # cfg.SOLVER.WARMUP_ITERS = ITERS_IN_ONE_EPOCH * 4  # 672/16*(4)=168
    # cfg.SOLVER.WARMUP_METHOD = "linear"



    # ################################################ Checkpoint ################################################ #

    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH * 10 - 1  # 672/16*10-1=419

    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH * 10  # 672/16*10=420

    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")
    return cfg


def main(args):
    cfg = setup(args)
    plain_register_dataset()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)  
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res


    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus = 2  
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


