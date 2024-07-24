from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True  # random horizontal flip
_C.INPUT.CROP.CROP_INSTANCE = True  # random multi-scale crop

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()



_C.MODEL.FCOS.NUM_CLASSES = 2
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05

_C.MODEL.FCOS.NMS_TH = 0.6 # NMS Threshold, deafult 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000

_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 500
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 500
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True


_C.MODEL.FCOS.BOX_QUALITY = "ctrness"  


_C.MODEL.FCOS.THRESH_WITH_CTR = False



_C.MODEL.FCOS.LOSS_ALPHA = 0.75  
_C.MODEL.FCOS.LOSS_GAMMA = 2.0  




_C.MODEL.FCOS.LOSS_NORMALIZER_CLS = "fg"
_C.MODEL.FCOS.LOSS_WEIGHT_CLS = 1.0



_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False


_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True  
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'  
_C.MODEL.FCOS.YIELD_PROPOSAL = False
 
# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()
_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256

# ---------------------------------------------------------------------------- #
# DLA backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.DLA = CN()
_C.MODEL.DLA.CONV_BODY = "DLA34"
_C.MODEL.DLA.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.DLA.NORM = "FrozenBN"

# ---------------------------------------------------------------------------- #
# BAText Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BATEXT = CN()
_C.MODEL.BATEXT.VOC_SIZE = 96
_C.MODEL.BATEXT.NUM_CHARS = 25
_C.MODEL.BATEXT.POOLER_RESOLUTION = (8, 32)
_C.MODEL.BATEXT.IN_FEATURES = ["p2", "p3", "p4"]
_C.MODEL.BATEXT.POOLER_SCALES = (0.25, 0.125, 0.0625)
_C.MODEL.BATEXT.SAMPLING_RATIO = 1
_C.MODEL.BATEXT.CONV_DIM = 256
_C.MODEL.BATEXT.NUM_CONV = 2
_C.MODEL.BATEXT.RECOGNITION_LOSS = "ctc"
_C.MODEL.BATEXT.RECOGNIZER = "attn"
_C.MODEL.BATEXT.CANONICAL_SIZE = 96  # largest min_size for level 3 (stride=8)

# ---------------------------------------------------------------------------- #
# BlendMask Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BLENDMASK = CN()
_C.MODEL.BLENDMASK.ATTN_SIZE = 14
_C.MODEL.BLENDMASK.TOP_INTERP = "bilinear"
_C.MODEL.BLENDMASK.BOTTOM_RESOLUTION = 56
_C.MODEL.BLENDMASK.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO = 1
_C.MODEL.BLENDMASK.POOLER_SCALES = (0.25,)
_C.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT = 1.0
_C.MODEL.BLENDMASK.VISUALIZE = False

# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet" 
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8

_C.MODEL.BASIS_MODULE.NUM_CLASSES = 2
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3

# ---------------------------------------------------------------------------- #
# MEInst Head
# ---------------------------------------------------------------------------- #
_C.MODEL.MEInst = CN()


_C.MODEL.MEInst.NUM_CLASSES = 2
_C.MODEL.MEInst.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.MEInst.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.MEInst.PRIOR_PROB = 0.01
_C.MODEL.MEInst.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.MEInst.INFERENCE_TH_TEST = 0.05

_C.MODEL.MEInst.NMS_TH = 0.6
_C.MODEL.MEInst.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.MEInst.PRE_NMS_TOPK_TEST = 1000

_C.MODEL.MEInst.POST_NMS_TOPK_TRAIN = 500
_C.MODEL.MEInst.POST_NMS_TOPK_TEST = 500
_C.MODEL.MEInst.TOP_LEVELS = 2
_C.MODEL.MEInst.NORM = "GN"  # Support GN or none
_C.MODEL.MEInst.USE_SCALE = True


_C.MODEL.MEInst.THRESH_WITH_CTR = False


_C.MODEL.MEInst.LOSS_ALPHA = 0.25
_C.MODEL.MEInst.LOSS_GAMMA = 2.0
_C.MODEL.MEInst.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.MEInst.USE_RELU = True
_C.MODEL.MEInst.USE_DEFORMABLE = False
_C.MODEL.MEInst.LAST_DEFORMABLE = False
_C.MODEL.MEInst.TYPE_DEFORMABLE = "DCNv1"  # or DCNv2.


_C.MODEL.MEInst.NUM_CLS_CONVS = 4
_C.MODEL.MEInst.NUM_BOX_CONVS = 4
_C.MODEL.MEInst.NUM_SHARE_CONVS = 0
_C.MODEL.MEInst.CENTER_SAMPLE = True
_C.MODEL.MEInst.POS_RADIUS = 1.5
_C.MODEL.MEInst.LOC_LOSS_TYPE = 'diou'

# ---------------------------------------------------------------------------- #
# Mask Encoding
# ---------------------------------------------------------------------------- #

_C.MODEL.MEInst.MASK_ON = True

_C.MODEL.MEInst.IOU_THRESHOLDS = [0.5]
_C.MODEL.MEInst.IOU_LABELS = [0, 1]

_C.MODEL.MEInst.AGNOSTIC = True

_C.MODEL.MEInst.WHITEN = True
_C.MODEL.MEInst.SIGMOID = True


_C.MODEL.MEInst.NUM_MASK_CONVS = 4


_C.MODEL.MEInst.DIM_MASK = 60
_C.MODEL.MEInst.MASK_SIZE = 28

_C.MODEL.MEInst.PATH_COMPONENTS = "datasets/coco/components/" \
                                   "coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz"

_C.MODEL.MEInst.FLAG_PARAMETERS = False

_C.MODEL.MEInst.MASK_LOSS_TYPE = "mse"


_C.MODEL.MEInst.USE_GCN_IN_MASK = False
_C.MODEL.MEInst.GCN_KERNEL_SIZE = 9

_C.MODEL.MEInst.LOSS_ON_MASK = False

# ---------------------------------------------------------------------------- #
# CondInst Options
# ---------------------------------------------------------------------------- #
_C.MODEL.CONDINST = CN()


_C.MODEL.CONDINST.MASK_OUT_STRIDE = 4
_C.MODEL.CONDINST.BOTTOM_PIXELS_REMOVED = -1


_C.MODEL.CONDINST.MAX_PROPOSALS = -1

_C.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM = -1

_C.MODEL.CONDINST.MASK_HEAD = CN()
_C.MODEL.CONDINST.MASK_HEAD.CHANNELS = 8
_C.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS = 3
_C.MODEL.CONDINST.MASK_HEAD.USE_FP16 = False
_C.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS = False

_C.MODEL.CONDINST.MASK_BRANCH = CN()
_C.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS = 8
_C.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.CONDINST.MASK_BRANCH.CHANNELS = 128
_C.MODEL.CONDINST.MASK_BRANCH.NORM = "BN"
_C.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS = 4
_C.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON = False


_C.MODEL.BOXINST = CN()

_C.MODEL.BOXINST.ENABLED = False
_C.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED = 10

_C.MODEL.BOXINST.PAIRWISE = CN()
_C.MODEL.BOXINST.PAIRWISE.SIZE = 3
_C.MODEL.BOXINST.PAIRWISE.DILATION = 2
_C.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = 10000
_C.MODEL.BOXINST.PAIRWISE.COLOR_THRESH = 0.3

# ---------------------------------------------------------------------------- #
# TOP Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TOP_MODULE = CN()
_C.MODEL.TOP_MODULE.NAME = "conv"
_C.MODEL.TOP_MODULE.DIM = 16

# ---------------------------------------------------------------------------- #
# BiFPN options
# ---------------------------------------------------------------------------- #

_C.MODEL.BiFPN = CN()

_C.MODEL.BiFPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.OUT_CHANNELS = 160
_C.MODEL.BiFPN.NUM_REPEATS = 6

_C.MODEL.BiFPN.NORM = ""

# ---------------------------------------------------------------------------- #
# SOLOv2 Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SOLOV2 = CN()


_C.MODEL.SOLOV2.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_C.MODEL.SOLOV2.FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
_C.MODEL.SOLOV2.FPN_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_C.MODEL.SOLOV2.SIGMA = 0.2

_C.MODEL.SOLOV2.INSTANCE_IN_CHANNELS = 256
_C.MODEL.SOLOV2.INSTANCE_CHANNELS = 512

_C.MODEL.SOLOV2.NUM_INSTANCE_CONVS = 4
_C.MODEL.SOLOV2.USE_DCN_IN_INSTANCE = False
_C.MODEL.SOLOV2.TYPE_DCN = 'DCN'
_C.MODEL.SOLOV2.NUM_GRIDS = [40, 36, 24, 16, 12]


_C.MODEL.SOLOV2.NUM_CLASSES = 2
_C.MODEL.SOLOV2.NUM_KERNELS = 256
_C.MODEL.SOLOV2.NORM = "GN"
_C.MODEL.SOLOV2.USE_COORD_CONV = True
_C.MODEL.SOLOV2.PRIOR_PROB = 0.01


_C.MODEL.SOLOV2.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.SOLOV2.MASK_IN_CHANNELS = 256
_C.MODEL.SOLOV2.MASK_CHANNELS = 128
_C.MODEL.SOLOV2.NUM_MASKS = 256


_C.MODEL.SOLOV2.NMS_PRE = 500
_C.MODEL.SOLOV2.SCORE_THR = 0.1
_C.MODEL.SOLOV2.UPDATE_THR = 0.05
_C.MODEL.SOLOV2.MASK_THR = 0.5

_C.MODEL.SOLOV2.MAX_PER_IMG = 500

_C.MODEL.SOLOV2.NMS_TYPE = "matrix"

_C.MODEL.SOLOV2.NMS_KERNEL = "gaussian"
_C.MODEL.SOLOV2.NMS_SIGMA = 2


_C.MODEL.SOLOV2.LOSS = CN()
_C.MODEL.SOLOV2.LOSS.FOCAL_USE_SIGMOID = True
_C.MODEL.SOLOV2.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.SOLOV2.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT = 1.0
_C.MODEL.SOLOV2.LOSS.DICE_WEIGHT = 3.0



# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()


_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR" 
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"  

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001

_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1

_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"


_C.SOLVER.CHECKPOINT_PERIOD = 5000


_C.SOLVER.IMS_PER_BATCH = 16


_C.SOLVER.REFERENCE_WORLD_SIZE = 0


_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY


_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})

_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"

_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

_C.SOLVER.AMP = CN({"ENABLED": False})
