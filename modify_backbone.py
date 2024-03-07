import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# load a pre-trained model for classification and return
# only the features
# mobilenet_v2的资料在这里：https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
# 我们在这里只用它提取特征的部分，不用后面的分类器。
backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
# ``FasterRCNN`` needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280
# 通道数某种意义上可以理解成特征图的数量。
# 一个有意思的链接，关于多层特征图的卷积核：https://cloud.tencent.com/developer/news/323068

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
# 众所周知，Mask RCNN原论文的Anchor每个位置生成九个，3 x 3。
# 这也是可以修改的。
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'], # 池化使用的特征图
    output_size=7, # 见下
    sampling_ratio=2 # 见下
)

# output_size (int or Tuple[int, int]) – 输出大小，用 (height, width) 表示。
# spatial_scale (float) – 将输入坐标映射到框坐标的比例因子。默认值1.0。
# sampling_ratio (int) – 插值网格中用于计算每个合并输出bin的输出值的采样点数目。
# 如果> 0，则恰好使用sampling_ratio x sampling_ratio网格点。
# 如果<= 0，则使用自适应数量的网格点(计算为cell (roi_width / pooled_w)，同样计算高度)。
# 默认值1。

# put the pieces together inside a Faster-RCNN model
# 魔改大成
model = FasterRCNN(
    backbone,
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)