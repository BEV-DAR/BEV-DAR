# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta
from mmcv.runner import BaseModule, force_fp32
from ..builder import HEADS
from ..losses import iou
import pdb
from mmcv.cnn import ConvModule
from mmseg.ops import resize
import numpy as np
from PIL import Image
import math

class RayGenerator:
    def __init__(self, image_size=(200, 200)):
        self.image_size = image_size
        self.center = (100, 0)
    
    def bresenham_line(self, start, end):
        """Bresenham's line algorithm to find points along the line."""
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        err = dx - dy

        points = []
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def find_intersection(self, x0, y0, angle):
        """Find the intersection point of the ray with the boundaries."""
        left, right = 0, self.image_size[1]-1
        top, bottom = 0, self.image_size[0]-1

        if angle == 90:  # Vertical line
            return (x0, bottom)

        angle_rad = math.radians(angle)
        tan_angle = math.tan(angle_rad)

        # Intersection with left boundary
        if tan_angle != float('inf'):
            x1, y1 = left, y0 + (left - x0) * tan_angle
            if top <= y1 <= bottom:
                return (x1, int(y1))

        # Intersection with right boundary
        x1, y1 = right, y0 + (right - x0) * tan_angle
        if top <= y1 <= bottom:
            return (x1, int(y1))

        # Intersection with top boundary
        if tan_angle != 0 and y0 > top:
            x1, y1 = x0 + (top - y0) / tan_angle, top
            if left <= x1 <= right:
                return (int(x1), y1)

        # Intersection with bottom boundary
        if tan_angle != 0:
            x1, y1 = x0 + (bottom - y0) / tan_angle, bottom
            if left <= x1 <= right:
                return (int(x1), y1)

        return None

    def generate_rays(self, angle_range):
        """Generate rays for a given range of angles."""
        line_index = []

        for angle in angle_range:
            end_point = self.find_intersection(*self.center, angle)
            if end_point:
                end_point = (int(end_point[0]), int(end_point[1]))
                line_points = self.bresenham_line(self.center, end_point)
                line_index.append(line_points)

        return line_index

def prior_uncertainty_loss(x, mask, priors):
    # priors shape: [14]-->[1,14,1,1]-->[bs,14,196,200]
    priors = x.new(priors).view(1, -1, 1, 1).expand_as(x)
    # F.binary_cross_entropy_with_logits(x, priors, reduce=False) return a tensor with the shape of x, i.e. [bs,14,196,200]
    xent = F.binary_cross_entropy_with_logits(x, priors, reduce=False)
    return (xent * (~mask).float().unsqueeze(1)).mean()

def balanced_binary_cross_entropy(logits, labels, mask, weights):
    # weights shape: [14]-->[14,1,1]-->[bs,14,196,200]
    weights = weights.to(logits)
    # print(weights.shape,logits.shape)
    weights = (logits.new(weights).view(-1, 1, 1) - 1) * labels.float() + 1.
    weights = weights * mask.unsqueeze(1).float()
    return F.binary_cross_entropy_with_logits(logits, labels.float(), weights)

def balanced_ray_cross_entropy(logits, labels, mask, weights):
    # weights shape: [14]-->[14,1]-->[bs,14,num_points]
    weights = weights.to(logits)
    # print(weights.shape,logits.shape)
    weights = (logits.new(weights).view(-1, 1) - 1) * labels.float() + 1.
    weights = weights * mask.unsqueeze(1).float()
    return F.binary_cross_entropy_with_logits(logits, labels.float(), weights)

class OccupancyCriterion(nn.Module):

    def __init__(self, priors=[0.44679, 0.02407, 0.14491, 0.02994, 0.02086, 0.00477, 0.00156, 0.00189, 0.00084, 0.00119, 0.00019, 0.00012, 0.00031, 0.00176],
                 xent_weight=1., uncert_weight=0.001,
                 weight_mode='sqrt_inverse'):
        super().__init__()

        self.xent_weight = xent_weight
        self.uncert_weight = 0

        self.priors = torch.tensor(priors)

        if weight_mode == 'inverse':
            self.class_weights = 1 / self.priors
        elif weight_mode == 'sqrt_inverse':
            self.class_weights = torch.sqrt(1 / self.priors)
        elif weight_mode == 'equal':
            self.class_weights = torch.ones_like(self.priors)
        else:
            raise ValueError('Unknown weight mode option: ' + weight_mode)

    def forward(self, logits, labels, mask, *args):
        # logits shape: bs,15,196,200,labels shape: bs,14,196,200,mask shape:bs,196,200
        # Compute binary cross entropy loss
        self.class_weights = self.class_weights.to(logits)
        bce_loss = balanced_binary_cross_entropy(
            logits, labels, mask, self.class_weights)

        # Compute uncertainty loss for unknown image regions
        self.priors = self.priors.to(logits)


        return bce_loss * self.xent_weight + uncert_loss * self.uncert_weight


class LinearClassifier(nn.Conv2d):

    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes, 1)

    def initialise(self, prior):
        prior = torch.tensor(prior)
        self.weight.data.zero_()
        self.bias.data.copy_(torch.log(prior / (1 - prior)))


class TopdownNetwork(nn.Sequential):

    def __init__(self, in_channels, channels, layers=[6, 1, 1],
                 strides=[1, 2, 2], blocktype='basic'):
        modules = list()
        self.downsample = 1
        for nblocks, stride in zip(layers, strides):
            # Add a new residual layer
            module = ResNetLayer(
                in_channels, channels, nblocks, 1 / stride, blocktype=blocktype)
            modules.append(module)

            # Halve the number of channels at each layer
            in_channels = module.out_channels
            channels = channels // 2
            self.downsample *= stride

        self.out_channels = in_channels

        super().__init__(*modules)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""

    # Fractional strides correspond to transpose convolution
    if stride < 1:
        stride = int(round(1 / stride))
        kernel_size = stride + 2
        padding = int((dilation * (kernel_size - 1) - stride + 1) / 2)
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size, stride, padding,
            output_padding=0, dilation=dilation, bias=False)

    # Otherwise return normal convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=int(stride),
                     dilation=dilation, padding=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""

    # Fractional strides correspond to transpose convolution
    if int(1 / stride) > 1:
        stride = int(1 / stride)
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=stride, stride=stride, bias=False)

    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=int(stride), bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.GroupNorm(16, planes)

        self.conv2 = conv3x3(planes, planes, 1, dilation)
        self.bn2 = nn.GroupNorm(16, planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), nn.GroupNorm(16, planes))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.GroupNorm(16, planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.GroupNorm(16, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.GroupNorm(16, planes * self.expansion)

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                nn.GroupNorm(16, planes * self.expansion))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class ResNetLayer(nn.Sequential):

    def __init__(self, in_channels, channels, num_blocks, stride=1,
                dilation=1, blocktype='bottleneck'):

        # Get block type
        if blocktype == 'basic':
            block = BasicBlock
        elif blocktype == 'bottleneck':
            block = Bottleneck
        else:
            raise Exception("Unknown residual block type: " + str(blocktype))

        # Construct layers
        layers = [block(in_channels, channels, stride, dilation)]
        for _ in range(1, num_blocks):
            layers.append(block(channels * block.expansion, channels, 1, dilation))

        self.in_channels = in_channels
        self.out_channels = channels * block.expansion

        super(ResNetLayer, self).__init__(*layers)
# def save_image(arr, filepath):
#     """Save a NumPy array as an image.
    
#     Args:
#         arr (np.ndarray): Array of shape [C, H, W].
#         filepath (str): Path to save the image.
#     """
#     # 将数组从 [C, H, W] 转换为 [H, W, C]
#     arr = np.transpose(arr.cpu(), (1, 2, 0)).astype(np.uint8)
#     # 创建 PIL Image 对象
#     img = Image.fromarray(arr)
#     # 保存图像
#     img.save(filepath)
def save_image(arr, filepath):
    """Save a NumPy array as an image.
    
    Args:
        arr (np.ndarray): Array of shape [C, H, W].
        filepath (str): Path to save the image.
    """
    # 将数组从 [C, H, W] 转换为 [H, W, C]
    arr = arr.cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    # 创建 PIL Image 对象
    img = Image.fromarray(arr)
    # 保存图像
    img.save(filepath)


def binary_cross_entropy_with_logits(logits, labels):
    """Compute binary cross entropy loss between logits and binary labels."""
    return F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

@HEADS.register_module()
class DAR_head(BaseModule, metaclass=ABCMeta):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,num_classes,align_corners=True, **kwargs ):
        super(DAR_head, self).__init__(**kwargs)

        # print('sad')
        # Build topdown network
        self.topdown = TopdownNetwork(64, 128, [4,4], [1,2], 'bottleneck')
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.classifier = LinearClassifier(self.topdown.out_channels, self.num_classes)
        if self.num_classes == 14:
            self.angle_range = np.arange(56, 124, 0.5)
            self.classifier.initialise([0.44679, 0.02407, 0.14491, 0.02994, 0.02086, 0.00477, 0.00156, 0.00189,
                                    0.00084, 0.00119, 0.00019, 0.00012, 0.00031, 0.00176])
            self.priors=[0.44679, 0.02407, 0.14491, 0.02994, 0.02086, 0.00477, 0.00156, 0.00189, 0.00084, 0.00119, 0.00019, 0.00012, 0.00031, 0.00176]
        else:
            self.angle_range = np.arange(55, 125, 0.5)
            self.classifier.initialise([0.34602, 0.03698, 0.00207, 0.01085, 0.00243, 0.01085, 0.02041, 0.00132])
            self.priors=[0.34602, 0.03698, 0.00207, 0.01085, 0.00243, 0.01085, 0.02041, 0.00132]
        # Build classifier


        self.criterion = OccupancyCriterion(self.priors)

    def forward(self, inputs):
        """Forward function."""
        # Apply topdown network
        td_feats = self.topdown(inputs)

        # Predict individual class log-probabilities
        logits = self.classifier(td_feats)
        return logits

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)

        losses = self.bresenham_losses(seg_logits, gt_semantic_seg,self.priors)
        # seg_logits.shape :[batch_size,14,196,200]
        # gt_semantic_seg.shape :[batch_size,1,15,196,200]
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        seg_label = seg_label.squeeze(1).bool()
        loss = dict()
        loss['acc_seg'] = iou(seg_logit.detach().sigmoid()>0.5, seg_label[:,:-1,...], seg_label[:,-1,...])
        loss['loss_seg'] = self.criterion(seg_logit,seg_label[:,:-1,...],seg_label[:,-1,...])
        return loss
    def remove_duplicates(self,lines):
        """
        Remove duplicate points in the lines and keep only the first occurrence.

        Args:
        lines (list): A list of lists representing lines as [[(x0, y0), (x1, y1)], ...].

        Returns:
        list: The updated list with duplicates removed.
        """
        # Dictionary to track the first occurrence of each point
        seen_points = {}
        updated_lines = []

        for line in lines:
            updated_line = []
            for point in line:
                # Add the point if it hasn't been seen before
                if point not in seen_points:
                    seen_points[point] = True
                    updated_line.append(point)
            updated_lines.append(updated_line)

        return updated_lines
    def l1_loss(predictions, targets):
     """Compute L1 loss between predictions and targets."""
     return torch.abs(predictions - targets).mean(dim=-1)  # 计算每个样本的 L1 损失
    @force_fp32(apply_to=('seg_logit', ))
    def bresenham_losses(self, seg_logit, seg_label,priors):
        # priors=[0.44679, 0.02407, 0.14491, 0.02994, 0.02086, 0.00477, 0.00156, 0.00189, 0.00084, 0.00119, 0.00019, 0.00012, 0.00031, 0.00176]
        priors = torch.tensor(priors)

        weights = torch.sqrt(1 / priors)
        # print(seg_logit.shape,seg_label.shape)
        b,c,w,h = seg_logit.shape
        """Compute segmentation loss."""
        ray_generator = RayGenerator(image_size=(200,200))
        rays = ray_generator.generate_rays(self.angle_range)
        rays = self.remove_duplicates(rays)
        # print(len(rays))
        batch_size = seg_label.shape[0]
        padding_zeros_logits = torch.zeros([batch_size, c, 4, h],device = 'cuda')
        padding_zeros_gt = torch.zeros([batch_size, 1, c+1, 4, h],device = 'cuda')
        losses_per_ray = []
        seg_logits = torch.cat([padding_zeros_logits, seg_logit], dim=2)
        gt_semantic_seg = torch.cat([padding_zeros_gt, seg_label], dim=3)
        
        for ray in rays:
            # 提取射线上的像素值
            ray_logits = []
            ray_labels = []
            for p in ray:
                ray_logits.append(seg_logits[:, :, p[1], p[0]] )
                ray_labels.append(gt_semantic_seg[:, :, :, p[1], p[0]])

            # 将提取的像素值堆叠成张量
            ray_logits = torch.stack(ray_logits, dim=-1)  # [batch_size, 14, num_points]
            ray_labels = torch.stack(ray_labels, dim=-1)  # [batch_size, 1, 15, num_points]
            ray_labels = ray_labels.squeeze(1)
            ray_labels = ray_labels[:,:-1,...]
            label_one_mask = ray_labels[:,-1,...]
            ray_losses = []
            # 计算损失
       
            loss = balanced_ray_cross_entropy(ray_logits, ray_labels,label_one_mask,weights)
            ray_losses.append(loss)

            ray_loss = sum(ray_losses) / len(ray_losses)  # 平均损失
            losses_per_ray.append(ray_loss)
        losses_per_ray = torch.stack(losses_per_ray, dim=-1)
         # 选择前 1-60 条最大的损失
        k = 100
        top_k_losses, _ = torch.topk(losses_per_ray, k=k, dim=-1)
        avg_top_k_loss = top_k_losses.mean(dim=-1)

        # return avg_top_k_loss
        # 沿着第二维（高度）拼接张量

        
        seg_label = seg_label.squeeze(1).bool()


        loss = dict()
        iou_loss = iou(seg_logit.detach().sigmoid()>0.5, seg_label[:,:-1,...], seg_label[:,-1,...])

        loss['acc_seg'] =  iou_loss
        loss['avg_top_k_loss'] = avg_top_k_loss * 10
        loss['loss_seg'] = self.criterion(seg_logit,seg_label[:,:-1,...],seg_label[:,-1,...])

        # print(loss) 
        return loss
