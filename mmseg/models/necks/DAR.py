import torch
import torch.nn as nn
from abc import ABCMeta
from mmcv.runner import BaseModule
from ..builder import NECKS
import torch.nn.functional as F
import pdb
from operator import mul
from functools import reduce
import math

def _make_grid(resolution, extents):
    # Create a grid of cooridinates in the birds-eye-view
    x1, z1, x2, z2 = extents
    zz, xx = torch.meshgrid(torch.arange(z1, z2, resolution), torch.arange(x1, x2, resolution))

    return torch.stack([xx, zz], dim=-1)


class TransformModule(nn.Module):
    def __init__(self, dim=8):
        super(TransformModule, self).__init__()
        self.dim = dim
        self.mat_list = nn.ModuleList()
        self.fc_transform = nn.Sequential(
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU(),
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU()
        )
    def forward(self, x):
        # shape x: B, C, H, W
        # x = self.bn(x)
        x = x.view(list(x.size()[:2]) + [self.dim * self.dim])
        view_comb = self.fc_transform(x)
        view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.dim, self.dim])
        return view_comb
# generate grids in BEV
class DenseTransformer(nn.Module):

    def __init__(self, in_channels, channels, resolution, grid_extents,
                 ymin, ymax, focal_length, groups=1):
        super().__init__()

        # Initial convolution to reduce feature dimensions
        self.conv = nn.Conv2d(in_channels, channels, 1)
        self.bn = nn.GroupNorm(16, channels)

        # Resampler transforms perspective features to BEV
        self.resampler = Resampler(resolution, grid_extents)

        # Compute input height based on region of image covered by grid
        self.zmin, zmax = grid_extents[1], grid_extents[3]
        self.in_height = math.ceil(focal_length * (ymax - ymin) / self.zmin)
        self.ymid = (ymin + ymax) / 2

        # Compute number of output cells required
        self.out_depth = math.ceil((zmax - self.zmin) / resolution)

        # Dense layer which maps UV features to UZ
        self.fc = nn.Conv1d(
            channels * self.in_height, channels * self.out_depth, 1, groups=groups
        )
        self.out_channels = channels

    def forward(self, features, calib, *args):
        # Crop feature maps to a fixed input height
        features = torch.stack([self._crop_feature_map(fmap, cal)
                                for fmap, cal in zip(features, calib)])

        # Reduce feature dimension to minimize memory usage
        features = F.relu(self.bn(self.conv(features)))

        # Flatten height and channel dimensions
        B, C, _, W = features.shape
        flat_feats = features.flatten(1, 2)
        bev_feats = self.fc(flat_feats).view(B, C, -1, W)

        # Resample to orthographic grid
        return self.resampler(bev_feats, calib)

    def _crop_feature_map(self, fmap, calib):
        # Compute upper and lower bounds of visible region
        focal_length, img_offset = calib[1, 1:]
        vmid = self.ymid * focal_length / self.zmin + img_offset
        vmin = math.floor(vmid - self.in_height / 2)
        vmax = math.floor(vmid + self.in_height / 2)

        # Pad or crop input tensor to match dimensions
        return F.pad(fmap, [0, 0, -vmin, vmax - fmap.shape[-2]])
def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats

class Resampler(nn.Module):

    def __init__(self, resolution, extents):
        super().__init__()

        # Store z positions of the near and far planes
        # extents[1]:zmin,extents[3]:zmax
        self.near = extents[1]
        self.far = extents[3]

        # Make a grid in the x-z plane
        self.grid = _make_grid(resolution, extents)

    def forward(self, features, calib):
        # Copy grid to the correct device
        self.grid = self.grid.to(features)

        # We ignore the image v-coordinate, and assume the world Y-coordinate
        # is zero, so we only need a 2x2 submatrix of the original 3x3 matrix
        # calib shape:[bs,3,3]-->[bs,2,3]-->[bs,2,2]-->[bs,1,1,2,2]
        calib = calib[:, [0, 2]][..., [0, 2]].view(-1, 1, 1, 2, 2)

        # Transform grid center locations into image u-coordinates
        cam_coords = torch.matmul(calib, self.grid.unsqueeze(-1)).squeeze(-1)

        # Apply perspective projection and normalize
        ucoords = cam_coords[..., 0] / cam_coords[..., 1]
        ucoords = ucoords / features.size(-1) * 2 - 1

        # Normalize z coordinates
        zcoords = (cam_coords[..., 1] - self.near) / (self.far - self.near) * 2 - 1

        # Resample 3D feature map
        grid_coords = torch.stack([ucoords, zcoords], -1).clamp(-1.1, 1.1)
        return F.grid_sample(features, grid_coords)


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None
class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
    
@NECKS.register_module()
class DAR(BaseModule, metaclass=ABCMeta):
    def __init__(self, use_light=False, downsample=32,  in_channels=768, bev_feature_channels=64, ogfH=1024, ogfW=1024,
                 grid_conf=dict(dbound=[1, 50, 1], xbound=[-25,25,0.5], zbound=[1, 50, 0.5], ybound=[-10,10,20]),resolution= 0.25 * reduce(mul, [1, 2]),
                 extents=[-25.0, 1.0, 25.0, 50.0], ymin=-2, ymax=4, focal_length=630.0):
        super(DAR, self).__init__()
        self.grid_conf = grid_conf
        
        self.use_light = use_light
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.downsample = downsample
        self.ogfH = ogfH
        self.ogfW = ogfW
        self.frustum = self.create_frustum()
        self.D ,_,_,_ = self.frustum.shape
        self.C = bev_feature_channels
        # by default, self.C = 64, self.D = 49
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, kernel_size=1, padding=0)
        self.use_quickcumsum = True

        # pyramid transformer
        self.use_light = use_light
        self.depth_list = [7,10,20,39,22]
        # pyramid transformer
        self.transformers = nn.ModuleList()
        DT_inchannels = [96,192,384,768]
        for i in range(4):
            # Scaled focal length for each transformer
            focal = focal_length / pow(2, i + 3)

            # Compute grid bounds for each transformer
            zmax = min(math.floor(focal * 2) * resolution, extents[3])
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]
            subset_extents = [extents[0], zmin, extents[2], zmax]
            # Build transformers
            tfm = DenseTransformer(DT_inchannels[i], bev_feature_channels, resolution,
                                subset_extents, ymin, ymax, focal)
            self.transformers.append(tfm)

    def create_frustum(self):
        fH, fW = self.ogfH//self.downsample, self.ogfW//self.downsample
        depth_samples = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        num_depth, _, _ = depth_samples.shape
        x_samples = torch.linspace(0, self.ogfW - 1, fW, dtype=torch.float).view(1,1,fW).expand(num_depth,fH,fW)
        y_samples = torch.linspace(0, self.ogfH - 1, fH, dtype=torch.float).view(1,fH,1).expand(num_depth,fH,fW)

        # D x H x W x 3
        frustum = torch.stack((x_samples,y_samples,depth_samples), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrinstics):
        B = intrinstics.shape[0]
        D,H,W,C = self.frustum.shape
        points = self.frustum.view(1,D,H,W,-1).expand(B,D,H,W,C)
        points = torch.cat([points[:,:,:,:,:2]* points[:,:,:,:,2:3],
                            points[:,:,:,:,2:3]],4)
        combine = torch.inverse(intrinstics)
        points = combine.view(B,1,1,1,3,3).matmul(points.unsqueeze(-1)).squeeze(-1).view(B,1,D,H,W,-1)
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)
        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (nx[1] * nx[2] * B) \
                + geom_feats[:, 1] * (nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[1], nx[2], nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 1], geom_feats[:, 2], geom_feats[:, 0]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def forward(self, feature_maps, intrinstics):
        # lss part
        # print(intrinstics)
        geom = self.get_geometry(intrinstics)  #geom.shape: bs,1,49,18,25,3
        b,_,_,h,w,_  = geom.shape
        if self.downsample==32:
            feature = feature_maps[3][:,:,:h,:w]   # feature.shape:bs,768,18,25
        elif self.downsample==16:
            feature = feature_maps[0][:, :, :h, :w]
        else:
            assert False
        feature = self.depthnet(feature)   # feature.shape:bs,113,18,25
        depth = self.get_depth_dist(feature[:, :self.D])     # depth.shape: bs,49,18,25
        new_feature = depth.unsqueeze(1) * feature[:, self.D:(self.D + self.C)].unsqueeze(2)  # new_feature.shape: bs,64,49,18,25
        # print('h,w',h,w)
        feature = new_feature.view(b, 1, self.C, self.D, h, w)  # feature.shape: bs,1,64,49,18,25
        feature = feature.permute(0, 1, 3, 4, 5, 2)  # feature.shape: bs,1,49,18,25,64
        feature_lss = self.voxel_pooling(geom, feature)
        # print('feature_lss.shape',feature_lss.shape)#feature_lss.shape torch.Size([12, 64, 98, 100])
        # pon+ray attention

        bev_feats = list()
        # scale = 8,16,32,64,128
        # calib.shape = [bs,3,3]
        for i, fmap in enumerate(feature_maps):
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2 ** i
            calib_downsamp = intrinstics.clone()
            calib_downsamp[:, :2] = intrinstics[:, :2] / scale
            temp = self.transformers[i](fmap, calib_downsamp)
            # Apply orthographic transformation to each feature map separately
            bev_feats.append(temp)
        
        bev_feats_pon = torch.cat(bev_feats[::-1], dim=-2)
        # print('bev_feats_pon',bev_feats_pon.shape)
        # print('feature_lss',feature_lss.shape)
        bev_feats_pon = F.interpolate(bev_feats_pon, size=(98, 100), mode='bilinear', align_corners=True)
        feature_final = bev_feats_pon + feature_lss
        return feature_final,bev_feats_pon,feature_lss
