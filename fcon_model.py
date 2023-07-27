import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from frustum import Frustum
from layers.conv_blobs import ConvBlobSpec2d, ConvBlobSpec3d
from layers.conv_layer import ConvArgs3d, ConvLayer, ConvLayerSpec, LayerOrder
from layers.nonlinearity import LeakyReluArgs
from layers.normalization import GroupNormArgs
from layers.residual_block import ResidualBlock, ResidualMode
from layers.unet import UNetSimple
from torch_utils import Lambda, enlarge_box_roi, expand_dim


@attr.s(kw_only=True, eq=False, repr=False)
class FCON(nn.Module):
    n_depth_bins: int = attr.ib()
    patch_size: int = attr.ib()

    def __attrs_post_init__(self):
        super().__init__()

        gn = lambda n: GroupNormArgs(num_per_group=n, affine=False)

        # 3D-UNet
        unet_channels = [32, 64, 128, 256]
        unet_spec = ConvBlobSpec3d(
            channels=unet_channels, strides=[2**i for i in range(len(unet_channels))]
        )
        conv_spec = ConvLayerSpec(
            layer_order=LayerOrder.NORM_NONLIN_CONV,
            nonlin_args=LeakyReluArgs(),
            norm_args=GroupNormArgs(num_groups=4),
            conv_args=ConvArgs3d(),
        )
        self.feature_module = nn.Sequential(
            ConvLayer(
                5, unet_spec.channels[0], spec=attr.evolve(conv_spec, norm_args=gn(1))
            ),
            UNetSimple(
                in_channels=unet_spec.channels[0],
                unet_spec=unet_spec,
                conv_spec=conv_spec,
            ),
            Lambda(lambda blobs: blobs[0]),
            ConvLayer(
                unet_spec.channels[0],
                1,
                spec=attr.evolve(conv_spec, norm_args=gn(4)).as_end_spec(),
            ),
        )

        # 2D-UNet
        unet_channels = [64, 128, 256, 512]
        unet_spec = ConvBlobSpec2d(
            channels=(self.n_depth_bins, *unet_channels),
            strides=(1, *[2**i for i in range(1, len(unet_channels) + 1)]),
        )
        conv_spec = ConvLayerSpec(
            layer_order=LayerOrder.NORM_NONLIN_CONV,
            nonlin_args=LeakyReluArgs(),
            norm_args=GroupNormArgs(num_groups=4),
        )
        self.unet2d = nn.Sequential(
            UNetSimple(
                in_channels=unet_spec.channels[0],
                unet_spec=unet_spec,
                conv_spec=conv_spec,
            ),
            Lambda(lambda blobs: blobs[0]),
        )

        # Final predictor/decoder module
        dconv2d = 128
        self.predictor = nn.Sequential(
            ConvLayer(
                self.n_depth_bins, dconv2d, spec=attr.evolve(conv_spec, norm_args=gn(1))
            ),
            ResidualBlock(
                dconv2d,
                [dconv2d, dconv2d],
                spec=attr.evolve(conv_spec, norm_args=gn(4)),
                mode=ResidualMode.GATED_RESIDUAL,
            ),
            ResidualBlock(
                dconv2d,
                [dconv2d, dconv2d],
                spec=attr.evolve(conv_spec, norm_args=gn(4)),
                mode=ResidualMode.GATED_RESIDUAL,
            ),
            ConvLayer(
                dconv2d,
                self.n_depth_bins,
                spec=ConvLayerSpec(
                    layer_order=LayerOrder.NORM_NONLIN_CONV
                ).as_end_spec(),
            ),
        )

    def _get_frustums(
        self,
        near_plane: torch.Tensor,
        far_plane: torch.Tensor,
        point_map: torch.Tensor,
        intrinsic: torch.Tensor,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        im_size: torch.Tensor,
        perturb: bool,
    ):
        boxes = enlarge_box_roi(boxes, im_size, 1.2)
        n_objects = len(boxes)
        near_planes = (
            expand_dim(point_map[2], dim=0, shape=[n_objects])
            .masked_fill(~masks, np.inf)
            .flatten(1, 2)
            .min(1)
            .values
        )
        far_planes = torch.zeros_like(near_planes).add(far_plane)

        # TODO (nikhil): un-magic these numbers
        # TODO (nikhil): make the perturb parameters relative
        near_plane_offset = 0.05
        far_plane_offset = 0.0
        if perturb:
            near_plane_offset = (
                torch.randn_like(near_planes)
                .mul(0.1)
                .add(near_plane_offset)
                .clip(0.05, 0.2)
            )
            far_plane_offset = (
                torch.rand_like(far_planes).sub(0.5).mul(0.2)
            )  # [-0.1, 0.1]

        near_planes = near_planes - near_plane_offset
        far_planes = far_planes - far_plane_offset

        return Frustum(
            roi=boxes,
            near_plane=near_planes,
            far_plane=far_planes,
            intrinsic=intrinsic,
        )

    def _voxelize_roi_point_clouds(
        self,
        frustums: Frustum,
        point_map: torch.Tensor,
    ):
        rois = F.pad(frustums.roi, (1, 0), value=0.0)

        # shape (n_objects, patch_size, patch_size)
        depth = torchvision.ops.roi_align(
            point_map[None, None, 2],
            rois,
            output_size=(self.patch_size, self.patch_size),
            sampling_ratio=2,
        ).squeeze(1)

        eps = 1e-3
        z_normed = (depth - frustums.near_plane[:, None, None]) / (
            frustums.far_plane[:, None, None] - frustums.near_plane[:, None, None]
        )
        z_endpoints = torch.linspace(0, 1, self.n_depth_bins + 1, device=rois.device)
        z_idx = torch.bucketize(z_normed.clamp(eps, 1 - eps), z_endpoints).sub(1)
        voxelized_roi_point_clouds = (
            F.one_hot(z_idx, self.n_depth_bins).permute(0, 3, 1, 2).bool()
        )
        return voxelized_roi_point_clouds

    def predict(
        self,
        rgb: torch.Tensor,
        intrinsic: torch.Tensor,
        point_map: torch.Tensor,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        near_plane: torch.Tensor,
        far_plane: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = rgb.device

        im_size = torch.tensor(rgb.shape[-2:], device=device)
        frustums = self._get_frustums(
            near_plane,
            far_plane,
            point_map,
            intrinsic,
            boxes,
            masks,
            im_size,
            perturb=False,
        )
        voxelized_roi_point_clouds = self._voxelize_roi_point_clouds(
            frustums, point_map
        )
        frustum_voxel_size = torch.tensor(
            [self.n_depth_bins, self.patch_size, self.patch_size], device=device
        )
        grid_pts_cam = frustums.to_grid(frustum_voxel_size)

        rois = F.pad(frustums.roi, (1, 0), value=0.0)

        rgb_p = torchvision.ops.roi_align(
            rgb[None],
            rois,
            output_size=(self.patch_size, self.patch_size),
            sampling_ratio=2,
        )

        mask_rois = torch.cat(
            [torch.arange(len(boxes), device=device)[:, None], boxes], dim=1
        )
        masks_p = torchvision.ops.roi_align(
            masks[:, None].float(),
            mask_rois,
            output_size=(self.patch_size, self.patch_size),
            sampling_ratio=2,
        )
        roi_feats = torch.cat(
            [
                rgb_p,
                masks_p,
                masks_p,
            ],
            dim=1,
        )

        roi_voxelized_features = torch.einsum(
            "bdhw,bchw->bcdhw", voxelized_roi_point_clouds, roi_feats
        )

        feats = self.feature_module(roi_voxelized_features).squeeze(1)
        feats = self.unet2d(feats)
        logits = self.predictor(feats)

        return {
            "logits": logits,
            "grid_centers": grid_pts_cam,
        }
