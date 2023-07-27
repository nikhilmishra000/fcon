import json
import os.path as osp

import attr
import numpy as np
import torch
import torch.nn.functional as F

from torch_utils import to_torch


@attr.s(frozen=True, kw_only=True, repr=False)
class COB3D(torch.utils.data.Dataset):
    root: str = attr.ib()
    scenes: dict[str, list[str]] = attr.ib()
    target_scale: float = attr.ib()

    @classmethod
    def load(cls, root: str, **kw) -> "COB3D":
        with open(osp.join(root, "dset.json")) as f:
            split = json.load(f)

        return cls(root=root, scenes=split, **kw)

    def __getitem__(self, scene_id: str):
        scene_path = osp.join(self.root, "scenes", f"{scene_id}.npz")
        f = np.load(scene_path, mmap_mode="r", allow_pickle=True)

        segm = f["segm"].item()
        visible_frac = segm["masks"].sum((1, 2)) / segm["amodal_masks"].sum((1, 2))
        keep_obj = visible_frac > 0.1
        item = {
            "rgb": f["rgb"],
            "intrinsic": f["intrinsic"],
            "depth_map": f["depth_map"],
            "boxes": segm["boxes"][keep_obj],
            "masks": segm["masks"][keep_obj],
            "amodal_masks": segm["amodal_masks"][keep_obj],
            "voxel_grid": {k: v[keep_obj] for k, v in f["voxel_grid"].item().items()},
            "obj_poses": {k: v[keep_obj] for k, v in f["obj_poses"].item().items()},
            "near_plane": f["near_plane"],
            "far_plane": f["far_plane"],
        }
        item = to_torch(item, recursive=True)

        h, w = item["rgb"].shape[-2:]
        scale_factor = self.target_scale / np.sqrt(h * w)

        item["rgb"] = F.interpolate(
            item["rgb"][None],
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )[0]
        item["intrinsic"][0:2] *= scale_factor
        item["depth_map"] = F.interpolate(
            item["depth_map"][None, None], scale_factor=scale_factor, mode="nearest"
        )[0, 0]
        item["boxes"] *= scale_factor
        item["masks"] = F.interpolate(
            item["masks"][:, None].byte(), scale_factor=scale_factor, mode="nearest"
        )[:, 0].bool()
        item["amodal_masks"] = F.interpolate(
            item["amodal_masks"][:, None].byte(),
            scale_factor=scale_factor,
            mode="nearest",
        )[:, 0].bool()
        return item
