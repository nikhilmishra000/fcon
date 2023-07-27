from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import np_utils as npu


DeviceLike = str | torch.device


def to_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x

    elif torch.is_tensor(x):
        return x.detach().cpu().numpy()

    else:
        return np.asarray(x)


def to_torch(
    x: Any,
    device: DeviceLike | None = None,
    dtype: torch.dtype | None = None,
    recursive: bool = False,
    strict: bool = False,
    preserve_doubles: bool = False,
) -> Any:
    """Map input to torch.

    When `recursive == False`, the following conversions are applied.
      1. Tensors:
        - We call `.to(device, dtype)` using the specified values.
      2. Numpy arrays of a numeric type (booleans, integers, floating-point numbers):
        - Unless `preserve_doubles` is set, float64 is cast to float32 (it is rare for doubles to be used in torch).
        - Any signed integers dtypes (int8, int16, int32) are cast to int64 (which is also preferred in torch).
        - Convert the cast array to a tensor, then apply (1).
      3. Numpy arrays of non-numeric types (strings, objects):
        - These are ignored if `strict == False`, otherwise an error is raised.
      4. `DeviceTransferable` objects:
        - We call their `.to()` method, mimicking (1).
      5. Other types:
        - If `strict == True`, an error is raised.
        - If `strict == False`, we return the value as is.

    If `recursive == True`, then the input can be a nested structure (list/tuple/set/dict).
    We traverse the structure and apply the non-recursive conversion rules to each leaf.

    Parameters
    ----------
    x : Any
    device : Optional[DeviceLike]
        If supplied, all tensors will be put onto this device.
    dtype : Optional[torch.dtype]
        If supplied, all tensors will be casted to this dtype.
    recursive : bool
        If True, `x` can be a nested structure, and we will traverse it.
    strict : bool
        If True, and we encounter types that cannot be converted to tensor, an error will be raised.
    preserve_doubles : bool
        If False (default), float64 arrays will be cast to float32.
        If True, the original dtype will be preserved. This option may become the default in the future.

    Returns
    -------
    Any
        A tensor, or a structure identical to `x`, with the above conversions applied.
    Raises
    ------
    ValueError
        If `strict == True` and we cannot convert a value to a tensor.
    """
    if recursive:
        return npu.map_until(
            lambda z: to_torch(
                z, device=device, dtype=dtype, recursive=False, strict=strict
            ),
            x,
        )
    else:
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)

        elif any(
            isinstance(x, primitive_type) for primitive_type in (bool, int, float)
        ):
            return x

        else:
            x_array = to_np(x)
            x_dtype = x_array.dtype
            if np.issubdtype(x_dtype, np.number) or np.issubdtype(x_dtype, bool):
                if np.issubdtype(x_dtype, np.floating):
                    if x_dtype == np.float64 and not preserve_doubles:
                        x_array = x_array.astype(np.float32)

                elif np.issubdtype(x_dtype, np.signedinteger):
                    x_array = x_array.astype(np.int64)

                if x_array.size > 0:
                    return torch.from_numpy(np.ascontiguousarray(x_array)).to(
                        device=device, dtype=dtype
                    )
                else:
                    return torch.from_numpy(np.copy(x_array)).to(
                        device=device, dtype=dtype
                    )

            elif strict:
                raise ValueError(
                    "Input must be a numpy array of kind boolean, signed integer, unsigned integer, or floating-point!"
                )
            else:
                return x


@torch.jit.script_if_tracing
def expand_dim(
    x: torch.Tensor,
    dim: int,
    shape: Union[torch.Tensor, list[int]],
) -> torch.Tensor:
    """Expand a tensor to have extra dimensions, tiling the values without a copy.

    Parameters
    ----------
    x : torch.Tensor
    dim : int
        Insert the new dimensions at this index in `x.shape`.
    shape : Union[torch.Tensor, List[int]]
        The inserted dimensions will have this shape.

    Returns
    -------
    torch.Tensor
    """
    if dim < 0:
        dim = x.ndim + 1 + dim

    if isinstance(shape, torch.Tensor):
        expanded_shape = torch.Size([int(s.item()) for s in shape])
    else:
        expanded_shape = torch.Size(shape)

    single_shape = (
        x.shape[:dim] + torch.Size([1 for _ in expanded_shape]) + x.shape[dim:]
    )
    full_shape = x.shape[:dim] + expanded_shape + x.shape[dim:]
    return x.view(single_shape).expand(full_shape)


def enlarge_box_roi(
    boxes: torch.Tensor,
    im_size: torch.Tensor,
    enlargement_factor: float,
) -> torch.Tensor:
    """
    Enlarge the 2D bounding boxes of a set of objects.

    This is useful for certain models to obtain more information about the area surrounding the object of interest.

    Inputs
    ------
    boxes : Tensor[DeviceT, Float32, SHAPE_K4]
        The 2D bounding boxes of the scene.
    im_size : Tensor[DeviceT, Float32, SHAPE_2]
        The size of the entire image, expressed as (height, with).
    enlargement_factor (default = 1.2) : float
        The factor by which to enlarge each 2D bounding box.

    Returns
    -------
    boxes : Tensor[DeviceT, Float32, SHAPE_K4]
        The enlarged boxes.

    """
    centers = 0.5 * (boxes[:, 0:2] + boxes[:, 2:4])
    halfsizes = 0.5 * (boxes[:, 2:4] - boxes[:, 0:2])
    new_halfsizes = enlargement_factor * halfsizes

    return torch.cat(
        [
            torch.clamp(centers - new_halfsizes, min=0),
            torch.clamp(centers + new_halfsizes, max=im_size.flip(-1).float()),
        ],
        dim=-1,
    )


def depth2cloud(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """Convert depth image to XYZ point cloud.

    Args
    ----
        depth : Tensor[DeviceT, FloatT, SHAPE_HW]
            Contains depth value for every pixel in the image.
        intrinsic : Tensor[DeviceT, FloatT, SHAPE_33]
            Master camera intrinsic.

    Returns
    -------
        pts : Tensor[DeviceT, FloatT, SHAPE_HW3]
            Point cloud.
    """
    device = depth.device
    height, width = int(depth.size(-2)), int(depth.size(-1))
    vu = torch.stack(
        torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        ),
        dim=0,
    )
    uv1 = F.pad(vu.flip(0), (0, 0, 0, 0, 0, 1), value=1)
    pts = torch.einsum("ij,jhw->hwi", intrinsic.inverse(), uv1 * depth.unsqueeze(0))
    return pts


def interp3d(x: torch.Tensor, p: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Implement a non-batched version of `batch_interp3d()`.

    This function exists for legacy support, and might be removed in the near future.

    Parameters
    ----------
    x : torch.Tensor
        Shape (C, D, H, W), any dtype.
    p : torch.Tensor
        Shape (..., 3), dtype float32.
    mode : str
        One of {"bilinear", "nearest"}.

    Returns
    -------
    torch.Tensor
        Interpolated values from `x`, dtype `x.dtype` and shape(..., C)
    """
    return batch_interp3d(x[None], p[None], mode=mode)[0]


def batch_interp3d(
    x: torch.Tensor, p: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    """Interpolate values from a 3D grid.

    Args
    ----
        x: shape(B, C, D, H, W), any dtype
        p: dtype float32, shape(B, ..., 3), format (d_idx, h_idx, w_idx)
           Note that this differs from `batch_interp2d()`, which uses the flipped format (w_idx, h_idx).
        mode: one of {"bilinear", "nearest"}

    Returns
    -------
        Interpolated values from `x`, dtype `x.dtype` and shape(B, ..., C)
    """
    if mode not in ("bilinear", "nearest"):
        raise ValueError(f"Unexpected interpolation mode: {mode}")

    if x.dtype not in (torch.float16, torch.float32) and mode != "nearest":
        raise ValueError(
            f'Must set mode="nearest" for non-floating dtypes (got dtype={x.dtype} and mode={mode}'
        )

    size = torch.tensor(x.shape[-3:], dtype=torch.float32, device=x.device)
    p_normed = (
        p.div(size.float().sub(1.0).mul(0.5)).sub(1.0).flip(-1)
    )  # same shape but [-1, 1] values

    x_interp = F.grid_sample(
        x.type_as(p_normed),
        p_normed.reshape(p_normed.size(0), -1, 1, 1, 3),
        mode=mode,
        align_corners=True,
    )  # shape(B, C, N, 1, 1)

    dims = [dim for dim in p.shape[:-1]] + [x.size(1)]
    return x_interp.permute(0, 2, 1, 3, 4).reshape(dims).type_as(x)


def transform_points(pose: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    rotmat = pose[..., :3, :3]
    pos = pose[..., :3, 3]
    return torch.einsum("...ij,...j->...i", rotmat, pts) + pos


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)
