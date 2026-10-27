import torch
from torch.nn import functional as F


def resize_volume(tensor, size, mode="trilinear"):
    size = tuple(size)
    if tuple(tensor.shape[-3:]) == size:
        return tensor
    options = {"align_corners": False} if mode == "trilinear" else {}
    return F.interpolate(tensor.float(), size=size, mode=mode, **options)


def support_at_resolution(image, target, valid, size):
    """Regenerate prompts on a resized categorical target; rescue a disappearing foreground seed."""
    if tuple(image.shape[-3:]) == tuple(size):
        return image, target, valid, False
    resized = resize_volume(image[None], size)[0]
    labels = resize_volume(target[None], size, "nearest-exact")[0]
    known = resize_volume(valid[None], size, "nearest-exact")[0].bool()
    labels *= known
    rescued = bool(target.any() and not labels.any())
    if rescued:
        # Map a real foreground voxel center, then rasterize the prompt at student resolution.
        point = torch.nonzero(
            target[0] & valid[0]
            if target.dtype == torch.bool
            else (target[0] > 0) & valid[0]
        )[0]
        mapped = (
            (
                (point.float() + 0.5)
                * point.new_tensor(size)
                / point.new_tensor(target.shape[-3:])
            )
            .floor()
            .long()
        )
        index = (0, *mapped.tolist())
        labels[index], known[index] = 1, True
    return resized, labels, known, rescued


def full_query(batch):
    context = getattr(batch, "query_context", None)
    return batch.query_images if context is None else context


def teacher_inputs(logits, batch, size=None):
    context = full_query(batch)
    size = tuple(size or context.shape[-3:])
    valid = resize_volume(batch.query_valid, size, "area") >= 1 - 1e-6
    return resize_volume(context, size), resize_volume(logits, size), valid


def foreground_probability(logits, size):
    from .losses import binary_logits

    return resize_volume(binary_logits(logits.float()), size).sigmoid()
