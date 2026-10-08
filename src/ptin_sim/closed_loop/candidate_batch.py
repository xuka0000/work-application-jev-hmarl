"""Lossless packed candidate features with the original per-state score masks."""
import torch


def validate_packed(candidates, valid, *, count, features):
    if (valid.ndim != 2 or valid.shape[0] != count or valid.dtype != torch.bool
            or not valid.any(dim=1).all() or candidates.ndim != 2
            or candidates.shape != (int(valid.sum()), features)):
        raise ValueError('Packed candidates must match the row-major availability mask')


def packed_scores(states, candidates, valid, score):
    """Only scalar scores are padded; no missing action enters a network."""
    validate_packed(candidates, valid, count=len(states), features=candidates.shape[-1])
    rows, columns = valid.nonzero(as_tuple=True)
    values = score(states[rows], candidates[:, None, :])[:, 0]
    out = values.new_full(valid.shape, -torch.inf)
    out[rows, columns] = values
    return out


def packed_selected(candidates, valid, chosen):
    counts = valid.sum(dim=1)
    starts = counts.cumsum(0) - counts
    within = (valid.long().cumsum(dim=1) - 1).gather(1, chosen[:, None])[:, 0]
    return candidates[starts + within]
