from typing import Iterable
from itertools import repeat
import os
import os.path as osp

import numpy as np


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


pair = _ntuple(2)


def mkdir(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def bytescale(band, mask, cmin, cmax, dtype=np.uint8):
    band = np.asfarray(band)
    dtype_in = band.dtype
    dtype_out = np.dtype(dtype)
    mask = np.asarray(mask).astype(np.bool)

    if dtype_in == dtype_out:
        return band
    imin_out = np.iinfo(dtype_out).min
    imax_out = np.iinfo(dtype_out).max

    cmin = np.asarray(cmin)
    cmax = np.asarray(cmax)

    imin_out = cmin / cmax * imax_out
    band = (band - cmin) / (cmax - cmin) * (imax_out - imin_out) + imin_out
    # print(band.max(axis=(1, 2)))
    band = (band.clip(imin_out, imax_out) * mask).astype(dtype_out)

    return band


def calc_length(poly):
    # poly : ndarray, Nx2
    p1 = poly
    p2 = np.roll(p1, -1, axis=0)
    delta = p1 - p2  # dx, dy
    _, dx, dy = np.split(delta, [0, 1], axis=1)
    side_length = np.hypot(dx, dy)

    return side_length.squeeze()


def calc_angle(poly, deg=True):
    # compute inter-angle, 0~pi
    p1 = np.roll(poly, -1, axis=0)
    p2 = np.roll(poly, 1, axis=0)
    d1 = p1 - poly
    d2 = p2 - poly
    l1 = np.linalg.norm(d1, axis=1)
    l2 = np.linalg.norm(d2, axis=1)
    dot = (d1 * d2).sum(axis=1)

    angle = np.arccos(dot / (l1 * l2 + 1e-6))
    if deg:
        angle = np.rad2deg(angle)

    return angle