"""Test data generators for debugging and benchmarking."""

import numpy as np
import xarray as xr
import colour_demosaicing as cdm

from . import conventions as c
from .raw import BayerPattern


def raw(cfa, dark, pattern, exposure, gain, metas, wl_range):
    """Raw data (CFA, dark and metadata)."""
    b, _, _ = cfa.shape
    sinvs, npeaks, wls = metas

    data = xr.DataArray(
        cfa,
        dims=c.cfa_dims,
        )

    if not np.isscalar(exposure):
        exposure = (c.image_index, exposure)

    raw = metas.assign({
        c.cfa_data: data,
        c.dark_reference_data: (c.dark_ref_dims, dark),
        c.cfa_pattern_data:
            (c.image_index, np.full((b,), BayerPattern.get(pattern).name)),
        c.camera_exposure:
            (c.image_index, np.full((b,), exposure)),
        c.camera_gain:
            (c.image_index, np.full((b,), gain)),
        })

    return raw


def cfa(size, pattern, R, G, B):
    """CFA data with given R, G and B values."""
    b, y, x = size

    pattern = BayerPattern.get(pattern).name
    masks = cdm.bayer.masks_CFA_Bayer((y, x), pattern)
    cfa = np.zeros(size, dtype=np.uint16)
    cfa[:, masks[0]] = R
    cfa[:, masks[1]] = G
    cfa[:, masks[2]] = B
    return cfa


def metadata(size, wl_range):
    idxs = size[0]
    wl_start, wl_end = wl_range

    # Number of peaks for each index
    npeaks = np.tile([1, 2, 3], idxs + idxs % 3)[:idxs]

    # distinct sinvs for adjacent indices
    tmp = np.array([[[0, 0, 1],   # R, G, B sinvs for 1st peak in the 1st image
                     [0, 0, 0],   # 2nd peak
                     [0, 0, 0]],  # 3rd peak
                    [[0, 1, 0],   # and so on for 2nd image
                     [0, 1, 1],
                     [0, 0, 0]],
                    [[1, 0, 0],   # and the 3rd image
                     [1, 0, 1],
                     [1, 1, 0]]])
    sinvs = np.tile(tmp, (idxs // 3 + 1, 1, 1))[:idxs, :, :]

    # Reasonable wavelengths for existing peaks
    mask = np.array([[i < npeaks[n] for i in range(3)] for n in range(idxs)])
    wls = np.zeros((idxs, 3))
    wls.T.flat[mask.T.flatten()] = np.linspace(*wl_range, np.sum(npeaks))
    return xr.Dataset(
        data_vars={
            c.sinv_data: (c.sinv_dims, sinvs),
            c.number_of_peaks: (c.image_index, npeaks),
            c.wavelength_data: ((c.image_index, c.peak_coord), wls),
            },
        coords={
            c.image_index: np.arange(idxs),
            c.peak_coord: np.array([1, 2, 3]),
            c.colour_coord: ['R', 'G', 'B'],
            },
        )


def rad(cfa, dark_level, exposure, metas, wl_range):
    """Radiance data corresponding to CFA with 1, 2 and 5 as R, G, and B.
    and given constant dark level.
    """
    k, y, x = cfa.shape
    b = int(np.sum(metas[c.number_of_peaks]))

    # Corresponding indices and peaks of the original images
    idxs = np.clip(np.tile([0, 1, 2, 1, 2, 2], b // 6 + b)[:b], 0, k-1)
    peak = np.tile(
            np.array(k * [1] + (k-1) * [2] + (k-2) * [3]), b // 6 + 1)[:b]

    # Purposefully computed by hand
    # Currently only works for the dark levels 0, 1
    values = np.array([5, 2, 1, 7, 6, 3], dtype=np.float64)
    dark_rad = dark_level * np.array([1, 1, 1, 2, 2, 2], dtype=np.float64)
    values = values - dark_rad

    values = values / exposure

    values = values.reshape(-1, 1, 1)
    values = np.tile(values, (b // 6 + 1, 1, 1))[:b]

    data = np.kron(np.ones((y, x), dtype=np.float64), values)
    wls = np.linspace(*wl_range, b)

    rad = xr.Dataset(
            data_vars={
                c.radiance_data: (c.radiance_dims, data),
                c.wavelength_data: (c.band_index, wls),
                },
            coords={
                c.image_index: (c.band_index, idxs),
                c.peak_coord: (c.band_index, peak),
                c.band_index: (c.band_index, np.arange(b) + 1),
                },
            attrs={
                'long_name': 'radiance per unit wavelength',
                'units': 'W sr-1 m-2 nm-1',
            })
    return rad


def dark(shape, level):
    """Dark frame with given shape and level."""
    y, x = shape
    return np.full((y, x), level, dtype=np.uint16)
