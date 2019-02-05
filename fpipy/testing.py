"""Test data generators for debugging and benchmarking."""

import numpy as np
import xarray as xr
import colour_demosaicing as cdm

from . import conventions as c
from .raw import BayerPattern


def raw(cfa, dark, pattern, exposure, metas):
    """Raw data (CFA, dark and metadata)."""
    b, y, x = cfa.shape
    sinvs, npeaks, wls = metas
    dc_included, dref = dark

    data = xr.DataArray(
        cfa,
        dims=c.cfa_dims,
        attrs={c.dc_included_attr: dc_included}
        )
    raw = xr.Dataset(
        data_vars={
            c.cfa_data: data,
            c.dark_reference_data: (c.dark_ref_dims, dref),
            c.number_of_peaks: (c.image_index, npeaks),
            c.sinv_data: (
                (c.image_index, c.peak_coord, c.colour_coord),
                sinvs
                ),
            c.cfa_pattern_data: pattern,
            c.camera_exposure: exposure,
            c.wavelength_data: ((c.image_index, c.peak_coord), wls),
            },
        coords={
            c.image_index: np.arange(b),
            c.peak_coord: np.array([1, 2, 3]),
            c.colour_coord: ['R', 'G', 'B'],
            },
        )

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
    tmp = np.array([[[0, 0, 1],
                     [0, 0, 0],
                     [0, 0, 0]],
                    [[0, 1, 0],
                     [0, 1, 1],
                     [0, 0, 0]],
                    [[1, 0, 0],
                     [1, 0, 1],
                     [1, 1, 0]]])
    sinvs = np.tile(tmp, (idxs // 3 + 1, 1, 1))[:idxs, :, :]

    # Reasonable wavelengths for existing peaks
    mask = np.array([[i < npeaks[n] for i in range(3)] for n in range(idxs)])
    wls = np.zeros((idxs, 3))
    wls.T.flat[mask.T.flatten()] = np.linspace(*wl_range, np.sum(npeaks))
    return sinvs, npeaks, wls


def rad(cfa, dark, exposure, metas, wl_range):
    """Radiance data corresponding to CFA with 1, 2 and 5 as R, G, and B."""
    k, y, x = cfa.shape
    _, npeaks, _ = metas
    hasdark, _ = dark
    b = np.sum(npeaks)

    if hasdark:
        values = np.array([4, 1, 0, 5, 4, 1], dtype=np.float64)
    else:
        values = np.array([5, 2, 1, 7, 6, 3], dtype=np.float64)

    values = values.reshape(-1, 1, 1)
    values = np.tile(values, (b // 6 + 1, 1, 1))[:b]
    data = np.kron(np.ones((y, x), dtype=np.float64), values) / exposure
    wls = np.linspace(*wl_range, b)

    rad = xr.Dataset(
            data_vars={
                c.radiance_data: (c.radiance_dims, data),
                c.wavelength_data: (c.band_index, wls),
                },
            coords={
                c.band_index: np.arange(b),
                }
            )
    return rad


def dark(shape):
    """Dark frame with given shape."""
    y, x = shape
    return np.ones((y, x), dtype=np.uint16)
