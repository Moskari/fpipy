# -*- coding: utf-8 -*-

""" Test images.
"""

import os.path as osp
from xarray import open_rasterio
from ..io import read_ENVI_cfa, read_calibration
from .. import conventions as c


data_dir = osp.abspath(osp.dirname(__file__))


__all__ = ['data_dir',
           'house_radiance',
           'house_raw',
           'house_calibration',
           ]


def house_calibration():
    """Calibration sequence for the house dataset.

    Calibration data for the `house_raw` dataset as read from the camera
    calibration file (instead of the VTT generated header).

    """
    return read_calibration(osp.join(data_dir, 'house_calib_seq.txt'))


def house_raw(**kwargs):
    """Raw images of a house and nearby foliage.

    Raw CFA data from the VTT FPI imager.

    CC0

    Parameters
    ----------

    **kwargs
        Keyword arguments passed on to `read_ENVI_cfa`.

    Returns
    -------

    house_raw : xarray.Dataset
        (4, 400, 400) cube of CFA images with metadata.

    """

    return read_ENVI_cfa(
            osp.join(data_dir, 'house_crop_4b_RAW.dat'),
            **kwargs
            )


def house_radiance(**kwargs):
    """ Radiance image of a house and foliage.

    Radiances calculated by the VTT software from the `house_raw`
    dataset.

    CC0

    Parameters
    ----------
    **kwargs
        Parameters passed on to `xr.open_rasterio`.

    Returns
    -------

    house_rad : xarray.Dataset
        (4, 400, 400) radiance cube with metadata.

    """

    res = open_rasterio(
            osp.join(data_dir, 'house_crop_4b_RAD.dat'),
            **kwargs
            )
    res = res.to_dataset(name=c.radiance_data)
    res = res.reset_coords()
    return res
