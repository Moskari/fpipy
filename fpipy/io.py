# -*- coding: utf-8 -*-

"""Functions for reading data from various formats.
"""

import xarray as xr
import os
import configparser
from .raw import _cfa_to_dataset
from .meta import metalist


def load_hdt(hdtfile):
    """Load metadata from a .hdt header file (VTT format)."""

    if not os.path.isfile(hdtfile):
        raise(IOError('Header file {} does not exist'.format(hdtfile)))

    meta = configparser.ConfigParser()
    meta.read(hdtfile)

    return meta


def read_cfa(filepath):
    """Read a raw CFA datafile and metadata to an xarray Dataset.

    For the fpi sensor in JYU, the metadata in the ENVI datafile is not
    relevant but is preserved as dataset.cfa.attrs just in case.
    Wavelength and fwhm data will be replaced with information from metadata
    and number of layers etc. are omitted as redundant.
    Gain and bayer pattern are assumed to be constant within each file.

    Parameters
    ----------
    filepath : str
        Path to the datafile to be opened, either with or without extension.
        Expects data and metadata to have extensions .dat and .hdt.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset derived from the raw image data and accompanying metadata.
    """

    base = os.path.splitext(filepath)[0]
    datfile = base + '.dat'
    hdtfile = base + '.hdt'

    cfa = xr.open_rasterio(datfile)
    meta = load_hdt(hdtfile)

    if 'fwhm' in cfa.coords:
        cfa = cfa.drop('fwhm')
    if 'wavelength' in cfa.coords:
        cfa = cfa.drop('wavelength')

    npeaks = ('band', metalist(meta, 'npeaks'))
    wavelength = (['band', 'peak'], metalist(meta, 'wavelengths'))
    fwhm = (['band', 'peak'], metalist(meta, 'fwhms'))
    setpoints = (['band', 'setpoint'], metalist(meta, 'setpoints'))
    sinvs = (['band', 'peak', 'rgb'], metalist(meta, 'sinvs'))

    attrs = {
        'fpi_temperature': meta.getfloat('Header', 'fpi temperature'),
        'description': meta.get('Header', 'description').strip('"'),
        'dark_layer_included':
            meta.getboolean('Header', 'dark layer included'),
        'gain': meta.getfloat('Image0', 'gain'),
        'exposure': meta.getfloat('Image0', 'exposure time (ms)'),
        'bayer_pattern': meta.getint('Image0', 'bayer pattern')
        }

    return _cfa_to_dataset(
            cfa,
            npeaks,
            wavelength,
            fwhm,
            setpoints,
            sinvs,
            attrs=attrs
            )
