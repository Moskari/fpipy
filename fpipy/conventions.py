# -*- coding: utf-8 -*-

"""Conventions for data, coordinate and attribute names."""

measurement_type = 'measurement'

peak_coord = 'peak'
"""Index coordinate denoting passband peaks of the FPI."""

number_of_peaks = 'npeaks'
"""Number of passband peaks included in a given image."""

setpoint_coord = 'setpoint_index'
"""Coordinate denoting the the setpoint (which voltage it denotes)"""

setpoint_data = 'setpoint'
"""Values of the setpoints."""

sinv_data = 'sinvs'
"""Inversion coefficients for radiance calculation."""

colour_coord = 'colour'
"""Colour coordinate for values indexed w.r.t. CFA colours."""

radiance_data = 'radiance'
"""DataArray containing a stack of radiance images."""

reflectance_data = 'reflectance'
"""DataArray containing a stack of reflectance images."""

cfa_data = 'dn'
"""DataArray containing a stack of raw CFA images."""

dark_corrected_cfa_data = 'dn_dark_corrected'
"""DataArray containing a stack of raw images with dark current corrected"""

rgb_data = 'rgb'
"""DataArray containing a stack of RGB images."""

dark_reference_data = 'dark'
"""DataArray containing a dark current reference."""

wavelength_data = 'wavelength'
"""Passband center wavelength values."""

fwhm_data = 'fwhm'
"""Full Width at Half Maximum values"""

camera_gain = 'gain'
"""Camera analog gain value."""

camera_exposure = 'exposure'
"""Camera exposure time in milliseconds."""

genicam_exposure = 'ExposureTime'
"""Camera exposure time in microseconds as defined by GenICam."""

cfa_pattern_data = 'bayer_pattern'
"""String denoting the pattern (e.g. RGGB) of the colour filter array."""

genicam_pattern_data = 'PixelColorFilter'
"""String denoting the pattern (e.g. BayerGB) of the colour filter array."""

image_index = 'index'
"""Index number of the image in a given stack."""

band_index = 'band'
"""Index for wavelength band data."""

image_width = 'width'
image_height = 'height'
height_coord = 'y'
width_coord = 'x'

cfa_dims = (image_index, height_coord, width_coord)
"""Convention for CFA image stacks."""

dark_ref_dims = (height_coord, width_coord)
"""Convention for dark reference image dimensions."""

radiance_dims = (band_index, height_coord, width_coord)
"""Convention for radiance image stacks."""

RGB_dims = (height_coord, width_coord, colour_coord)
"""Convention for RGB images."""

sinv_dims = (image_index, peak_coord, colour_coord)
"""Convention for inversion coefficient dimensions."""
