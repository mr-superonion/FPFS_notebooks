#
# Detect peaks and get the shear response of the detection
#
# Copyright 20220123 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib

import gc
import logging
import numpy as np
import scipy.ndimage as ndi
import numpy.lib.recfunctions as rfn

from fpfs.imgutil import gauss_kernel

_gsigma=3.*2*np.pi/64.

def try_numba_njit(func):
    try:
        import numba
        return numba.njit(func)
    except ImportError:
        return func

@try_numba_njit
def test_numba_njit(n):
    out=0
    for _ in range(n):
        out+=1
    return out

def test_numba(n):
    out=0
    for _ in range(n):
        out+=1
    return out

logging.info('pdet uses 4 neighboring pixels for detection.')
# 3x3 pixels
_default_inds=[(1,2),(2,1),(2,2),(2,3),(3,2)]
_peak_types=[('pdet_f12','f8'), ('pdet_f21','f8'),  ('pdet_f22','f8'),\
            ('pdet_f23','f8'),  ('pdet_f32','f8'),\
            ('pdet_f12r1','f8'),('pdet_f21r1','f8'),('pdet_f22r1','f8'),\
            ('pdet_f23r1','f8'),('pdet_f32r1','f8'),\
            ('pdet_f12r2','f8'),('pdet_f21r2','f8'),('pdet_f22r2','f8'),\
            ('pdet_f23r2','f8'),('pdet_f32r2','f8')]
_pdet_types=[('pdet_v12','f8'), ('pdet_v21','f8'),  ('pdet_v22','f8'),\
            ('pdet_v23','f8'),  ('pdet_v32','f8'),\
            ('pdet_v12r1','f8'),('pdet_v21r1','f8'),('pdet_v22r1','f8'),\
            ('pdet_v23r1','f8'),('pdet_v32r1','f8'),\
            ('pdet_v12r2','f8'),('pdet_v21r2','f8'),('pdet_v22r2','f8'),\
            ('pdet_v23r2','f8'),('pdet_v32r2','f8')]

_ncov_types=[]
for (j,i) in _default_inds:
    _ncov_types.append(('pdet_N00V%d%dr1'  %(j,i),'>f8'))
    _ncov_types.append(('pdet_N00V%d%dr2'  %(j,i),'>f8'))
    _ncov_types.append(('pdet_N22cV%d%dr1' %(j,i),'>f8'))
    _ncov_types.append(('pdet_N22sV%d%dr2' %(j,i),'>f8'))

def detect_coords(imgCov,thres):
    """
    detect peaks and return the coordinates (y,x)
    Parameters:
    ----
    imgCov:     convolved image
    thres:      detection threshold

    Returns:
    ----
    coord_array:     ndarray of coordinates (y,x)
    """
    footprint = np.ones((3, 3))
    footprint[1, 1] = 0
    footprint[0, 0] = 0
    footprint[0, 2] = 0
    footprint[2, 2] = 0
    footprint[2, 0] = 0
    filtered=   ndi.maximum_filter(imgCov,footprint=footprint,mode='constant')
    data    =   np.int_(np.asarray(np.where(((imgCov > filtered)&(imgCov>thres)))))
    out     =   np.array(np.zeros(data.size//2),dtype=[('pdet_y','i4'),('pdet_x','i4')])
    out['pdet_y']=data[0]
    out['pdet_x']=data[1]
    ny,nx = imgCov.shape
    msk   = (out['pdet_y']>20)&(out['pdet_y']<ny-20)\
                &(out['pdet_x']>20)&(out['pdet_x']<nx-20)
    coord_array= out[msk]
    return coord_array

def get_shear_response(imgData,psfData,gsigma=_gsigma,thres=0.01,coords=None):
    """
    Get the shear response for pixels identified as peaks
    Parameters:
    ----
    imgData:    observed image [ndarray]
    psfData:    PSF image center at middle [ndarray]
    gsigma:     sigma of the Gaussian smoothing kernel [float]
    thres:      detection threshold

    Returns:
    ----
    peak_array    peak values and the shear responses
    """

    assert imgData.shape==psfData.shape, 'image and PSF should have the same\
            shape. Please do padding before using this function.'
    ny,nx   =   psfData.shape
    psfF    =   np.fft.fft2(np.fft.ifftshift(psfData))

    gKer,(k2grid,k1grid)=gauss_kernel(ny,nx,gsigma,return_grid=True)

    # convolved images
    imgF    =   np.fft.fft2(imgData)/psfF*gKer
    del psfF,psfData
    imgCov  =   np.fft.ifft2(imgF).real
    # Q
    imgFQ1  =   imgF*(k1grid**2.-k2grid**2.)/gsigma**2.
    imgFQ2  =   imgF*(2.*k1grid*k2grid)/gsigma**2.
    imgCovQ1=   np.fft.ifft2(imgFQ1).real
    imgCovQ2=   np.fft.ifft2(imgFQ2).real
    del imgFQ1,imgFQ2 # these images take a lot of memory

    # D
    imgFD1  =   imgF*(-1j*k1grid)
    imgFD2  =   imgF*(-1j*k2grid)
    imgCovD1=   np.fft.ifft2(imgFD1).real
    imgCovD2=   np.fft.ifft2(imgFD2).real
    del imgFD1,imgFD2,imgF,k1grid,k2grid # these images take a lot of memory
    gc.collect()

    if coords is None:
        # the coordinates is not given, so we do another detection
        if not isinstance(thres,(int, np.floating)):
            raise ValueError('thres must be float, but now got %s' %type(thres))
        coords  =   detect_coords(imgCov,thres)
    peak_array  =   _make_peak_array(coords,imgCov,imgCovQ1,imgCovQ2,imgCovD1,imgCovD2)
    return peak_array

def get_shear_response_rfft(imgData,psfData,gsigma=_gsigma,thres=0.01,coords=None):
    """
    Get the shear response for pixels identified as peaks.
    This fucntion ueses np.fft.rfft2 instead of np.fft.fft2
    (This is about 1.35 times faster and only use 0.85 memory)
    Parameters:
    ----
    imgData:    observed image [ndarray]
    psfData:    PSF image (the average PSF of the exposure) [ndarray]
    gsigma:     sigma of the Gaussian smoothing kernel in Fourier space [float]
    thres:      detection threshold

    Returns:
    ----
    peak_array: peak values and the shear responses
    """

    assert imgData.shape==psfData.shape, 'image and PSF should have the same\
            shape. Please do padding before using this function.'
    ny,nx   =   psfData.shape
    psfF    =   np.fft.rfft2(np.fft.ifftshift(psfData))
    gKer,(k2grid,k1grid)=gauss_kernel(ny,nx,gsigma,return_grid=True,use_rfft=True)

    # convolved images
    imgF    =   np.fft.rfft2(imgData)/psfF*gKer
    del psfF,psfData
    imgCov  =   np.fft.irfft2(imgF,(ny,nx))
    # Q
    imgFQ1  =   imgF*(k1grid**2.-k2grid**2.)/gsigma**2.
    imgFQ2  =   imgF*(2.*k1grid*k2grid)/gsigma**2.
    imgCovQ1=   np.fft.irfft2(imgFQ1,(ny,nx))
    imgCovQ2=   np.fft.irfft2(imgFQ2,(ny,nx))
    del imgFQ1,imgFQ2 # these images take a lot of memory
    # D
    imgFD1  =   imgF*(-1j*k1grid)
    imgFD2  =   imgF*(-1j*k2grid)
    imgCovD1=   np.fft.irfft2(imgFD1,(ny,nx))
    imgCovD2=   np.fft.irfft2(imgFD2,(ny,nx))
    del imgFD1,imgFD2,imgF,k1grid,k2grid # these images take a lot of memory
    gc.collect()

    if coords is None:
        # the coordinates is not given, so we do another detection
        if not isinstance(thres,(int, float)):
            raise ValueError('thres must be float, but now got %s' %type(thres))
        coords  =   detect_coords(imgCov,thres)
    peak_array  =   _make_peak_array(coords,imgCov,imgCovQ1,imgCovQ2,imgCovD1,imgCovD2)

    return peak_array

def _make_peak_array(coords,imgCov,imgCovQ1,imgCovQ2,imgCovD1,imgCovD2):
    """
    make the peak array and the shear response of the peak array
    Parameters:
    ----
    coords:     coordinate array
    imgCov:     unsmeared image (cov) Gaussian
    imgCovQ1:   unsmeared image (cov) Gaussian (Q1)
    imgCovQ2:   unsmeared image (cov) Gaussian (Q2)
    imgCovD1:   unsmeared image (cov) Gaussian (D1)
    imgCovD2:   unsmeared image (cov) Gaussian (D2)

    Returns:
    ----
    out:        peak array
    """
    out     =   np.array(np.zeros(coords.size),dtype=_peak_types)
    for _j,_i in _default_inds:
        # the smoothed pixel value
        _y  = coords['pdet_y']+_j-2
        _x  = coords['pdet_x']+_i-2
        _v  = imgCov[_y,_x]
        out['pdet_f%d%d' %(_j,_i)]=_v
        # responses for the smoothed pixel value
        _r1 = imgCovQ1[_y,_x]+(_i-2.)*imgCovD1[_y,_x]-(_j-2.)*imgCovD2[_y,_x]
        _r2 = imgCovQ2[_y,_x]+(_j-2.)*imgCovD1[_y,_x]+(_i-2.)*imgCovD2[_y,_x]
        out['pdet_f%d%dr1' %(_j,_i)]=_r1
        out['pdet_f%d%dr2' %(_j,_i)]=_r2
    out     =   rfn.merge_arrays([coords,out], flatten = True, usemask = False)
    return out

def peak2det(peaks):
    """
    convert peak array (merged with fpfs catalog) to detection array
    Parameters:
    ----
    peaks:  peak array

    Returns:
    ----
    out:    detection array
    """
    # prepare the output

    if 'pdet_N22sF22r2' in peaks.dtype.names:
        noicov  =   True
        out     =   np.array(np.zeros(peaks.size),dtype=_pdet_types+_ncov_types)
    else:
        noicov=False
        out     =   np.array(np.zeros(peaks.size),dtype=_pdet_types)

    # column name for noise covariance for pixel values
    inm10   =   'pdet_N22cF22r1'
    inm20   =   'pdet_N22sF22r2'
    inm30   =   'pdet_N00F22r1'
    inm40   =   'pdet_N00F22r2'

    # v and two components of shear response (vr1, vr2)
    rlist   =   ['', 'r1', 'r2']
    fnmv0   =   'pdet_f22%s'
    for ind in _default_inds:
        # the shear response of detection modes
        for rr in rlist:
            # get the detection modes from the pixel values
            fnmv =  'pdet_f%d%d%s'  %(ind+(rr,))
            cnmv =  'pdet_v%d%d%s'  %(ind+(rr,))
            fnmv22= fnmv0 %(rr)
            if ind  !=  (2,2):
                out[cnmv] = peaks[fnmv22]-peaks[fnmv]
            else:
                out[cnmv] = peaks[fnmv]
            del fnmv,cnmv,fnmv22

        if noicov:
            inm1    =   'pdet_N22cF%d%dr1' %ind
            onm1    =   'pdet_N22cV%d%dr1' %ind
            inm2    =   'pdet_N22sF%d%dr2' %ind
            onm2    =   'pdet_N22sV%d%dr2' %ind
            inm3    =   'pdet_N00F%d%dr1' %ind
            onm3    =   'pdet_N00V%d%dr1' %ind
            inm4    =   'pdet_N00F%d%dr2' %ind
            onm4    =   'pdet_N00V%d%dr2' %ind
            if ind  !=  (2,2):
                out[onm1] = peaks[inm10]-peaks[inm1]
                out[onm2] = peaks[inm20]-peaks[inm2]
                out[onm3] = peaks[inm30]-peaks[inm3]
                out[onm4] = peaks[inm40]-peaks[inm4]
            else:
                out[onm1] = peaks[inm1]
                out[onm2] = peaks[inm2]
                out[onm3] = peaks[inm3]
                out[onm4] = peaks[inm4]
            del inm1,onm1,inm2,onm2,inm3,onm3,inm4,onm4
    return out
