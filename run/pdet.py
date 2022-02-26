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

_simple_detect=False
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

if not _simple_detect:
    logging.info('pdet uses 8 neighboring pixels for detection.')
    # 3x3 pixels
    _default_inds=[(_j,_i) for _j in range(1,4) for _i in range(1,4)]
    _peak_types=[('pdet_f11','f8'),('pdet_f12','f8'), ('pdet_f13','f8'),\
                ('pdet_f21','f8'), ('pdet_f22','f8'), ('pdet_f23','f8'),\
                ('pdet_f31','f8'), ('pdet_f32','f8'), ('pdet_f33','f8'),\
                ('pdet_f11r1','f8'), ('pdet_f12r1','f8'),('pdet_f13r1','f8'),\
                ('pdet_f21r1','f8'), ('pdet_f22r1','f8'),('pdet_f23r1','f8'),\
                ('pdet_f31r1','f8'), ('pdet_f32r1','f8'),('pdet_f33r1','f8'),\
                ('pdet_f11r2','f8'), ('pdet_f12r2','f8'),('pdet_f13r2','f8'),\
                ('pdet_f21r2','f8'), ('pdet_f22r2','f8'),('pdet_f23r2','f8'),\
                ('pdet_f31r2','f8'), ('pdet_f32r2','f8'),('pdet_f33r2','f8')]
    _pdet_types=[('pdet_y','i4'),  ('pdet_x','i4'),\
                ('pdet_v11','f8'),('pdet_v12','f8'), ('pdet_v13','f8'),\
                ('pdet_v21','f8'), ('pdet_v22','f8'), ('pdet_v23','f8'),\
                ('pdet_v31','f8'), ('pdet_v32','f8'), ('pdet_v33','f8'),\
                ('pdet_v11r1','f8'), ('pdet_v12r1','f8'),('pdet_v13r1','f8'),\
                ('pdet_v21r1','f8'), ('pdet_v22r1','f8'),('pdet_v23r1','f8'),\
                ('pdet_v31r1','f8'), ('pdet_v32r1','f8'),('pdet_v33r1','f8'),\
                ('pdet_v11r2','f8'), ('pdet_v12r2','f8'),('pdet_v13r2','f8'),\
                ('pdet_v21r2','f8'), ('pdet_v22r2','f8'),('pdet_v23r2','f8'),\
                ('pdet_v31r2','f8'), ('pdet_v32r2','f8'),('pdet_v33r2','f8')]
else:
    logging.info('pdet uses 4 neighboring pixels for detection.')
    # 3x3 pixels
    _default_inds=[(1,2),(2,1),(2,2),(2,3),(3,2)]
    _peak_types=[('pdet_f12','f8'), ('pdet_f21','f8'),  ('pdet_f22','f8'),\
                ('pdet_f23','f8'),  ('pdet_f32','f8'),\
                ('pdet_f12r1','f8'),('pdet_f21r1','f8'),('pdet_f22r1','f8'),\
                ('pdet_f23r1','f8'),('pdet_f32r1','f8'),\
                ('pdet_f12r2','f8'),('pdet_f21r2','f8'),('pdet_f22r2','f8'),\
                ('pdet_f23r2','f8'),('pdet_f32r2','f8')]
    _pdet_types=[('pdet_y','i4'),  ('pdet_x','i4'), \
                ('pdet_v12','f8'), ('pdet_v21','f8'),  ('pdet_v22','f8'),\
                ('pdet_v23','f8'),  ('pdet_v32','f8'),\
                ('pdet_v12r1','f8'),('pdet_v21r1','f8'),('pdet_v22r1','f8'),\
                ('pdet_v23r1','f8'),('pdet_v32r1','f8'),\
                ('pdet_v12r2','f8'),('pdet_v21r2','f8'),('pdet_v22r2','f8'),\
                ('pdet_v23r2','f8'),('pdet_v32r2','f8')]

def detect_coords(imgCov,thres):
    """
    detect peaks and return the coordinates (y,x)
    Parameters:
        imgCov:     convolved image
        thres:      detection threshold
    Returns:
        ndarray of coordinates (y,x)
    """
    footprint = np.ones((3, 3))
    footprint[1, 1] = 0
    if _simple_detect:
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
    return out[msk]

def get_shear_response(imgData,psfData,gsigma=3.*2*np.pi/64,thres=0.01,coords=None):
    """
    Get the shear response for pixels identified as peaks
    Parameters:
        imgData:    observed image [ndarray]
        psfData:    PSF image center at middle [ndarray]
        gsigma:     sigma of the Gaussian smoothing kernel [float]
        thres:      detection threshold
    Returns:
        peak values and the shear responses
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
        imgData:    observed image [ndarray]
        psfData:    PSF image (the average PSF of the exposure) [ndarray]
        gsigma:     sigma of the Gaussian smoothing kernel in Fourier space [float]
        thres:      detection threshold
    Returns:
        peak values and the shear responses
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
        coords:     coordinate array
        imgCov:     unsmeared image (cov) Gaussian
        imgCovQ1:   unsmeared image (cov) Gaussian (Q1)
        imgCovQ2:   unsmeared image (cov) Gaussian (Q2)
        imgCovD1:   unsmeared image (cov) Gaussian (D1)
        imgCovD2:   unsmeared image (cov) Gaussian (D2)

    Returns:
        peak array
    """
    out     =   np.array(np.zeros(coords.size),dtype=_peak_types)
    for _j,_i in _default_inds:
        # the smoothed pixel value
        _y  = coords['pdet_y']+_j-2
        _x  = coords['pdet_x']+_i-2
        _v  = imgCov[_y,_x]
        out['pdet_v%d%d' %(_j,_i)]=_v
        # responses for the smoothed pixel value
        _r1 = imgCovQ1[_y,_x]+(_i-2.)*imgCovD1[_y,_x]-(_j-2.)*imgCovD2[_y,_x]
        _r2 = imgCovQ2[_y,_x]+(_j-2.)*imgCovD1[_y,_x]+(_i-2.)*imgCovD2[_y,_x]
        out['pdet_v%d%dr1' %(_j,_i)]=_r1
        out['pdet_v%d%dr2' %(_j,_i)]=_r2
    out     =   rfn.merge_arrays([coords,out], flatten = True, usemask = False)
    return out

def peak2det(peaks):
    """
    from peak array to detection array
    Parameters:
    peaks:      peak array

    Returns:
        detection array
    """
    # prepare the output
    out     =   np.array(np.zeros(peaks.size),dtype=_pdet_types)
    # x,y are the same
    out['pdet_y']  = peaks['pdet_y']
    out['pdet_x']  = peaks['pdet_x']
    cnmv0   =   'pdet_v22'
    cnmr10  =   'pdet_v22r1'
    cnmr20  =   'pdet_v22r2'
    for ind in _default_inds:
        cnmv=   'pdet_v%d%d'  %ind
        cnmr1=  'pdet_v%d%dr1'%ind
        cnmr2=  'pdet_v%d%dr2'%ind
        fnmv =  'pdet_f%d%d'  %ind
        fnmr1=  'pdet_f%d%dr1'%ind
        fnmr2=  'pdet_f%d%dr2'%ind
        if ind  !=  (2,2):
            out[fnmv] = peaks[cnmv0]-peaks[cnmv]
            out[fnmr1]= peaks[cnmr10]-peaks[cnmr1]
            out[fnmr2]= peaks[cnmr20]-peaks[cnmr2]
        else:
            out[fnmv] = peaks[cnmv]
            out[fnmr1]= peaks[cnmr1]
            out[fnmr2]= peaks[cnmr2]
    return out

def get_detbias(dets,cut,dcc,inds=_default_inds,dcutz=True):
    """
    Parameters:
        dets: 	    detection array
        cut:        selection cut
        dcc:        bin size when estimating marginal density
        inds:       shifting indexes
    """
    if not isinstance(inds,list):
        if isinstance(inds,tuple):
            inds=[inds]
        else:
            raise TypeError('inds should be a list of tuple or a tuple')
    cor1 =0.
    cor2 =0.
    for ind in inds:
        fnmv ='pdet_f%d%d'  %ind
        fnmr1='pdet_f%d%dr1'%ind
        fnmr2='pdet_f%d%dr2'%ind
        ll=cut;uu=cut+dcc
        if ind!=(2,2) and dcutz:
            ll=0.;uu=dcc
        msk=(dets[fnmv]>ll)&(dets[fnmv]<uu)
        if np.sum(msk)>2:
            cor1=cor1+np.sum(dets[fnmr1][msk])/dcc
            cor2=cor2+np.sum(dets[fnmr2][msk])/dcc
    return cor1,cor2

def detbias(sel,selresEll,cut,dcc):
    """
    Parameters:
        sel: 	    selection funciton
        selresEll:  response of selection function times ellipticities
        cut:        selection cut
        dcc:        bin size when estimating marginal density
    """
    msk=(sel>cut)&(sel<cut+dcc)
    if np.sum(msk)==0:
        cor=0.
    else:
        cor=np.sum(selresEll[msk])/dcc
    return cor
