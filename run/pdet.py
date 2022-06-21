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

from fpfs.imgutil import gauss_kernel

# _gsigma=3.*2*np.pi/64.

logging.info('pdet uses 4 neighboring pixels for detection.')
# 3x3 pixels
_default_inds=[(1,2),(2,1),(2,2),(2,3),(3,2)]
_peak_types=[('fpfs_peak_x','i4'), ('fpfs_peak_y','i4'),
            ('fpfs_f12','<f8'), ('fpfs_f21','<f8'),  ('fpfs_f22','<f8'),\
            ('fpfs_f23','<f8'),  ('fpfs_f32','<f8'),\
            ('fpfs_f12r1','<f8'),('fpfs_f21r1','<f8'),('fpfs_f22r1','<f8'),\
            ('fpfs_f23r1','<f8'),('fpfs_f32r1','<f8'),\
            ('fpfs_f12r2','<f8'),('fpfs_f21r2','<f8'),('fpfs_f22r2','<f8'),\
            ('fpfs_f23r2','<f8'),('fpfs_f32r2','<f8')]

_fpfs_types=[ ('fpfs_peak_x','i4'), ('fpfs_peak_y','i4'),
            ('fpfs_v12','<f8'), ('fpfs_v21','<f8'),  ('fpfs_v22','<f8'),\
            ('fpfs_v23','<f8'),  ('fpfs_v32','<f8'),\
            ('fpfs_v12r1','<f8'),('fpfs_v21r1','<f8'),('fpfs_v22r1','<f8'),\
            ('fpfs_v23r1','<f8'),('fpfs_v32r1','<f8'),\
            ('fpfs_v12r2','<f8'),('fpfs_v21r2','<f8'),('fpfs_v22r2','<f8'),\
            ('fpfs_v23r2','<f8'),('fpfs_v32r2','<f8')]

_ncov_types=[]
for (j,i) in _default_inds:
    _ncov_types.append(('fpfs_N00V%d%dr1'  %(j,i),'<f8'))
    _ncov_types.append(('fpfs_N00V%d%dr2'  %(j,i),'<f8'))
    _ncov_types.append(('fpfs_N22cV%d%dr1' %(j,i),'<f8'))
    _ncov_types.append(('fpfs_N22sV%d%dr2' %(j,i),'<f8'))

def detect_coords(imgCov,thres,thres2=0.):
    """Detects peaks and returns the coordinates (y,x)
    Args:
        imgCov (ndarray):       convolved image
        thres (float):          detection threshold
        thres2 (float):         peak identification difference threshold
    Returns:
        coord_array (ndarray):  ndarray of coordinates (y,x)
    """
    footprint = np.ones((3, 3))
    footprint[1, 1] = 0
    footprint[0, 0] = 0
    footprint[0, 2] = 0
    footprint[2, 2] = 0
    footprint[2, 0] = 0
    filtered=   ndi.maximum_filter(imgCov,footprint=footprint,mode='constant')
    data    =   np.int_(np.asarray(np.where(((imgCov > filtered+thres2)&(imgCov>thres)))))
    out     =   np.array(np.zeros(data.size//2),dtype=[('fpfs_peak_y','i4'),('fpfs_peak_x','i4')])
    out['fpfs_peak_y']=data[0]
    out['fpfs_peak_x']=data[1]
    ny,nx = imgCov.shape
    # avoid pixels near boundary
    msk     =   (out['fpfs_peak_y']>20)&(out['fpfs_peak_y']<ny-20)\
                &(out['fpfs_peak_x']>20)&(out['fpfs_peak_x']<nx-20)
    coord_array= out[msk]
    return coord_array

def get_shear_response(imgData,psfData,gsigma,thres=0.04,thres2=-0.01,klim=-1.,coords=None):
    """Returns the shear response for pixels identified as peaks
    Args:
        imgData (ndarray):      observed image [ndarray]
        psfData (ndarray):      PSF image center at middle [ndarray]
        gsigma (float):         sigma of the Gaussian smoothing kernel in Fourier space [float]
        thres (float):          detection threshold
        thres2 (float):         peak identification difference threshold
        klim (float):           limiting wave number in Fourier space
        coords (ndarray):       coordinates (x,y)
    Returns:
        peak_array (ndarray):   peak values and the shear responses
    """

    assert imgData.shape==psfData.shape, 'image and PSF should have the same\
            shape. Please do padding before using this function.'
    ny,nx   =   psfData.shape
    psfF    =   np.fft.fft2(np.fft.ifftshift(psfData))

    gKer,(k2grid,k1grid)=gauss_kernel(ny,nx,gsigma,return_grid=True)

    # convolved images
    imgF    =   np.fft.fft2(imgData)/psfF*gKer
    del psfF,psfData
    if klim>0.:
        # apply a truncation in Fourier space
        nxklim  =   int(klim*nx/np.pi/2.+0.5)
        nyklim  =   int(klim*ny/np.pi/2.+0.5)
        imgF[:ny//2-nyklim,:]    =    0.
        imgF[ny//2+nyklim+1:,:]    =    0.
        imgF[:,:nx//2-nxklim]    =    0.
        imgF[:,nx//2+nxklim+1:]    =    0.
    else:
        # no truncation in Fourier space
        pass
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
        coords  =   detect_coords(imgCov,thres,thres2)
    peak_array  =   _make_peak_array(coords,imgCov,imgCovQ1,imgCovQ2,imgCovD1,imgCovD2)
    return peak_array

def get_shear_response_rfft(imgData,psfData,gsigma,thres=0.04,thres2=-0.01,klim=-1,coords=None):
    """Returns the shear response for pixels identified as peaks.
    This fucntion ueses np.fft.rfft2 instead of np.fft.fft2
    (This is about 1.35 times faster and only use 0.85 memory)
    Args:
        imgData (ndarray):      observed image
        psfData (ndarray):      PSF image (the average PSF of the exposure)
        gsigma (float):         sigma of the Gaussian smoothing kernel in Fourier space
        thres (float):          detection threshold
        thres2 (float):         peak identification difference threshold
        klim (float):           limiting wave number in Fourier space
        coords (ndarray):       coordinates of detected peaks (x,y)
    Returns:
        peak_array (ndarray):   peak values and the shear responses
    """

    assert imgData.shape==psfData.shape, 'image and PSF should have the same\
            shape. Please do padding before using this function.'
    ny,nx   =   psfData.shape
    psfF    =   np.fft.rfft2(np.fft.ifftshift(psfData))
    gKer,(k2grid,k1grid)=gauss_kernel(ny,nx,gsigma,return_grid=True,use_rfft=True)

    # convolved images
    imgF    =   np.fft.rfft2(imgData)/psfF*gKer
    if klim>0.:
        # apply a truncation in Fourier space
        nxklim  =   int(klim*nx/np.pi/2.+0.5)
        nyklim  =   int(klim*ny/np.pi/2.+0.5)
        imgF[nyklim+1:-nyklim,:] = 0.
        imgF[:,nxklim+1:] = 0.
    else:
        # no truncation in Fourier space
        pass
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
    imgCovD1=   np.fft.irfft2(imgFD1,(ny,nx))
    del imgFD1
    imgFD2  =   imgF*(-1j*k2grid)
    imgCovD2=   np.fft.irfft2(imgFD2,(ny,nx))
    del imgFD2,imgF,k1grid,k2grid # these images take a lot of memory

    if coords is None:
        # the coordinates is not given, so we do another detection
        if not isinstance(thres,(int, float)):
            raise ValueError('thres must be float, but now got %s' %type(thres))
        coords  =   detect_coords(imgCov,thres,thres2)
    peak_array  =   _make_peak_array(coords,imgCov,imgCovQ1,imgCovQ2,imgCovD1,imgCovD2)
    del coords,imgCov,imgCovQ1,imgCovQ2,imgCovD1,imgCovD2
    return peak_array

def _make_peak_array(coords,imgCov,imgCovQ1,imgCovQ2,imgCovD1,imgCovD2):
    """Returns the peak array and the shear response of the peak array
    Args:
        coords (ndarray):     coordinate array
        imgCov (ndarray):     unsmeared image (cov) Gaussian
        imgCovQ1 (ndarray):   unsmeared image (cov) Gaussian (Q1)
        imgCovQ2 (ndarray):   unsmeared image (cov) Gaussian (Q2)
        imgCovD1 (ndarray):   unsmeared image (cov) Gaussian (D1)
        imgCovD2 (ndarray):   unsmeared image (cov) Gaussian (D2)
    Returns:
        out (ndarray):        peak array
    """
    out     =   np.array(np.zeros(coords.size),dtype=_peak_types)
    for _j,_i in _default_inds:
        # the smoothed pixel value
        _y  = coords['fpfs_peak_y']+_j-2
        _x  = coords['fpfs_peak_x']+_i-2
        _v  = imgCov[_y,_x]
        out['fpfs_f%d%d' %(_j,_i)]=_v
        # responses for the smoothed pixel value
        _r1 = imgCovQ1[_y,_x]+(_i-2.)*imgCovD1[_y,_x]-(_j-2.)*imgCovD2[_y,_x]
        _r2 = imgCovQ2[_y,_x]+(_j-2.)*imgCovD1[_y,_x]+(_i-2.)*imgCovD2[_y,_x]
        out['fpfs_f%d%dr1' %(_j,_i)]=_r1
        out['fpfs_f%d%dr2' %(_j,_i)]=_r2
    out['fpfs_peak_x']=coords['fpfs_peak_x']
    out['fpfs_peak_y']=coords['fpfs_peak_y']
    return out

def peak2det(peaks):
    """Converts peak array (merged with fpfs catalog) to detection array
    Args:
        peaks (ndarray):  peak array
    Returns:
        out (ndarray):    detection array
    """
    # prepare the output

    if 'fpfs_N22sF22r2' in peaks.dtype.names:
        noicov  =   True
        out     =   np.array(np.zeros(peaks.size),dtype=_fpfs_types+_ncov_types)
    else:
        noicov=False
        out     =   np.array(np.zeros(peaks.size),dtype=_fpfs_types)

    # column name for noise covariance for pixel values
    inm10   =   'fpfs_N22cF22r1'
    inm20   =   'fpfs_N22sF22r2'
    inm30   =   'fpfs_N00F22r1'
    inm40   =   'fpfs_N00F22r2'

    # v and two components of shear response (vr1, vr2)
    rlist   =   ['', 'r1', 'r2']
    fnmv0   =   'fpfs_f22%s'
    for ind in _default_inds:
        # the shear response of detection modes
        for rr in rlist:
            # get the detection modes from the pixel values
            fnmv =  'fpfs_f%d%d%s'  %(ind+(rr,))
            cnmv =  'fpfs_v%d%d%s'  %(ind+(rr,))
            fnmv22= fnmv0 %(rr)
            if ind  !=  (2,2):
                out[cnmv] = peaks[fnmv22]-peaks[fnmv]
            else:
                out[cnmv] = peaks[fnmv]
            del fnmv,cnmv,fnmv22

        if noicov:
            inm1    =   'fpfs_N22cF%d%dr1' %ind
            onm1    =   'fpfs_N22cV%d%dr1' %ind
            inm2    =   'fpfs_N22sF%d%dr2' %ind
            onm2    =   'fpfs_N22sV%d%dr2' %ind
            inm3    =   'fpfs_N00F%d%dr1' %ind
            onm3    =   'fpfs_N00V%d%dr1' %ind
            inm4    =   'fpfs_N00F%d%dr2' %ind
            onm4    =   'fpfs_N00V%d%dr2' %ind
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
    out['fpfs_peak_x']= peaks['fpfs_peak_x']
    out['fpfs_peak_y']= peaks['fpfs_peak_y']
    return out
