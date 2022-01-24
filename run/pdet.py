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

import numpy as np
import scipy.ndimage as ndi
import numpy.lib.recfunctions as rfn

from fpfs.imgutil import gauss_kernel

def detect_coords(imgCov,thres):
    """
    detect peaks and return the coordinates (y,x)
    Parameters:
        imgCov:     convolved image
        thres:      detection threshold
    Returns:
        ndarray of coordinates (y,x)
    """
    npixt = 1
    sizet = 1 + 2 * npixt
    footprint = np.ones((sizet, sizet))
    footprint[npixt, npixt] = 0
    filtered=ndi.maximum_filter(imgCov,footprint=footprint,mode='constant')
    data  = np.int_(np.asarray(np.where(((imgCov > filtered)&(imgCov>thres)))))
    out   = np.array(np.zeros(data.size//2),dtype=[('pdet_y','i4'),('pdet_x','i4')])
    out['pdet_y']=data[0]
    out['pdet_x']=data[1]
    return out

def get_shear_response(imgData,psfData,gsigma=3.*2*np.pi/64,separate=True,thres=0.01,coords=None):
    """
    Get the shear response for pixels identified as peaks
    Parameters:
        imgData:    observed image [ndarray]
        psfData:    PSF image (the average PSF of the exposure) [ndarray]
        gsigma:     sigma of the Gaussian smoothing kernel [float]
        separate:   whether separate responses for two shear component [bool]
        thres:      detection threshold
    Returns:
        peak values and the shear responses
    """

    assert imgData.shape==psfData.shape, 'image and PSF should have the same\
            shape. Please do padding before using this function.'
    ny,nx   =   psfData.shape
    psfF    =   np.fft.fft2(np.fft.ifftshift(psfData))

    gKer,grids=gauss_kernel(ny,nx,gsigma,return_grid=True)
    k2grid,k1grid=grids

    # convolved images
    imgF    =   np.fft.fft2(imgData)/psfF*gKer
    imgCov  =   np.fft.ifft2(imgF).real
    # Q
    imgFQ1  =   imgF*(k1grid**2.-k2grid**2.)/gsigma**2.
    imgFQ2  =   imgF*(2.*k1grid*k2grid)/gsigma**2.
    imgCovQ1=   np.fft.ifft2(imgFQ1).real
    imgCovQ2=   np.fft.ifft2(imgFQ2).real
    # D
    imgFD1  =   imgF*(-1j*k1grid)
    imgFD2  =   imgF*(-1j*k2grid)
    imgCovD1=   np.fft.ifft2(imgFD1).real
    imgCovD2=   np.fft.ifft2(imgFD2).real

    if coords is None:
        if type(thres) is not float:
            raise ValueError('If coords is none, thres must be float')
        coords  =   detect_coords(imgCov,thres)

    types   =   [('pdet_v11','>f8'),('pdet_v12','>f8'), ('pdet_v13','>f8'),\
                ('pdet_v21','>f8'), ('pdet_v22','>f8'), ('pdet_v23','>f8'),\
                ('pdet_v31','>f8'), ('pdet_v32','>f8'), ('pdet_v33','>f8')]
    if not separate:
        nn  =   [('pdet_v11r','>f8'),('pdet_v12r','>f8'),('pdet_v13r','>f8'),\
                ('pdet_v21r','>f8'), ('pdet_v22r','>f8'),('pdet_v23r','>f8'),\
                ('pdet_v31r','>f8'), ('pdet_v32r','>f8'),('pdet_v33r','>f8')]
    else:
        nn  =   [('pdet_v11r1','>f8'),('pdet_v12r1','>f8'),('pdet_v13r1','>f8'),\
                ('pdet_v21r1','>f8'), ('pdet_v22r1','>f8'),('pdet_v23r1','>f8'),\
                ('pdet_v31r1','>f8'), ('pdet_v32r1','>f8'),('pdet_v33r1','>f8'),\
                ('pdet_v11r2','>f8'), ('pdet_v12r2','>f8'),('pdet_v13r2','>f8'),\
                ('pdet_v21r2','>f8'), ('pdet_v22r2','>f8'),('pdet_v23r2','>f8'),\
                ('pdet_v31r2','>f8'), ('pdet_v32r2','>f8'),('pdet_v33r2','>f8')]
    types=types+nn

    out     =   np.array(np.zeros(coords.size),dtype=types)
    for j in range(1,4):
        for i in range(1,4):
            # the smoothed pixel value
            _y  = coords['pdet_y']+j-2
            _x  = coords['pdet_x']+i-2
            val = imgCov[_y,_x]
            out['pdet_v%d%d' %(j,i)]=val
            # responses for the smoothed pixel value
            res1= imgCovQ1[_y,_x]+(i-2.)*imgCovD1[_y,_x]-(j-2.)*imgCovD2[_y,_x]
            res2= imgCovQ2[_y,_x]+(j-2.)*imgCovD1[_y,_x]+(i-2.)*imgCovD2[_y,_x]
            if not separate:
                out['pdet_v%d%dr' %(j,i)]=(res1+res2)/2.
            else:
                out['pdet_v%d%dr1' %(j,i)]=res1
                out['pdet_v%d%dr2' %(j,i)]=res2
    out     =   rfn.merge_arrays([coords,out], flatten = True, usemask = False)
    return out
