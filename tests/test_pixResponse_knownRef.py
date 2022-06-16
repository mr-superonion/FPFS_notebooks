# shear response of detection
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

import fpfs
import pdet
import numpy as np
import astropy.io.fits as pyfits



def test_centerref(ishear=1):
    """
    Testing the pixel resopnse with referencve point set to galaxy center
    Parameters:
        ishear
    """
    ngrid   =   64
    ngal    =   10
    img1    =   fpfs.simutil.make_basic_sim('basicCenter2_psf60','g%d-0000' %ishear,0,\
                ny=ngal,nx=ngal,do_write=False,return_array=True)
    img2    =   fpfs.simutil.make_basic_sim('basicCenter2_psf60','g%d-2222' %ishear,0,\
                ny=ngal,nx=ngal,do_write=False,return_array=True)
    ngrid2  =   ngrid*ngal
    # PSF
    psf     =   pyfits.getdata('psf-60.fits')
    npad    =   (ngrid2-psf.shape[0])//2
    psfData =   np.pad(psf,(npad+1,npad),mode='constant')
    assert psfData.shape[0]==ngrid2
    gsigma  =   6.*2.*np.pi/64

    indX    =   np.arange(32,ngal*64,64)
    indY    =   np.arange(32,ngal*64,64)
    inds    =   np.meshgrid(indY,indX,indexing='ij')
    coords  = np.array(np.zeros(inds[0].size),dtype=[('fpfs_y','i4'),('fpfs_x','i4')])
    coords['fpfs_y']=   np.ravel(inds[0])
    coords['fpfs_x']=   np.ravel(inds[1])

    out1=pdet.get_shear_response_rfft(img1,psfData,gsigma=gsigma,coords=coords)
    out2=pdet.get_shear_response_rfft(img2,psfData,gsigma=gsigma,coords=coords)
    for (j,i) in fpfs.base.det_inds:
        resEst 	= 	(out2['fpfs_f%d%d' %(j,i)]-out1['fpfs_f%d%d'%(j,i)])/0.04
        res 	= 	(out2['fpfs_f%d%dr%d'%(j,i,ishear)]+out1['fpfs_f%d%dr%d'%(j,i,ishear)])/2.
        mask    =   np.abs(resEst)>np.average(np.abs(resEst))/8.
        _ 		= 	np.average((res-resEst)/np.abs(resEst),weights=mask.astype(float))
        np.testing.assert_almost_equal(_,0,3)
    return

if __name__ == '__main__':
    test_centerref(1)
    test_centerref(2)
