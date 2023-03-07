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

import pdet
import fpfs
import numpy as np
import astropy.io.fits as pyfits
import numpy.lib.recfunctions as rfn

def test_peak2det():
    """
    Testing the pixel resopnse with referencve point set to galaxy center
    Parameters:
        ishear
    """
    ngal    =   1
    ngrid   =   64
    img     =   fpfs.simutil.make_basic_sim('basicCenter_psf60','g1-0000',0,\
                ny=ngal,nx=ngal,do_write=False,return_array=True)
    ngrid2  =   ngrid*ngal
    rcut    =   16
    beg     =   ngrid//2-rcut
    end     =   beg+2*rcut
    gsigma  =   6.*2.*np.pi/64
    beta    =   0.75
    # PSF
    psf     =   pyfits.getdata('psf-60.fits')
    # pad to ngrid2 x ngrid2
    npad    =   (ngrid2-psf.shape[0])//2
    psfData =   np.pad(psf,(npad+1,npad),mode='constant')
    assert psfData.shape[0]==ngrid2
    fpTask0 =   fpfs.base.fpfsTask(psfData[beg:end,beg:end],beta=beta)
    a0      =   fpTask0.measure(img[beg:end,beg:end])

    indX    =   np.arange(32,ngal*64,64)
    indY    =   np.arange(32,ngal*64,64)
    inds    =   np.meshgrid(indY,indX,indexing='ij')
    coords  =   np.array(np.zeros(inds[0].size),dtype=[('fpfs_y','i4'),('fpfs_x','i4')])
    coords['fpfs_y']=   np.ravel(inds[0])
    coords['fpfs_x']=   np.ravel(inds[1])

    b0      =   pdet.get_shear_response_rfft(img,psfData,gsigma=gsigma,coords=coords)
    out0    =   rfn.merge_arrays([a0,b0],flatten=True,usemask=False)
    out1    =   pdet.peak2det(out0)

    cnmv0   =   'fpfs_f22'
    cnmr10  =   'fpfs_f22r1'
    cnmr20  =   'fpfs_f22r2'
    for ind in pdet._default_inds:
        cnmv=   'fpfs_f%d%d'  %ind
        cnmr1=  'fpfs_f%d%dr1'%ind
        cnmr2=  'fpfs_f%d%dr2'%ind
        vnmv=   'fpfs_v%d%d'  %ind
        vnmr1=  'fpfs_v%d%dr1'%ind
        vnmr2=  'fpfs_v%d%dr2'%ind
        if ind  !=  (2,2):
            _=out1[vnmv]-(out0[cnmv0]-out0[cnmv])
            np.testing.assert_almost_equal(_,0,4)
            _=out1[vnmr1]-(out0[cnmr10]-out0[cnmr1])
            np.testing.assert_almost_equal(_,0,4)
            _=out1[vnmr2]-(out0[cnmr20]-out0[cnmr2])
            np.testing.assert_almost_equal(_,0,4)
        else:
            _=out1[vnmv]-out0[cnmv]
            np.testing.assert_almost_equal(_,0,4)
            _=out1[vnmr1]-out0[cnmr1]
            np.testing.assert_almost_equal(_,0,4)
            _=out1[vnmr1]-out0[cnmr1]
            np.testing.assert_almost_equal(_,0,4)
    return

if __name__ == '__main__':
    test_peak2det()
