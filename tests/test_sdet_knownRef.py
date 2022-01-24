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

import sys
sys.path.insert(1, '../run/')
import pdet
import numpy as np
import astropy.io.fits as pyfits
from fpfs import simutil


def test_centerref(ishear=1):
	"""
	Testing the pixel resopnse with referencve point set to galaxy center
	Parameters:
		ishear
	"""
	ngrid   =   64
	ngal    =   10
	img1    =   simutil.make_basic_sim('basicCenter_psf60','g%d-0000' %ishear,0,\
                ny=ngal,nx=ngal,do_write=False)
	img2    =   simutil.make_basic_sim('basicCenter_psf60','g%d-2222' %ishear,0,\
                ny=ngal,nx=ngal,do_write=False)
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
	coords  = np.array(np.zeros(inds[0].size),dtype=[('pdet_y','i4'),('pdet_x','i4')])
	coords['pdet_y']=   np.ravel(inds[0])
	coords['pdet_x']=   np.ravel(inds[1])

	out1=pdet.get_shear_response(img1,psfData,gsigma=gsigma,coords=coords)
	out2=pdet.get_shear_response(img2,psfData,gsigma=gsigma,coords=coords)
	for j in range(1,4):
		for i in range(1,4):
			resEst 	= 	(out2['pdet_v%d%d' %(j,i)]-out1['pdet_v%d%d'%(j,i)])/0.04
			res 	= 	(out2['pdet_v%d%dr%d'%(j,i,ishear)]+out1['pdet_v%d%dr%d'%(j,i,ishear)])/2.
			_ 		= 	np.average((res-resEst)/np.abs(resEst))
			np.testing.assert_almost_equal(_,0,3)
	return

if __name__ == '__main__':
    test_centerref(1)
    test_centerref(2)
