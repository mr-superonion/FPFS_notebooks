import numpy as np
import astropy.io.fits as pyfits
from fpfs import simutil
from fpfs.imgutil import gauss_kernel


def test_knownref(ishear=1):
    """
    Testing the pixel resopnse with known referencve point
    Parameters:
        ishear
    """
    ngal    =   1
    ngrid   =   64
    ngrid2  =   ngrid*ngal
    img1    =   simutil.make_basic_sim('basic_psf60','g%d-0000'%ishear,0,\
                ny=ngal,nx=ngal,do_write=False)
    img2    =   simutil.make_basic_sim('basic_psf60','g%d-2222'%ishear,0,\
                ny=ngal,nx=ngal,do_write=False)

    # PSF
    psf     =   pyfits.getdata('psf-60.fits')
    npad    =   (ngrid2-psf.shape[0])//2
    psfData =   np.pad(psf,(npad+1,npad),mode='constant')
    assert psfData.shape[0]==ngrid2
    psfF    =   np.fft.fft2(np.fft.ifftshift(psfData))
    gsigma  =   3.*2.*np.pi/64

    gKer,grids=gauss_kernel(ngrid2,ngrid2,gsigma,return_grid=True)
    k2grid,k1grid=grids

    imgF1   =   np.fft.fft2(img1)/psfF*gKer
    imgFQA1 =   imgF1*(k1grid**2.-k2grid**2.)/gsigma**2.
    imgFQB1 =   imgF1*(2.*k1grid*k2grid)/gsigma**2.
    imgFDA1 =   imgF1*(-1j*k1grid)
    imgFDB1 =   imgF1*(-1j*k2grid)

    imgF2   =   np.fft.fft2(img2)/psfF*gKer
    imgFQA2 =   imgF2*(k1grid**2.-k2grid**2.)/gsigma**2.
    imgFQB2 =   imgF2*(2.*k1grid*k2grid)/gsigma**2.
    imgFDA2 =   imgF2*(-1j*k1grid)
    imgFDB2 =   imgF2*(-1j*k2grid)

    imgCov1 =   np.fft.ifft2(imgF1).real
    imgCovQA1=  np.fft.ifft2(imgFQA1).real
    imgCovQB1=  np.fft.ifft2(imgFQB1).real
    imgCovDA1=  np.fft.ifft2(imgFDA1).real
    imgCovDB1=  np.fft.ifft2(imgFDB1).real

    imgCov2 =   np.fft.ifft2(imgF2).real
    imgCovQA2=  np.fft.ifft2(imgFQA2).real
    imgCovQB2=  np.fft.ifft2(imgFQB2).real
    imgCovDA2=  np.fft.ifft2(imgFDA2).real
    imgCovDB2=  np.fft.ifft2(imgFDB2).real

    ind1    =   (32,32)
    ind2    =   (32,32)

    if ishear==1:
        res1    =   imgCovQA1[ind1]+0.5*imgCovDA1[ind1]-0.5*imgCovDB1[ind1]
        res2    =   imgCovQA2[ind2]+0.5*imgCovDA2[ind2]-0.5*imgCovDB2[ind2]
        meas1   =   imgCov1[ind1]
        meas2   =   imgCov2[ind2]
    elif ishear==2:
        res1    =   imgCovQB1[ind1]+0.5*imgCovDA1[ind1]+0.5*imgCovDB1[ind1]
        res2    =   imgCovQB2[ind2]+0.5*imgCovDA2[ind2]+0.5*imgCovDB2[ind2]
        meas1   =   imgCov1[ind1]
        meas2   =   imgCov2[ind2]
    else:
        raise(ValueError('ishear should be either 1 or 2'))

    response=   np.average((res1+res2)/2.)
    resEst  =   np.average((meas2-meas1)/0.04)
    out     =   np.abs((response-resEst)/response)
    np.testing.assert_almost_equal(out, 0, 2)
    return

if __name__ == '__main__':
    test_knownref(1)
    test_knownref(2)
