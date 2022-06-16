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

def truncSine_func(x,mu=0.,sigma=1.5,deriv=0):
    """Truncated sine function used for selection
    """
    t=(x-mu)/sigma
    if deriv==0:
        return np.piecewise(t, [t<-1,(t>=-1)&(t<=1),t>1],\
                [0.,lambda t:1./2.+np.sin(t*np.pi/2.)/2., 1.])
    elif deriv==1:
        return np.piecewise(t, [t<-1,(t>=-1)&(t<=1),t>1],\
                [0.,lambda t: np.pi/2./sigma*np.cos(t*np.pi/2.)/(1.+np.sin(t*np.pi/2.)), 0.] )

def sigmoid_func(x,mu=0.,sigma=1.5,deriv=0):
    """sigmoid function used for selection
    """
    expx=np.exp(-(x-mu)/sigma)
    if deriv==0:
        # sigmoid function
        return 1./(1. + expx)
    elif deriv==1:
        # multiplicative factor for the first derivative
        return 1./sigma*expx/(1. + expx)

def get_detbias(ells,w_sel,icut,isig,ind,use_sig=False):
    """
    Get detection bias due to a cut

    Parameters:
        ells (ndarray):     ellipticity array
        w_sel (ndarray):    selection weight
        icut (float):       selection cut
        isig (float):       sigma of sigmoid function
        inds (tuple):       pixel index

    Returns:
        cor1 (float):       correction for shear1
        cor2 (float):       correction for shear2
    """
    fnmv   =  'fpfs_v%d%d'  %ind
    fnmr1  =  'fpfs_e1v%d%dr1'%ind
    fnmr2  =  'fpfs_e2v%d%dr2'%ind
    if use_sig:
        wselb  =  sigmoid_func(ells[fnmv],mu=icut,sigma=isig,deriv=1)
    else:
        wselb  =  truncSine_func(ells[fnmv],mu=icut,sigma=isig*3,deriv=1)
    out1=np.sum(ells[fnmr1]*wselb*w_sel)
    out2=np.sum(ells[fnmr2]*wselb*w_sel)
    return out1,out2

def get_detbias_list(ells,w_sel,indsl,cutsl,bcutl):
    """
    Get detection bias due to a list of lower boundary cuts

    Parameters:
        ells (ndarray):     ellipticity array
        w_sel (ndarray):    selection weight
        indsl (list):       a list of pixel index
        cutsl (list):       a list of selection cuts (sigmoid mu)
        bcutl (list):       a list of width of sigmoid functions

    Returns:
        cor1 (float):       correction for shear1
        cor2 (float):       correction for shear2
    """
    ncut = len(indsl) # assert
    assert len(cutsl)==ncut, 'number of cuts does not match'
    assert len(bcutl)==ncut, 'number of boundary bin size does not match'

    cor1Sum=0.  # initialize the correction terms
    cor2Sum=0.
    for _ in range(ncut):
        fnmv =  'fpfs_v%d%d' %indsl[_]
        bcut  =  bcutl[_]
        icut  =  cutsl[_]
        print('apply cut on %s at %.3f' %(fnmv,icut))
        c1,c2=get_detbias(ells,w_sel,icut,bcut,indsl[_])
        cor1Sum+=c1
        cor2Sum+=c2
    return cor1Sum,cor2Sum
