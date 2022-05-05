import gc
import numpy as np

def get_detbias(dets,ells,icut,bcut,ind):
    """
    Parameters:
        dets:       detection array
        icut:       selection cut
        bcut:       bin size when estimating marginal density
        inds:       pixel index
    """
    fnmv   =  'pdet_v%d%d'  %ind
    fnmr1  =  'fpfs_e1v%d%dr1'%ind
    fnmr2  =  'fpfs_e2v%d%dr2'%ind
    ll     =  icut;uu  =  icut+bcut
    msk    =  (dets[fnmv]>ll)&(dets[fnmv]<uu)
    cor1=np.sum(ells[fnmr1][msk])/bcut
    cor2=np.sum(ells[fnmr2][msk])/bcut
    return cor1,cor2

def get_selbias(ells,cut,bcut):
    """
    Parameters:
        cut:        selection cut
        bcut:       bin size when estimating marginal density
        inds:       pixel index
    """
    fnmv   =  'fpfs_s0'
    fnmr1  =  'fpfs_RS'
    fnmr2  =  'fpfs_RS'
    ll     =  cut;uu  =  cut+bcut
    msk    =  (ells[fnmv]>ll)&(ells[fnmv]<uu)
    cor1=np.sum(ells[fnmr1][msk])/bcut
    cor2=np.sum(ells[fnmr2][msk])/bcut
    return cor1,cor2

def get_detbias_all(dets,ells,cutd,cuts,bcut,ind):
    """
    Parameters:
        dets:       detection array
        cut:        selection cut
        bcut:        bin size when estimating marginal density
        inds:       pixel index
    """
    cord1,cord2=get_detbias(dets,ells,cutd,bcut,ind)
    fnmv   =  'pdet_v%d%d'  %ind
    ells2= ells[dets[fnmv]>cutd+bcut]
    cors1,cors2=get_selbias(ells2,cuts,bcut)
    return cord1+cors1,cord2+cors2

def get_detbias_list(dets,ells,indsl,cutsl,bcutl):
    """
    get detection bias due to lower boundar cuts
    Parameters:
        dets:       ndarray
                    detection array
        ells:       ndarray
                    ellipticity array
        indsl:      list
                    a list of pixel index
        cutsl:      list
                    a list of selection cut
        bcutl:       list
                    a list of bin size when estimating marginal density
    """
    ncut = len(indsl) # assert
    assert len(cutsl)==ncut, 'number of cuts does not match'
    assert len(bcutl)==ncut, 'number of boundary bin size does not match'

    cor1Sum=0.  # initialize the correction terms
    cor2Sum=0.
    for _ in range(ncut):
        fnmv =  'pdet_v%d%d' %indsl[_]
        bcut  =  bcutl[_]
        icut  =  cutsl[_]
        print('apply cut on %s at %.3f' %(fnmv,icut))
        cor1,cor2=get_detbias(dets,ells,icut,bcut,indsl[_])
        cor1Sum+=cor1
        cor2Sum+=cor2
        _m=dets[fnmv]>=icut+bcut
        dets=dets[_m]
        ells=ells[_m]
        print(len(dets))
        del _m
        gc.collect()
    return cor1Sum,cor2Sum
