#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 20082014 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib
import os
import gc
import pdet
import fpfs
import numpy as np
import astropy.io.fits as pyfits
import numpy.lib.recfunctions as rfn
from lsst.utils.timer import timeMethod

from readDataSim import readDataSimTask
# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.table as afwTable

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.pool import Pool
from lsst.ctrl.pool.parallel import BatchPoolTask

class processBasicDriverConfig(pexConfig.Config):
    doHSM   = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether run HSM",
    )
    doFPFS  = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether run FPFS",
    )
    readDataSim= pexConfig.ConfigurableField(
        target  = readDataSimTask,
        doc     = "Subtask to run measurement hsm"
    )
    galDir      = pexConfig.Field(
        dtype=str,
        default="galaxy_basic2Center_psf60",#"small2_psf60",
        doc="Input galaxy directory"
    )
    noiName     = pexConfig.Field(
        dtype=str,
        # default="var1em9",
        # default="var0em0",
        default="var4em3",
        doc="noise variance name"
    )
    inDir       = pexConfig.Field(
        dtype=str,
        default="./",
        doc="input directory"
    )
    outDir      = pexConfig.Field(
        dtype=str,
        default="./",
        doc="output directory"
    )

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        self.readDataSim.doWrite=   False
        self.readDataSim.doDeblend= True
        self.readDataSim.doAddFP=   False
        tname   =   'try4'
        psfFWHM =   self.galDir.split('_psf')[-1]
        gnm     =   self.galDir.split('galaxy_')[-1].split('_psf')[0]
        self.outDir  =  os.path.join(self.outDir,'srcfs2_%s-%s_%s' %(gnm,self.noiName,tname),'psf%s'%(psfFWHM))
        self.galDir  =  os.path.join(self.inDir,self.galDir)

    def validate(self):
        assert os.path.exists(os.path.join(self.inDir,'noise'))
        assert os.path.exists(self.galDir)
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)

class processBasicRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minIndex    =  parsedCmd.minIndex
        maxIndex    =  parsedCmd.maxIndex
        return [(ref, kwargs) for ref in range(minIndex,maxIndex)]

class processBasicDriverTask(BatchPoolTask):
    ConfigClass = processBasicDriverConfig
    RunnerClass = processBasicRunner
    _DefaultName = "processBasicDriver"

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument('--minIndex', type= int,
                        default=0,
                        help='minimum Index number')
        parser.add_argument('--maxIndex', type= int,
                        default=1,
                        help='maximum Index number')
        return parser

    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.schema     =   afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("readDataSim",schema=self.schema)

    @timeMethod
    def runDataRef(self,index):
        #Prepare the pool
        pool    =   Pool("processBasic")
        pool.cacheClear()
        perBatch=   200
        pool.storeSet(doHSM=self.config.doHSM)
        pool.storeSet(doFPFS=self.config.doFPFS)
        pool.storeSet(galDir=self.config.galDir)
        pool.storeSet(outDir=self.config.outDir)
        fieldList=np.arange(perBatch*index,perBatch*(index+1))
        pool.map(self.process,fieldList)
        return

    @timeMethod
    def process(self,cache,nid):

        # necessary directories
        nn          =   100
        ngrid       =   64
        ngrid2      =   ngrid*nn
        pixScale    =   0.168
        galDir      =   cache.galDir
        psfFWHM     =   galDir.split('_psf')[-1]
        #psfFWHMF    =   eval(psfFWHM)/100.

        # FPFS Basic
        # gsigma    =   6.*2.*np.pi/64. # try1 --- this is very unstable
        # gsigma    =   3.*2.*np.pi/64. # try2
        # beta      =   0.75# try3
        beta        =   0.50# try4
        rcut        =   16#max(min(int(psfFWHMF/pixScale*4+0.5),15),12)
        beg         =   ngrid//2-rcut
        end         =   beg+2*rcut

        if 'small' in galDir:
            self.log.info('Using small galaxies')
            if "var0em0" not in cache.outDir:
                gid  =   nid//8
            else:
                gid  =   nid
        elif 'star' in galDir:
            self.log.info('Using stars')
            if "var0em0" not in cache.outDir:
                raise ValueError("stars do not support noiseless simulations")
            gid  =   0
        elif 'basic1' in galDir:
            # for COSMOS galaxies, 4 noise realizations share one galaxy
            self.log.info('Using cosmosis parametric galaxies v1.')
            gid  =   nid//4
        elif 'basic2' in galDir:
            self.log.info('Using cosmosis parametric galaxies v2.')
            gid  =  nid
        else:
            raise ValueError("galDir should cantain either 'small', 'star' or basic1/2")
        self.log.info('running for galaxy field: %s, noise field: %s' %(gid,nid))

        # PSF
        psfFname    =   os.path.join(galDir,'psf-%s.fits' %psfFWHM)
        psfData     =   pyfits.open(psfFname)[0].data
        npad        =   (ngrid-psfData.shape[0])//2
        psfData2    =   np.pad(psfData,(npad+1,npad),mode='constant')
        psfData2    =   psfData2[beg:end,beg:end]
        # PSF2
        npad        =   (ngrid2-psfData.shape[0])//2
        psfData3    =   np.pad(psfData,(npad+1,npad),mode='constant')

        # FPFS Task
        if "var0em0" not in cache.outDir:
            # noise
            _tmp        =   cache.outDir.split('var')[-1]
            noiVar      =   eval(_tmp[0])*10**(-1.*eval(_tmp[3]))
            self.log.info('noisy setup with variance: %.3f' %noiVar)
            noiFname    =   os.path.join('noise','noi%04d.fits' %nid)
            if not os.path.isfile(noiFname):
                self.log.info('Cannot find input noise file: %s' %noiFname)
                return
            # multiply by 10 since the noise has variance 0.01
            noiData     =   pyfits.open(noiFname)[0].data*10.*np.sqrt(noiVar)
            # Also times 100 for the noivar model
            powIn       =   np.load('corPre/noiPows2.npy',allow_pickle=True).item()['%s'%rcut]*noiVar*100
            powModel    =   np.zeros((1,powIn.shape[0],powIn.shape[1]))
            powModel[0] =   powIn
            fpTask      =   fpfs.base.fpfsTask(psfData2,noiFit=powModel[0],beta=beta)
        else:
            noiVar      =   1e-20
            self.log.info('We are using noiseless setup')
            # by default noiFit=None
            fpTask      =   fpfs.base.fpfsTask(psfData2,beta=beta)
            noiData     =   None
        # self.log.info('%s' %(fpTask.klim/fpTask._dk))
        # self.log.info('%s' %fpTask.sigmaF)
        # isList        =   ['g1-0000','g2-0000','g1-2222','g2-2222']
        # isList        =   ['g1-1111']
        isList          =   ['g1-0000','g1-2222']
        # isList        =   ['g1-0000']
        for ishear in isList:
            galFname    =   os.path.join(galDir,'image-%s-%s.fits' %(gid,ishear))
            if not os.path.isfile(galFname):
                self.log.info('Cannot find input galaxy file: %s' %galFname)
                return
            galData     =   pyfits.getdata(galFname)
            if noiData is not None:
                galData =   galData+noiData

            outFname    =   os.path.join(cache.outDir,'src-%04d-%s.fits' %(nid,ishear))
            if not os.path.exists(outFname) and cache.doHSM:
                self.log.info('HSM measurement: %04d, %s' %(nid,ishear))
                exposure=   fpfs.simutil.makeLsstExposure(galData,psfData,pixScale,noiVar)
                src     =   self.readDataSim.measureSource(exposure)
                wFlag   =   afwTable.SOURCE_IO_NO_FOOTPRINTS
                src.writeFits(outFname,flags=wFlag)
                del exposure,src
                gc.collect()
            else:
                self.log.info('Skip HSM measurement: %04d, %s' %(nid,ishear))
            pp  =   'cut%d' %rcut
            outFname    =   os.path.join(cache.outDir,'fpfs-%s-%04d-%s.fits' %(pp,nid,ishear))
            if not os.path.exists(outFname) and cache.doFPFS:
                self.log.info('FPFS measurement: %04d, %s' %(nid,ishear))
                if 'Center' in galDir and 'det' not in pp:
                    # fake detection
                    indX    =   np.arange(32,ngrid2,64)
                    indY    =   np.arange(32,ngrid2,64)
                    inds    =   np.meshgrid(indY,indX,indexing='ij')
                    coords  =   np.array(np.zeros(inds[0].size),dtype=[('fpfs_peak_y','i4'),('fpfs_peak_x','i4')])
                    coords['fpfs_peak_y']=   np.ravel(inds[0])
                    coords['fpfs_peak_x']=   np.ravel(inds[1])
                    del indX,indY,inds
                else:
                    coords  =   None
                thres   =   max(np.sqrt(noiVar)*2.5,0.05)
                out1    =   pdet.get_shear_response_rfft(galData,psfData3,gsigma=fpTask.sigmaF,\
                            thres=thres,thres2=-0.004,klim=fpTask.klim,coords=coords)
                gc.collect()
                self.log.info('number of sources: %d' %len(out1))
                imgList =   [galData[cc['fpfs_peak_y']-rcut:cc['fpfs_peak_y']+rcut,\
                            cc['fpfs_peak_x']-rcut:cc['fpfs_peak_x']+rcut] for cc in out1]
                out     =   fpTask.measure(imgList)
                out     =   rfn.merge_arrays([out,out1],flatten=True,usemask=False)
                pyfits.writeto(outFname,out)
                del imgList,out,out1
                gc.collect()
            else:
                self.log.info('Skip FPFS measurement: %04d, %s' %(nid,ishear))
            # self.log.info('The memory used is: %.3f' %(psutil.Process().memory_info().rss/1024**3.))
            del galData,outFname
            gc.collect()
        self.log.info('finish %s' %(nid))
        return

    def _getConfigName(self):
        """It's not worth preserving the configuration"""
        return None
    def _getMetadataName(self):
        """There's no metadata to write out"""
        return None
