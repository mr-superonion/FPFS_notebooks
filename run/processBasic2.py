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
import psutil
import numpy as np
from fpfs import simutil
from fpfs import fpfsBase
import astropy.io.fits as pyfits
import pdet
import numpy.lib.recfunctions as rfn

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
        default="galaxy_basicCenter_psf60",#"small2_psf60",
        doc="Input galaxy directory"
    )
    noiName     = pexConfig.Field(
        dtype=str,
        default="var0em0",#"var7em3",
        doc="noise variance name"
    )
    outDir     = pexConfig.Field(
        dtype=str,
        default="",
        doc="output directory"
    )

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        self.readDataSim.doWrite=   False
        self.readDataSim.doDeblend= True
        self.readDataSim.doAddFP=   False
        psfFWHM =   self.galDir.split('_psf')[-1]
        if 'small' in self.galDir:
            irr     =   eval(self.galDir.split('_psf')[0].split('small')[-1])
            gnm     =   'Small%d' %irr
        elif 'star' in self.galDir:
            gnm     =   'Star'
        else:
            gnm     =   'Basic'
        self.outDir  =   os.path.join('out%s-%s' %(gnm,self.noiName),'psf%s'%(psfFWHM))

    def validate(self):
        assert os.path.exists('noise')
        assert os.path.exists(self.galDir)
        if not os.path.isdir(self.outDir):
            os.mkdir(self.outDir)

class processBasicRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minIndex    =  parsedCmd.minIndex
        maxIndex    =  parsedCmd.maxIndex
        return [(ref, kwargs) for ref in range(minIndex,maxIndex)]
def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)
class processBasicDriverTask(BatchPoolTask):
    ConfigClass = processBasicDriverConfig
    RunnerClass = processBasicRunner
    _DefaultName = "processBasicDriver"
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))
    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.schema     =   afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("readDataSim",schema=self.schema)

    @pipeBase.timeMethod
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

    @pipeBase.timeMethod
    def process(self,cache,ifield):
        # Basic
        nn          =   100
        ngrid       =   64
        ngrid2      =   ngrid*nn

        beta        =   0.75
        gsigma      =   6.*2.*np.pi/64
        pixScale    =   0.168

        # necessary directories
        # galDir      =   'galaxy_basic_psf%s' %psfFWHM
        galDir      =   cache.galDir
        psfFWHM     =   galDir.split('_psf')[-1]
        #psfFWHMF    =   eval(psfFWHM)/100.
        rcut        =   16#max(min(int(psfFWHMF/pixScale*4+0.5),15),12)
        beg         =   ngrid//2-rcut
        end         =   beg+2*rcut
        if 'small' in galDir:
            if "var0em0" not in cache.outDir:
                igroup  =   ifield//8
            else:
                igroup  =   ifield
        elif 'star' in galDir:
            if "var0em0" not in cache.outDir:
                raise ValueError("stars do not support noiseless simulations")
            igroup  =   0
        else:
            if "var0em0" not in cache.outDir:
                igroup  =   ifield//4
            else:
                igroup  =   ifield
        self.log.info('running for group: %s, field: %s' %(igroup,ifield))

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
            noiVar      =   7e-3
            noiFname    =   os.path.join('noise','noi%04d.fits' %ifield)
            if not os.path.isfile(noiFname):
                self.log.info('Cannot find input noise file: %s' %noiFname)
                return
            # multiply by 10 since the noise has variance 0.01
            noiData     =   pyfits.open(noiFname)[0].data*10.
            # same for the noivar model
            powIn       =   np.load('corPre/noiPows2.npy',allow_pickle=True).item()['%s'%rcut]*noiVar*100
            powModel    =   np.zeros((1,powIn.shape[0],powIn.shape[1]))
            powModel[0] =   powIn
            fpTask      =   fpfsBase.fpfsTask(psfData2,noiFit=powModel[0],beta=beta)
        else:
            noiVar      =   1e-20
            self.log.info('noiseless setup')
            fpTask      =   fpfsBase.fpfsTask(psfData2,beta=beta)
            noiData     =   None

        # isList      =   ['g1-0000','g2-0000','g1-2222','g2-2222']
        # isList      =   ['g1-1111']
        isList      =   ['g1-0000','g1-2222']
        # isList        =   ['g1-0000']
        for ishear in isList:
            galFname    =   os.path.join(galDir,'image-%s-%s.fits' %(igroup,ishear))
            if not os.path.isfile(galFname):
                self.log.info('Cannot find input galaxy file: %s' %galFname)
                return
            galData     =   pyfits.getdata(galFname)
            if noiData is not None:
                galData =   galData+noiData*np.sqrt(noiVar)

            outFname    =   os.path.join(cache.outDir,'src-%04d-%s.fits' %(ifield,ishear))
            if not os.path.exists(outFname) and cache.doHSM:
                self.log.info('HSM measurement: %04d, %s' %(ifield,ishear))
                exposure=   simutil.makeHSCExposure(galData,psfData,pixScale,noiVar)
                src     =   self.readDataSim.measureSource(exposure)
                wFlag   =   afwTable.SOURCE_IO_NO_FOOTPRINTS
                src.writeFits(outFname,flags=wFlag)
                del exposure,src
                gc.collect()
            else:
                self.log.info('Skip HSM measurement: %04d, %s' %(ifield,ishear))
            self.log.info('The memory used is: %.3f' %(psutil.Process().memory_info().rss/1024**3.))

            outFname    =   os.path.join(cache.outDir,'fpfs-cut%d-%04d-%s.fits' %(rcut,ifield,ishear))
            if not os.path.exists(outFname) and cache.doFPFS:
                self.log.info('FPFS measurement: %04d, %s' %(ifield,ishear))
                if 'basicCenter' in galDir:
                    indX    =   np.arange(32,ngrid2,64)
                    indY    =   np.arange(32,ngrid2,64)
                    inds    =   np.meshgrid(indY,indX,indexing='ij')
                    coords  =   np.array(np.zeros(inds[0].size),dtype=[('pdet_y','i4'),('pdet_x','i4')])
                    coords['pdet_y']=   np.ravel(inds[0])
                    coords['pdet_x']=   np.ravel(inds[1])
                    del indX,indY,inds
                else:
                    coords  =   []
                    pass
                imgList =   [galData[cc['pdet_y']-rcut:cc['pdet_y']+rcut,\
                            cc['pdet_x']-rcut:cc['pdet_x']+rcut] for cc in coords]
                out     =   fpTask.measure(imgList)
                out1    =   pdet.get_shear_response(galData,psfData3,gsigma=gsigma,\
                            coords=coords)
                out     =   rfn.merge_arrays([out,out1],flatten=True,usemask=False)
                pyfits.writeto(outFname,out)
                del out,imgList,coords,out1
                gc.collect()
            else:
                self.log.info('Skip FPFS measurement: %04d, %s' %(ifield,ishear))
            self.log.info('The memory used is: %.3f' %(psutil.Process().memory_info().rss/1024**3.))

            del galData,outFname
            gc.collect()
        self.log.info('finish %s' %(ifield))
        return

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

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numCpus):
        return None
    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass
    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass
    def writeMetadata(self, ifield):
        pass
    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass
    def _getConfigName(self):
        return None
    def _getEupsVersionsName(self):
        return None
    def _getMetadataName(self):
        return None
