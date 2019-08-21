#!/usr/bin/env python
import sys
import numpy as np
from mt_siac import do_one_s2

def doit(fs):
    cams_dir      = '/work/scratch/marcyin/CAMS/'
    dem           = '/work/scratch/marcyin/DEM/global_dem.vrt'
    mcd43_dir     = '/work/scratch/marcyin/MCD43/'
    mcd43_vrt_dir = '/work/scratch/marcyin/MCD43_VRT/'
    jasmin = True
    ret = do_one_s2(fs, cams_dir, dem, mcd43_dir, mcd43_vrt_dir, jasmin)

if __name__ == "__main__":
    ind = int(sys.argv[1])-1
    f = np.load('toBeProcess.npz', allow_pickle=True)
    fs = f.f.fss[ind]
    doit(fs)