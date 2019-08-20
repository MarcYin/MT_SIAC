#/usr/bin/env python
from __future__ import absolute_import
import os
from glob import glob
from l8_angle import do_l8_angle

def l8_pre_processing(l8_dir, temp_angle = False, l8_angle_dir = '/home/users/marcyin/acix/l8_angle/'):
    metafiles = []
    for (dirpath, dirnames, filenames)  in os.walk(l8_dir):
        if len(filenames)>0:
            temp = [dirpath + '/' + i for i in filenames]
            for j in temp:
                if 'mtl.' in j.lower():
                    metafiles.append(j)
    l8_tiles = []
    for metafile in metafiles:
        ret = do_l8_angle(metafile, temp_angle, l8_angle_dir)
        l8_tiles.append(ret)
    return l8_tiles
