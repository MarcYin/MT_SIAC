#!/usr/bin/env python
from __future__ import absolute_import
import os
import gc
import sys
import gdal
import logging
import numpy as np
from glob import glob
import multiprocessing
from reproject import reproject_data
from s2_angle import resample_s2_angles
from create_logger import create_logger
from skimage.morphology import disk, binary_dilation, binary_erosion


try:
    import cPickle as pkl
except:
    import pickle as pkl
file_path = os.path.dirname(os.path.realpath(__file__))
gc.disable()
cl = pkl.load(open(file_path + '/data/sen2cloud_detector.pkl', 'rb'))
gc.enable()
cl.n_jobs = multiprocessing.cpu_count()


def do_cloud(cloud_bands, cloud_name = None):
    toas = [reproject_data(str(band), cloud_bands[0], dstNodata=0, resample=5).data for band in cloud_bands]
    toas = np.array(toas)/10000.
    mask = np.all(toas >= 0.0001, axis=0)
    valid_pixel = mask.sum()
    cloud_proba = cl.predict_proba(toas[:, mask].T)
    cloud_mask = np.zeros_like(toas[0])
    cloud_mask[mask] = cloud_proba[:,1]
    cloud_mask[~mask] = 256
    if cloud_name is None:
        return cloud_mask
    else:
        g =  gdal.Open(cloud_bands[0])
        dst = gdal.GetDriverByName('GTiff').Create(cloud_name, g.RasterXSize, g.RasterYSize, 1, gdal.GDT_Byte,  options=["TILED=YES", "COMPRESS=DEFLATE"])
        dst.SetGeoTransform(g.GetGeoTransform())
        dst.SetProjection  (g.GetProjection())
        dst.GetRasterBand(1).WriteArray((cloud_mask * 100).astype(int))
        dst.GetRasterBand(1).SetNoDataValue(256)
        dst=None; g=None 
        return cloud_mask


def s2_pre_processing(s2_dir):
    s2_dir = os.path.abspath(s2_dir)
    scihub = []
    aws    = []
    logger = create_logger()
    logger.propagate = False
    for (dirpath, dirnames, filenames)  in os.walk(s2_dir):
        if len(filenames)>0:
            temp = [dirpath + '/' + i for i in filenames]
            for j in temp:
                if ('MTD' in j) & ('TL' in j) & ('xml' in j):
                    scihub.append(j)
                if 'metadata.xml' in j:
                    aws.append(j)
    s2_tiles = []
    for metafile in scihub + aws:
        with open(metafile) as f:
            for i in f.readlines(): 
                if 'TILE_ID' in i:
                    tile  = i.split('</')[0].split('>')[-1]
        top_dir = os.path.dirname(metafile) 
        log_file = top_dir + '/SIAC_S2.log'
        if os.path.exists(log_file):
            os.remove(log_file)
        logger.info('Preprocessing for %s'%tile)
        logger.info('Doing per pixel angle resampling.')
        bands    = ['B01', 'B02', 'B03','B04','B05' ,'B06', 'B07', 'B08','B8A', 'B09', 'B10', 'B11', 'B12']
        if len(glob(top_dir + '/ANG_DATA/*.tif')) == 14:
            sun_ang_name   = top_dir + '/ANG_DATA/SAA_SZA.tif'
            view_angs      = glob(top_dir + '/ANG_DATA/VAA_VZA_B??.tif')
            toa_refs       = glob(top_dir + '/IMG_DATA/*B??.jp2')
            view_ang_names = sorted(view_angs, key = lambda fname: bands.index(fname.split('_')[-1].split('.tif')[0]))
            toa_refs       = sorted(toa_refs,  key = lambda toa_ref: bands.index('B' + toa_ref.split('B')[-1][:2]))
            cloud_name     = top_dir + '/cloud.tif'
        else:
            sun_ang_name, view_ang_names, toa_refs, cloud_name = resample_s2_angles(metafile)
        cloud_bands = np.array(toa_refs)[[0,1,3,4,7,8,9,10,11,12]]
        logger.info('Getting cloud mask.')
        if not os.path.exists(cloud_name):
            cloud = do_cloud(cloud_bands, cloud_name)
        else:
            cloud = gdal.Open(cloud_name).ReadAsArray()
        cloud_mask = (cloud > 40) & (cloud != 256)# 0.4 is the sentinel hub default
        clean_pixel = ((cloud != 256).sum() - cloud_mask.sum()) / cloud_mask.size * 100.
        valid_pixel =  (cloud != 256).sum() / cloud_mask.size * 100.
        logger.info('Valid pixel percentage: %.02f'%(valid_pixel))
        logger.info('Clean pixel percentage: %.02f'%(clean_pixel))
        cloud_mask = binary_dilation(binary_erosion (cloud_mask, disk(2)), disk(3)) | (cloud < 0)
        s2_tiles.append([sun_ang_name, view_ang_names, toa_refs, cloud_name, cloud_mask, metafile])
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
    return s2_tiles
