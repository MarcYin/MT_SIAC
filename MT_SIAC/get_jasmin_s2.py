import os
import gc
import gdal
import zipfile
import datetime
import numpy as np
from glob import glob
import multiprocessing
from collections import namedtuple
import xml.etree.ElementTree as ET

try:
    import cPickle as pkl
except:
    import pickle as pkl
file_path = '/home/users/marcyin/MT_SIAC/MT_SIAC/' #os.path.dirname(os.path.realpath(__file__))
gc.disable()
cl = pkl.load(open(file_path + '/data/sen2cloud_detector.pkl', 'rb'))
gc.enable()
cl.n_jobs = multiprocessing.cpu_count()

def do_cloud(cloud_bands, outputBounds, dstSRS):
    toas = [] 
    for band in cloud_bands:
        g = gdal.Warp('', band, outputBounds=outputBounds, xRes = 60, yRes = 60, dstNodata=0, resampleAlg=5, format='MEM', dstSRS=dstSRS)
        data = g.ReadAsArray()
        toas.append(data)
    toas = np.array(toas)/10000.
    mask = np.all(toas >= 0.0001, axis=0)
    valid_pixel = mask.sum()
    if valid_pixel>0:
        cloud_proba = cl.predict_proba(toas[:, mask].T)
        cloud_mask = np.ones_like(toas[0]) * 2.56
        cloud_mask[mask] = cloud_proba[:,1]
    else:
        cloud_mask = np.ones_like(toas[0]) * 2.56
    dst = gdal.GetDriverByName('MEM').Create('', g.RasterXSize, g.RasterYSize, 1, gdal.GDT_Byte)
    dst.SetGeoTransform(g.GetGeoTransform())
    dst.SetProjection  (g.GetProjection())
    dst.GetRasterBand(1).WriteArray((cloud_mask * 100).astype(int))
    dst.GetRasterBand(1).SetNoDataValue(256)
    return dst

def get_angle(meta):
    tree = ET.ElementTree(ET.fromstring(meta))
    root = tree.getroot()
    mvz = {} 
    mva = {} 
    msz = []
    msa = []
    for child in root: 
        for j in child: 
            for mvia in j.findall('Mean_Viewing_Incidence_Angle_List'): 
                for i in mvia.findall('Mean_Viewing_Incidence_Angle'): 
                    mvz[int(i.attrib['bandId'])] = float(i.find('ZENITH_ANGLE').text) 
                    mva[int(i.attrib['bandId'])] = float(i.find('AZIMUTH_ANGLE').text) 
            for ms in j.findall('Mean_Sun_Angle'):
                    msz = float(ms.find('ZENITH_ANGLE').text)
                    msa = float(ms.find('AZIMUTH_ANGLE').text)
    return msz, msa, mvz, mva
  
def creat_ang_g(angs, g):
    shape = (2,) + g.ReadAsArray().shape
    ang_array = np.ones(shape)
    driver = gdal.GetDriverByName('MEM')          
    dst = driver.Create('', ang_array.shape[-1], ang_array.shape[-2], 2, gdal.GDT_Int16)                       
    dst.SetGeoTransform(g.GetGeoTransform())
    dst.SetProjection  (g.GetProjection())
    for i in range(2):                      
        dst.GetRasterBand(i+1).WriteArray((ang_array[i] * angs[i] * 100).astype(int))  
    return dst

def find_on_jasmin(tiles, obs_time, outputBounds, dstSRS):
    sat = '?'
    obs_start = obs_time + datetime.timedelta(days = -15)
    obs_end = obs_time + datetime.timedelta(days =  15)
    s2s = []
    #clouds = []
    s2_obs = namedtuple('s2_obs', 'sun_angs view_angs toa cloud meta obs_time cloud_percentage ID')
    for tile in tiles:
        for i in range(30):
            this_date = (obs_start + datetime.timedelta(i)).strftime('/%Y/%m/%d/')
            ff = glob('/neodc/sentinel2' + sat + '/data/L1C_MSI' +  this_date + '*%s*' %tile +  '.zip')
            if len(ff)>0:
                for j in ff:
                    zipfname = os.path.realpath(j)
                    Zip = zipfile.ZipFile(zipfname)
                    flist = Zip.namelist()
                    bands   = ['B01', 'B02', 'B03','B04','B05' ,'B06', 'B07', 'B08','B8A', 'B09', 'B10', 'B11', 'B12']
                    s2_refs = ['/'.join(['/vsizip', zipfname, k]) for k in flist if '.jp2' in k and 'TCI' not in k and 'PVI' not in k]
                    s2_refs = sorted(s2_refs, key = lambda ref: bands.index(ref.split('.jp2')[0][-3:]))
                    metafile = [k for k in flist if ('MTD' in k) & ('TL' in k) & ('xml' in k)][0]
                    meta = Zip.read(metafile)
                    
                    sza, saa, vza, vaa = get_angle(meta)

                    vza = np.array([vza[bands.index(i)] if bands.index(i) in vza.keys() else np.nanmean(list(vza.values())) for i in bands ]) # np.nanmean(list(vza.values())) 
                    vaa = np.array([vaa[bands.index(i)] if bands.index(i) in vaa.keys() else np.nanmean(list(vaa.values()))  for i in bands ]) # np.nanmean(list(vaa.values())) 
                    
                    #s2_gmls = ['/'.join([zipfname, k]) for k in flist if '.gml' in k and 'MSK_DETFOO' in k and 'QI_DATA' in k]
                    #s2_gmls = sorted(s2_gmls, key = lambda ref: bands.index(ref.split('.gml')[0][-3:]))
                    sens_time = datetime.datetime.strptime(s2_refs[0].split('/')[-1][7:22], '%Y%m%dT%H%M%S')
                    cloud_bands = np.array(s2_refs)[[0,1,3,4,7,8,9,10,11,12]]
                    cloud_g = do_cloud(cloud_bands, outputBounds, dstSRS)
                    cloud   = cloud_g.ReadAsArray()
                    cloud_proportion = (cloud > 40).sum()/cloud.size
                    saa_sza = creat_ang_g([saa, sza], cloud_g)
                    vaa_vza = [creat_ang_g([vaa[i], vza[i]], cloud_g) for i in range(13)]
                    s2s.append(s2_obs(saa_sza, vaa_vza, s2_refs, cloud_g, zipfname + '/' + metafile, sens_time, cloud_proportion, s2_refs[0].split('/')[-5]))
                    #clouds.append(cloud_proportion)            
    return sorted(s2s, key = lambda i: i.cloud_percentage)