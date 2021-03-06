import os
import sys
import osr
import ogr
import mgrs
import gdal
import shutil
import psutil
import requests
import datetime
import kernels
from numba import jit
from io import StringIO
import pandas as pd
import numpy as np
from glob import glob
import multiprocessing
import lightgbm as lgb
from Two_NN import Two_NN
from smoothn import smoothn
from functools import partial
# from Get_S2 import get_s2_files
from get_MCD43 import get_mcd43
from multiprocessing import Pool
from scipy import ndimage, signal
from scipy.stats import linregress
from shapely.geometry import Point
from collections import namedtuple
from sklearn.externals import joblib 
from scipy import optimize, interpolate
from get_jasmin_s2 import find_on_jasmin
from s2_preprocessing import s2_pre_processing
from l8_preprocessing import l8_pre_processing
from scipy.interpolate import NearestNDInterpolator
from skimage.morphology import disk, binary_dilation, binary_closing, binary_opening
procs =  multiprocessing.cpu_count()
file_path = os.path.dirname(os.path.realpath(__file__))
from create_logger import create_logger

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def do_one_site(site):
    s2_fnames = glob(acix_2 + '/S2s_30/%s/*'%site)
    l8_fnames = glob(acix_2 + '/L8s_30_LZW/%s/*'%site)
    #aoi = glob(acix_2 + '/%s_location.json'%site)[0]
    s2_times = [datetime.datetime.strptime(i.split('_MSIL1C_')[1][:8], '%Y%m%d') for i in s2_fnames]
    l8_times = [datetime.datetime.strptime(i.split('LC08_')[1][12:20], '%Y%m%d') for i in l8_fnames]
    s2_files = sorted(s2_fnames, key = lambda s2_fname: s2_times[s2_fnames.index(s2_fname)])
    l8_files = sorted(l8_fnames, key = lambda l8_fname: l8_times[l8_fnames.index(l8_fname)])
    s2_times = sorted(s2_times)
    l8_times = sorted(l8_times)
    
    return s2_files, l8_files, s2_times, l8_times

def do_one_site(site):
    s2_fnames = glob(acix_2 + '/S2s*30/%s/*'%site)
    l8_fnames = glob(acix_2 + '/L8s_30_LZW*/%s/*'%site)
    #aoi = glob(acix_2 + '/%s_location.json'%site)[0]
    s2_times = [datetime.datetime.strptime(i.split('_MSIL1C_')[1][:8], '%Y%m%d') for i in s2_fnames]
    l8_times = [datetime.datetime.strptime(i.split('LC08_')[1][12:20], '%Y%m%d') for i in l8_fnames]
    s2_files = sorted(s2_fnames, key = lambda s2_fname: s2_times[s2_fnames.index(s2_fname)])
    l8_files = sorted(l8_fnames, key = lambda l8_fname: l8_times[l8_fnames.index(l8_fname)])
    s2_times = sorted(s2_times)
    l8_times = sorted(l8_times)
    
    return s2_files, l8_files, s2_times, l8_times


def get_location():

    aeronet_locations_url = "https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt"
    r = requests.get(aeronet_locations_url)
    txt = r.content.decode("utf-8")
    aero_sites = pd.read_csv(StringIO(txt), skiprows=1)
    sel_sites = aero_sites[aero_sites.Site_Name.isin(sites)][aero_sites.columns[0]].values
    lons = aero_sites[aero_sites.Site_Name.isin(sites)][aero_sites.columns[1]].values
    lats = aero_sites[aero_sites.Site_Name.isin(sites)][aero_sites.columns[2]].values
    m = mgrs.MGRS()
    for ii, site in enumerate(sel_sites):
        lat = lats[ii]
        lon = lons[ii]
        tile = m.toMGRS(lat, lon, MGRSPrecision=0).decode()
        fs = glob(acix_2 + '/S2s_30/' + site + '/*.SAFE/GR*/*/IMG_DATA/*B02.jp2')
        if len(fs) <= 0:
            print(site)
            continue
        g = gdal.Open(fs[0])
        proj = g.GetProjectionRef()
        inputEPSG = 4326
        inSpatialRef = osr.SpatialReference()
        inSpatialRef.ImportFromEPSG(inputEPSG)
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromWkt(proj)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(lon, lat)
        coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
        point.Transform(coordTransform)
        test = Point(point.GetX(), point.GetY())
        buf = test.buffer(15000, cap_style=3)
        buf = ogr.CreateGeometryFromWkt(buf.to_wkt())
        coordTransform = osr.CoordinateTransformation(outSpatialRef, inSpatialRef)
        buf.Transform(coordTransform)
        with open(site + '_location.json', 'w') as f:
            f.write(buf.ExportToJson())

def pre_processing(s2_fnames, l8_fnames):
    s2_files = []
    l8_files = []
    for s2_fname in s2_fnames:
        ret = s2_pre_processing(s2_fname, temp_angle = True, s2_angle_dir = './')    
        s2_files.append(ret[0])
    for l8_fname in l8_fnames:
        ret = l8_pre_processing(l8_fname, temp_angle = True, l8_angle_dir = './')    
        l8_files.append(ret[0])
    return s2_files, l8_files

# def _find_around_files(site):                                      
#     f = np.load('S2_files_to_be_down.npz')
#     fs = [i[0] for i in f.f.rets]
#     s2_files, l8_files, s2_times, l8_times = do_one_site(site)
#     res = []           
#     tiles = np.unique([i.split('_')[-2] for i in s2_files]).tolist()
#     for i in s2_times + l8_times:
#         all_time  = [i+ datetime.timedelta(days=ii) for ii in range(-20, 20)]
#         need_files = [j for j in fs if (datetime.datetime.strptime(j[11:19], '%Y%m%d') in all_time) & (j[38:44] in tiles)]
#         res += need_files   
#     return np.unique(res).tolist()


def find_around_files(site):
    s2_files, l8_files, s2_times, l8_times = do_one_site(site)
    aoi = glob(acix_2 + '/%s_location.json'%site)[0] 
    res = [] 
    for i in s2_times: 
        s2_fnames = [] 
        l8_fnames = []
        for j in s2_times: 
            if abs((j - i).days)<15: 
                s2_fnames.append(s2_files[s2_times.index(j)]) 
        for k in l8_times: 
            if abs((k - i).days)<15: 
                l8_fnames.append(l8_files[l8_times.index(k)]) 
        res.append([aoi, s2_files[s2_times.index(i)], s2_fnames, l8_fnames])
    return res

def find_around_files(site):
    s2_files, l8_files, s2_times, l8_times = do_one_site(site)
    aoi = glob(acix_2 + '/%s_location.json'%site)[0] 
    res = [] 
    # for i in s2_times: 
    for i, s2_file in enumerate(s2_files): 
        s2_fnames = []
        l8_fnames = []
        s2_diff = [] 
        l8_diff = [] 
        for _, j in enumerate(s2_times): 
            if (abs((j - s2_times[i]).days)<15) & (abs((j - s2_times[i]).days) not in s2_diff) & (s2_files[_] != s2_file): 
                s2_fnames.append(s2_files[_]) 
                s2_diff.append(abs((j - s2_times[i]).days))
        for _, k in enumerate(l8_times): 
            if (abs((k - s2_times[i]).days)<15) & (abs((k - s2_times[i]).days) not in l8_diff):
                l8_fnames.append(l8_files[_]) 
                l8_diff.append(abs((k - s2_times[i]).days))
        middle_file = s2_file
        diffs  = s2_diff + l8_diff
        fnames = s2_fnames + l8_fnames
        fnames = sorted(fnames, key = lambda kk: diffs[fnames.index(kk)])[:3]
        s2_fnames = [] 
        l8_fnames = []
        for fname in fnames:
            if '_MSIL1C_' in fname:
                s2_fnames.append(fname)
            else:
                l8_fnames.append(fname)
        res.append([aoi, middle_file, 0, [s2_file,] + s2_fnames, l8_fnames])

    for i, l8_file in enumerate(l8_files): 
        s2_fnames = [] 
        l8_fnames = []
        s2_diff = [] 
        l8_diff = [] 
        for _, j in enumerate(s2_times): 
            if (abs((j - l8_times[i]).days)<15) & (abs((j - l8_times[i]).days) not in s2_diff): 
                s2_fnames.append(s2_files[_]) 
                s2_diff.append(abs((j - l8_times[i]).days))
        for _, k in enumerate(l8_times): 
            if (abs((k - l8_times[i]).days)<15) & (abs((k - l8_times[i]).days) not in l8_diff) & (l8_files[_] != l8_file): 
                l8_fnames.append(l8_files[_])
                l8_diff.append(abs((k - l8_times[i]).days)) 
        middle_file = l8_file
        diffs  = s2_diff + l8_diff
        fnames = s2_fnames + l8_fnames
        fnames = sorted(fnames, key = lambda i: diffs[fnames.index(i)])[:3]
        s2_fnames = [] 
        l8_fnames = []
        for fname in fnames:
            if '_MSIL1C_' in fname:
                s2_fnames.append(fname)
            else:
                l8_fnames.append(fname)
        res.append([aoi, middle_file, 0, s2_fnames, [l8_file,] + l8_fnames])
    return res

def get_kk(angles):                                                                                                                                                                            
    vza ,sza,raa = angles
    kk = kernels.Kernels(vza ,sza,raa,\
                         RossHS=False,MODISSPARSE=True,\
                         RecipFlag=True,normalise=1,\
                         doIntegrals=False,LiType='Sparse',RossType='Thick')
    return kk



def warp_data(fname, outputBounds,  xRes, yRes,  dstSRS):
    modis_sinu = osr.SpatialReference() 
    sinu = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    modis_sinu.ImportFromProj4 (sinu)
    g = gdal.Warp('',fname, format = 'MEM',  dstSRS = dstSRS, resampleAlg = 0, outputType = gdal.GDT_Float32,
                  outputBounds=outputBounds, xRes = xRes, yRes = yRes, srcSRS=modis_sinu, )  # warpOptions = ['NUM_THREADS=ALL_CPUS']
    return g.ReadAsArray() 

def read_MCD43(fnames, dstSRS, outputBounds):
    par = partial(warp_data, outputBounds = outputBounds, xRes = 500, yRes = 500, dstSRS = dstSRS) 
    p = Pool(procs)
    ret = p.map(par,  fnames)
    p.close()
    p.join()
    n_files = int(len(fnames)/2)
    #ret = parmap(par, fnames) 
    das = np.array(ret[:n_files])     
    qas = np.array(ret[n_files:]) 
    #ws  = 0.618034**qas       
    #ws[qas==255]                     = 0 
    #ws[np.any((das==32767), axis=1)] = 0
    #das[das==32767]                  = -3276
    return das, qas
     
def get_boa(mcd43_dir, boa_bands, obs_time, outputBounds, dstSRS, temporal_filling = 16):
     
    qa_temp = 'MCD43_%s_BRDF_Albedo_Band_Mandatory_Quality_Band%d.vrt'
    da_temp = 'MCD43_%s_BRDF_Albedo_Parameters_Band%d.vrt'
    doy = obs_time.strftime('%Y%j')
    #print(temporal_filling)
    #if temporal_filling == True:
    #    temporal_filling = 16
    #print(temporal_filling)
    if temporal_filling:
        days   = [(obs_time - datetime.timedelta(days = int(i))) for i in np.arange(temporal_filling, 0, -1)] + \
                 [(obs_time + datetime.timedelta(days = int(i))) for i in np.arange(0, temporal_filling+1,  1)]
        #print(days)
        fnames = []
        for temp in [da_temp, qa_temp]:                                                                                    
            for day in days:
                for band in boa_bands:
                    fname = mcd43_dir + '/'.join([day.strftime('%Y-%m-%d'), temp%(day.strftime('%Y%j'), band)])
                    fnames.append(fname) 
    else:
        fnames = []
        for temp in [da_temp, qa_temp]:
            for band in boa_bands:
                    fname = mcd43_dir + '/'.join([datetime.datetime.strftime(obs_time, '%Y-%m-%d'), temp%(doy, band)])
                    fnames.append(fname)
    das, ws = read_MCD43(fnames, dstSRS, outputBounds) 
    return das, ws


# def smooth(da_w):
#     da, w = da_w
#     if (da.shape[-1]==0) | (w.shape[-1]==0):
#         return da, w
#     data  = np.array(smoothn(da, s=10., smoothOrder=1., axis=0, TolZ=0.001, verbose=False, isrobust=True, W = w))[[0, 3],]                                                                                                                          
#     return data[0], data[1]

def read_l8(l8s, pix_res, dstSRS, outputBounds, vrt_dir, temporal_window ):
    l8_toas = []                             
    l8_surs = []                             
    l8_uncs = []                             
    l8_clds = []                             
    for i in l8s:                            
        toas = []                            
        cloud = gdal.Warp('', i.cloud, format = 'MEM', outputBounds = outputBounds, dstSRS = dstSRS, 
                            xRes = pix_res, yRes = pix_res, resampleAlg = gdal.GRIORA_NearestNeighbour).ReadAsArray()             
        l8_clds.append(cloud)                
        cloud_mask = (~((cloud  >= 2720) & ( cloud <= 2732))) | (cloud <= 0)
        with open(i.meta) as f:              
            for line in f:                   
                if 'REFLECTANCE_MULT_BAND' in line:
                    scale = float(line.split()[-1])
                elif 'REFLECTANCE_ADD_BAND' in line:
                    off = float(line.split()[-1])
        sza    = gdal.Warp('', i.sun_angs, format = 'MEM', outputBounds = outputBounds, dstSRS = dstSRS,
                            xRes = pix_res, yRes = pix_res, resampleAlg = gdal.GRIORA_Average).ReadAsArray()[1] * 0.01
        scale  = scale / np.cos(np.deg2rad(sza))
        off    = off / np.cos(np.deg2rad(sza))
        for toa_band in i.toa:               
            data = gdal.Warp('', toa_band, format = 'MEM', outputBounds = outputBounds, dstSRS = dstSRS,
                                xRes = pix_res, yRes = pix_res, resampleAlg = gdal.GRIORA_Average).ReadAsArray()
            data = data * scale + off        
            #data[cloud_mask] = np.nan       
            toas.append(np.ma.array(data, mask = cloud_mask))
        l8_toas.append(np.ma.array(toas))    
        boa_bands = [3, 4, 1, 2, 5, 6]       
        das, qas = get_boa(vrt_dir, boa_bands, i.obs_time, outputBounds, dstSRS, temporal_filling = temporal_window )
        saa, sza = gdal.Warp('', i.sun_angs, format = 'MEM', outputBounds = outputBounds,
                             xRes = 500, yRes = 500, dstSRS = dstSRS, dstNodata=-32767, outputType = gdal.GDT_Int16).ReadAsArray() / 100
        bad_ang = (saa<-180.) | (sza < 0.)
        if (~bad_ang).sum()>3:
            new_shp  = saa.shape                 
            new_xys  = np.where(np.ones_like(sza))
            f_interp = NearestNDInterpolator(np.where(sza>=0), sza[sza>=0])
            sza = f_interp(new_xys).reshape(new_shp)
            f_interp = NearestNDInterpolator(np.where(saa>-180), saa[saa>-180]) 
            saa = f_interp(new_xys).reshape(new_shp)
            bad_ang = (saa<-180.) | (sza < 0.)
        else:
            bad_ang = (saa<-180.) | (sza < 0.)
        l8_sur = []                          
        l8_unc = []                          
        for _, band in enumerate([1, 2, 3, 4, 5, 6]):
            vaa, vza = gdal.Warp('', i.view_angs[band], format = 'MEM', outputBounds = outputBounds, 
                                 xRes = 500, yRes = 500, dstSRS = dstSRS, dstNodata=-32767).ReadAsArray() / 100.
            raa    = vaa  - saa              
            kk     = get_kk((vza ,sza, raa)) 
            k_vol  = kk.Ross                  
            k_geo  = kk.Li                   
            kers   = np.array([np.ones(k_vol.shape), k_vol, k_geo])
            fs     = das[_::6]               
            qa     = qas[_::6]               
            bad_fs = np.any(fs==32767, axis=1) | (qa >=2 )
            qa     = qa * 1.                 
            sur    = np.sum(fs *  kers * 0.001, axis=1)
            mask   = (vaa<-180.) | (vza<-90.) | bad_ang 
            sur[:,mask] = np.nan             
            sur[sur<=0] = np.nan             
            sur[bad_fs] = np.nan             
            qa[np.isnan(sur)] = np.nan       
            unc = 0.015*0.618034**(-np.nanmedian(qa, axis=0))
            l8_sur.append(np.nanmedian(sur, axis=0))
            l8_unc.append(unc)               
        l8_surs.append(l8_sur)               
        l8_uncs.append(l8_unc)  
    return l8_toas, l8_surs, l8_uncs, l8_clds

def read_s2(s2s, pix_res, dstSRS, outputBounds, vrt_dir, temporal_window):
    s2_toas = []
    s2_surs = []                                                                                                                                                          
    s2_uncs = []                                
    s2_clds = []                                
    for i in s2s:                               
        toas = []                               
        g = gdal.Warp('', i.cloud, format = 'MEM', outputBounds = outputBounds, 
                      xRes = pix_res, yRes = pix_res, resampleAlg = gdal.GRIORA_Average, dstSRS = dstSRS)
        cloud = g.ReadAsArray()                 
        s2_clds.append(cloud)                   
        cloud_mask = (cloud > 40) | (cloud < 0) 
        for toa_band in i.toa:                  
            data = gdal.Warp('', toa_band, format = 'MEM', outputBounds = outputBounds, 
                             dstSRS = dstSRS, xRes = pix_res, yRes = pix_res, resampleAlg = gdal.GRIORA_Average).ReadAsArray()/10000.
            #data[cloud_mask] = np.nan          
            toas.append(np.ma.array(data, mask = cloud_mask))
        s2_toas.append(np.ma.array(toas))       
        boa_bands = [3, 4, 1, 2, 5, 6]          
        das, qas = get_boa(vrt_dir, boa_bands, i.obs_time, outputBounds, dstSRS, temporal_filling = temporal_window )
        saa, sza = gdal.Warp('', i.sun_angs, format = 'MEM', outputBounds = outputBounds, 
                             xRes = 500, yRes = 500, dstSRS = dstSRS, dstNodata=-32767, outputType = gdal.GDT_Int16).ReadAsArray() / 100
        bad_ang = (saa<-180.) | (sza < 0.)
        if (~bad_ang).sum()>3:
            new_shp  = saa.shape                    
            new_xys  = np.where(np.ones_like(sza))  
            f_interp = NearestNDInterpolator(np.where(sza>=0), sza[sza>=0]) 
            sza = f_interp(new_xys).reshape(new_shp)
            f_interp = NearestNDInterpolator(np.where(saa>-180), saa[saa>-180]) 
            saa = f_interp(new_xys).reshape(new_shp)
            bad_ang = (saa<-180.) | (sza < 0.)      
        else:
            bad_ang = (saa<-180.) | (sza < 0.)
        s2_sur = []                             
        s2_unc = []                             
        for _, band in enumerate([1, 2, 3, 8, 11, 12]):
            vaa, vza = gdal.Warp('', i.view_angs[band], format = 'MEM', outputBounds = outputBounds, 
                                 xRes = 500, yRes = 500, dstSRS = dstSRS, dstNodata=-32767, outputType = gdal.GDT_Int16).ReadAsArray() / 100.
            raa    = vaa  - saa                 
            kk     = get_kk((vza ,sza, raa))    
            k_vol  = kk.Ross                                                                                                                           
            k_geo  = kk.Li                      
            kers   = np.array([np.ones(k_vol.shape), k_vol, k_geo])
            fs     = das[_::6]                  
            qa     = qas[_::6]                  
            bad_fs = np.any(fs==32767, axis=1) | (qa >=2 )
            qa     = qa * 1.                    
            sur    = np.sum(fs *  kers * 0.001, axis=1)
            mask   = (vaa<-180.) | (vza<-90.) | bad_ang 
            sur[:,mask] = np.nan                
            sur[sur<=0] = np.nan                
            sur[bad_fs] = np.nan                
            qa[np.isnan(sur)] = np.nan          
            unc = 0.015*0.618034**(-np.nanmedian(qa, axis=0))
            s2_sur.append(np.nanmedian(sur, axis=0))
            s2_unc.append(unc)                  
        s2_surs.append(s2_sur)                  
        s2_uncs.append(s2_unc) 
    return s2_toas, s2_surs, s2_uncs, s2_clds


def redo_cloud_shadow(s2_toas, l8_toas, s2_clds, l8_clds):
    if len(s2_toas + l8_toas) == 0:
        raise IOError('At least one S2 or L8 image is needed.')
    else:
        shape = (s2_toas + l8_toas)[0][0].shape
    if len(s2_toas)>0:
        s2_temp = np.ma.array(s2_toas)[:,[1, 2, 3, 8, 11, 12]]
    else:
        s2_temp = np.zeros((0, 6) + shape)
    if len(l8_toas)>0: 
        l8_temp = np.ma.array(l8_toas)[:,[1, 2, 3, 4, 5, 6]]
    else:
        l8_temp = np.zeros((0, 6) + shape)

    month_obs = np.ma.vstack([s2_temp, l8_temp])
    med_obs = np.ma.median(month_obs, axis = 0)

    struct = disk(5)                            
    s2_shadows = []                             
    s2_clouds  = []                              
    s2_changes = []
    for _, s2_toa in enumerate(s2_toas):        
        diff = s2_toa.data[[1, 2, 3, 8, 11, 12]] - med_obs  
        vis_diff = diff[:3].mean(axis=0)        
        certain_cloud = (s2_clds[_] > 30) & (vis_diff > 0.075)
        certain_cloud = binary_dilation(certain_cloud, selem = struct).astype(certain_cloud.dtype) & (vis_diff>0.025) & (s2_clds[_] > 20)
        certain_cloud = binary_opening(certain_cloud)
        certain_cloud[vis_diff.mask] = (s2_clds[_]>40)[vis_diff.mask]
        nir_diff = diff[3:].mean(axis=0)        
        shadow   = nir_diff < -0.1              
        shadow   = binary_opening(shadow) & (~certain_cloud)
        s2_clouds.append(certain_cloud)         
        s2_shadows.append(shadow)               
        s2_changes.append(diff)

    l8_shadows = []                             
    l8_clouds  = []     
    l8_changes = []                         
    for _, l8_toa in enumerate(l8_toas):        
        diff = l8_toa.data[[1, 2, 3, 4, 5, 6]] - med_obs  
     
        vis_diff       = diff[:3].mean(axis=0)  
        certain_cloud  = (((l8_clds[_] >> 5) & 3) > 1) & (vis_diff > 0.075)
        certain_cloud  = binary_dilation(certain_cloud, selem = struct).astype(certain_cloud.dtype) & (vis_diff > 0.025) 
        certain_cloud  = binary_opening(certain_cloud)
        certain_cloud[vis_diff.mask] = (((l8_clds[_] >> 5) & 3) > 0)[vis_diff.mask] 
     
        nir_diff       = diff[3:].mean(axis=0)  
        certain_shadow = (((l8_clds[_] >> 7) & 3) > 1) & (nir_diff < -0.1)
        certain_shadow = binary_dilation(certain_shadow, selem = struct).astype(certain_cloud.dtype) & (nir_diff < -0.025)
        certain_shadow = binary_opening(certain_shadow) & (~certain_cloud)
        l8_clouds.append(certain_cloud)                                                                                                                                   
        l8_shadows.append(certain_shadow) 
        l8_changes.append(diff)
    return s2_clouds, s2_shadows,s2_changes, l8_clouds, l8_shadows, l8_changes

def gaussian(xstd, ystd, norm = True):
    winx = 2*int(np.ceil(1.96*xstd))
    winy = 2*int(np.ceil(1.96*ystd))
    xgaus = signal.gaussian(winx, xstd)
    ygaus = signal.gaussian(winy, ystd)
    gaus  = np.outer(xgaus, ygaus)
    if norm:
        return gaus/gaus.sum()
    else:
        return gaus 


#@jit()
def cost(p, toa, sur, pxs, pys, pcxs, pcys, gaus):
    xshift, yshift = p
    point_xs     = pxs + xshift
    point_ys     = pys + yshift
    mask         = (point_xs<toa.shape[0]) & (point_xs>0) & (point_ys<toa.shape[1]) & (point_ys>0)
    point_xs     = point_xs#[mask]
    point_ys     = point_ys#[mask]
    points       = np.array([point_xs, point_ys]).T#np.array([np.repeat(point_xs, len(point_ys)), np.tile(point_ys, len(point_xs))]).T
    mask         = (points[:,0]<toa.shape[0]) & (points[:,0]>0) & (points[:,1]<toa.shape[1]) & (points[:,1]>0) 
    conv_toa     = points_convolve(toa, gaus, points)
    mask         = mask & (conv_toa>=0.001) & ((conv_toa - sur[pcxs, pcys])<0.1)
    toa = conv_toa[mask]
    sur = sur[pcxs, pcys][mask]
    ret = np.corrcoef(sur, toa)
    cost = 1-ret[0,1]
    return cost, points, mask

@jit(nopython=True)
def convolve(data, kernel, points): 
    kx   = int(np.ceil(kernel.shape[0]/2.))
    ky   = int(np.ceil(kernel.shape[1]/2.))
    rets = np.zeros(len(points)) 
    # padx = int(np.ceil(kernel.shape[0]/2.))
    # pady = int(np.ceil(kernel.shape[1]/2.)) 
    for _ in range(len(points)): 
        x, y    = points[_]
        batch   = data[x: x + 2*kx, y: y + 2*ky][:kernel.shape[0],:kernel.shape[1]]
        if batch.size == 0:
            rets[_] = np.nan
        else:
            counts  = np.sum(np.isfinite(batch)*kernel)
            if counts == 0:
                rets[_] = np.nan
            else:
                rets[_] = np.nansum(batch*kernel) / counts
    return rets

@jit(nopython=True)
def points_convolve(im, kernel, points): 
    rows, cols     = im.shape
    k_rows, k_cols = kernel.shape
    # padx = int(k_rows/2.)
    # pady = int(k_cols/2.)
    data = np.zeros((rows + 2*k_rows, cols + 2*k_cols))
    #data = np.pad(im, (2*padx, 2*pady), mode='reflect') 
    data[:rows, :cols] = im
    return convolve(data, kernel, points) 

def do_s2_psf(s2s, s2_toas, s2_clouds, s2_shadows, s2_surs, possible_x_y, struct, gaus, pointXs, pointYs):
    s2_conv_toas = []                    
    s2_cors_surs = []
    s2_cors_pots = []
    logger = create_logger()
    logger.propagate = False
    for _, s2_toa in enumerate(s2_toas):      
        sur        = s2_surs[_][-1]            
        c_x, c_y   = sur.shape   
        data       = s2_toa[12].data.copy()   
        border     = np.ones_like(data).astype(bool)
        border[1:-1, 1:-1] = False       
        bad_pix    = s2_clouds[_] | s2_shadows[_] | border
        bad_pix    = ndimage.binary_dilation(bad_pix, structure = struct, iterations=1)
        good_xy    = (~bad_pix)[pointXs, pointYs]
        p_xs, p_ys = pointXs[good_xy], pointYs[good_xy]
        p_corse_xs, p_corse_ys = np.repeat(range(c_x), c_y)[good_xy], np.tile(range(c_y), c_x)[good_xy]

        if np.isnan(sur).sum() / sur.size > 0.95:
            logger.info('More than 95% surface reflectance is invalid and psf optimisation is passed.')
            points = np.array([p_xs, p_ys]).T
            mask   = np.zeros_like(p_xs).astype(bool)
        else:
            par = partial(cost, toa = data, sur = s2_surs[_][-1], pxs = p_xs, pys = p_ys, pcxs = p_corse_xs, pcys = p_corse_ys, gaus = gaus)
            # p   = Pool()                     
            ret = list(map(par, possible_x_y))        
            # p.close()                        
            # p.join()                         
            costs = np.array(ret, dtype=np.object)[:,0].astype(float)             
            if np.isfinite(costs).any():
                mind = np.nanargmin(costs)
                shift_x, shift_y   = possible_x_y[mind]
                un_r, points, mask = ret[mind]   
                logger.info('X_shift: %d, Y_shift: %d, rValue: %.02f'%(shift_x, shift_y, 1-un_r))
            else:
                logger.warning('PSF estimation failed and default values are used.')
                un_r, points, mask = ret[114] 
                logger.info('X_shift: %d, Y_shift: %d'%(-11, -16))

        s2_conv_toa = []          
        s2_cors_sur = []       
        for band in [1, 2, 3, 8, 11, 12]:                                                                                                                                 
            data = s2_toa[band].data.copy()   
            data[s2_clouds[_]] = np.nan       
            data[s2_shadows[_]] = np.nan      
            conv_toa = points_convolve(data, gaus, points)
            conv_toa[conv_toa<=0.001] = np.nan
            s2_conv_toa.append(conv_toa[mask])      
        s2_cors_sur = np.array(s2_surs[_])[:, p_corse_xs[mask], p_corse_ys[mask]]
        
        s2_conv_toas.append(s2_conv_toa) 
        s2_cors_surs.append(s2_cors_sur)
        s2_cors_pots.append(np.array([p_corse_xs[mask], p_corse_ys[mask]]))

    s2_obss = [] 
    for _, s2_toa in enumerate(s2s):
        s2_obs = namedtuple('s2_obs', 'cors_sur conv_toa cors_pts')
        sensor = s2_toa[2][0].split('_MSIL1C_')[0][-3:]
        mask, s2_cors_sur = spectral_mapping(s2_cors_surs[_], s2_conv_toas[_], sensor)
        #s2_cors_surs[_]   = np.array(s2_cors_sur)[:, mask]
        #s2_cors_pots[_]   = np.array(s2_cors_pots[_])[:, mask]
        #s2_conv_toas[_]   = np.array(s2_conv_toas[_])[:, mask]
        ret = s2_obs(np.array(s2_cors_sur)[:, mask], np.array(s2_conv_toas[_])[:, mask], np.array(s2_cors_pots[_])[:, mask])
        s2_obss.append(ret)
    return s2_obss

def do_l8_psf(l8s, l8_toas, l8_clouds, l8_shadows, l8_surs, possible_x_y, struct, gaus, pointXs, pointYs):    
    l8_conv_toas = []                    
    l8_cors_surs = []
    l8_cors_pots = []
    logger = create_logger()
    logger.propagate = False
    for _, l8_toa in enumerate(l8_toas):      
        sur        = l8_surs[_][0]
        c_x, c_y   = sur.shape
        data       = l8_toa[6].data.copy()   
        border     = np.ones_like(data).astype(bool)
        border[1:-1, 1:-1] = False       
        bad_pix    = l8_clouds[_] | l8_shadows[_] | border
        bad_pix    = ndimage.binary_dilation(bad_pix, structure = struct, iterations=1)
        good_xy    = (~bad_pix)[pointXs, pointYs]
        p_xs, p_ys = pointXs[good_xy], pointYs[good_xy]
        p_corse_xs, p_corse_ys = np.repeat(range(c_x), c_y)[good_xy], np.tile(range(c_y), c_x)[good_xy]

        if np.isnan(sur).sum() / sur.size > 0.95:
            logger.info('More than 95% surface reflectance is invalid and psf optimisation is passed.')
            points = np.array([p_xs, p_ys]).T
            mask   = np.zeros_like(p_xs).astype(bool)
        else:
            par = partial(cost, toa = data, sur = l8_surs[_][-1], pxs = p_xs, pys = p_ys, pcxs = p_corse_xs, pcys = p_corse_ys, gaus = gaus)
            #p   = Pool()                     
            ret = list(map(par, possible_x_y))        
            #p.close()                        
            #p.join()           
            costs = np.array(ret, dtype=np.object)[:,0].astype(float)              
            if np.isfinite(costs).any():
                mind = np.nanargmin(costs)
                shift_x, shift_y   = possible_x_y[mind]
                un_r, points, mask = ret[mind]   
                logger.info('X_shift: %d, Y_shift: %d, rValue: %.02f'%(shift_x, shift_y, 1-un_r))
            else:
                logger.warning('PSF estimation failed and default values are used.')
                un_r, points, mask = ret[114] 
                logger.info('X_shift: %d, Y_shift: %d'%(-11, -16))
        l8_conv_toa = []          
        for band in [1, 2, 3, 4, 5, 6]:                                                                                                                                 
            data = l8_toa[band].data.copy()   
            data[l8_clouds[_]] = np.nan       
            data[l8_shadows[_]] = np.nan      
            conv_toa = points_convolve(data, gaus, points)
            conv_toa[conv_toa<=0.001] = np.nan
            l8_conv_toa.append(conv_toa[mask])      
        l8_cors_sur = np.array(l8_surs[_])[:, p_corse_xs[mask], p_corse_ys[mask]]
    
        l8_conv_toas.append(l8_conv_toa) 
        l8_cors_surs.append(l8_cors_sur)
        l8_cors_pots.append(np.array([p_corse_xs[mask], p_corse_ys[mask]]))
    l8_obss = []
    for _, l8_toa in enumerate(l8s):
        l8_obs = namedtuple('l8_obs', 'cors_sur conv_toa cors_pts')
        sensor = 'L8'
        mask, l8_cors_sur = spectral_mapping(l8_cors_surs[_], l8_conv_toas[_], sensor)
        #l8_cors_surs[_]   = np.array(l8_cors_sur)[:, mask]
        #l8_cors_pots[_]   = np.array(l8_cors_pots[_])[:, mask]
        #l8_conv_toas[_]   = np.array(l8_conv_toas[_])[:, mask]
        ret = l8_obs(np.array(l8_cors_sur)[:, mask], np.array(l8_conv_toas[_])[:, mask], np.array(l8_cors_pots[_])[:, mask])
        l8_obss.append(ret)
    return l8_obss

def cal_psf_points(pix_res, sur_x, sur_y):
    xstd, ystd      = 260/pix_res, 340/pix_res
    pointXs         = ((np.arange(sur_x) * 500) // pix_res ) 
    pointYs         = ((np.arange(sur_y) * 500) // pix_res ) 
    pointXs,pointYs = np.repeat(pointXs, len(pointYs)), np.tile(  pointYs, len(pointXs))
    gaus         = gaussian(xstd, ystd, norm = True)
    # coming from the solving of a lot of tiles to reach smaller range
    possible_x_y = [(np.arange(-25,-5), np.arange(-25,-5) +i) for i in range(-10, 10)]
    possible_x_y = np.array(possible_x_y).transpose(1,0,2).reshape(2, -1).T
    struct       = disk(5) 
    return possible_x_y, struct, gaus, pointXs, pointYs

def spectral_mapping(sur, toa, sensor):
    # pmins = [[ 0.81793009, -1.55666629,  0.03879234,  0.02664923],
    #          [ 0.50218134, -0.94398654, -0.36284911,  0.02876391],
    #          [ 0.61609484, -1.12717424, -0.24037129,  0.0239488 ],
    #          [ 0.67499803, -1.1988073 , -0.18331019,  0.02179141],
    #          [ 0.23458873, -0.4048219 , -0.56692888,  0.02484466],
    #          [ 0.08220874, -0.13492051, -0.74972003, -0.0331204 ]]
    # pmaxs = [[-0.76916621,  1.8524333 , -1.43464388,  0.34984857],
    #          [-0.91464915,  1.96174322, -1.38302832,  0.28090987],
    #          [-0.9199249 ,  1.9681306 , -1.3704881 ,  0.28924671],
    #          [-0.87389258,  1.89261443, -1.30929285,  0.28807412],
    #          [-0.71647392,  1.34657557, -0.79536697,  0.13551599],
    #          [-0.34076349,  0.60544841, -0.34178543,  0.09669959]]
    bounds =   [[-0.32435316099489,    0.33568660536379],
                [-0.41078127804613995, 0.26727484647285],
                [-0.38370667189964,    0.2757377221351],
                [-0.3650360028266299,  0.27516957905042],
                [-0.40238228381132995, 0.12769626138308],
                [-0.50553709841854,    0.09334193977751]]
    spec_map     =  Two_NN(np_model_file=file_path+'/spectral_mapping/Aqua_%s_spectral_mapping.npz'%sensor)
    sur     = np.array(spec_map.predict(sur.T)).squeeze()
    mask = True                                 
    if sur.shape[1] > 3: 
        for i in range(len(toa)):           
            # pmin = np.poly1d(pmins[i])
            # pmax = np.poly1d(pmaxs[i])
            pmin, pmax = bounds[i]
            diff = toa[i] - sur[i]
            mas  = (diff >= pmin) & (diff <= pmax)
            # mas  = (diff >= pmin(sur[i])) & (diff <= pmax(sur[i]))
            if mas.sum() > 0:
                mmin, mmax = np.percentile(toa[i][mas] - sur[i][mas], [5, 95])
                mas  = mas & (diff >= mmin) & (diff <= mmax)
            mask = mask & mas
    else:        
        mask = np.zeros(sur.shape[1]).astype(bool)
    return mask, sur

def get_stable_targets(changes, clouds, shadows, toas):
    stable_targets = []
    for i in range(len(changes)):
        toa    = toas[i]
        change = np.mean([abs(changes[i][-1]).data + abs(changes[i][-2].data)], axis=0)
        stable_target = (change < 0.025) & (~clouds[i]) & (~shadows[i]) & (change / np.mean(toa[-2:].data, axis=0) < 0.1)
        stable_targets.append(stable_target)
    return stable_targets

def get_bounds(aoi, toa, pix_res):
    g = gdal.Warp('', toa, format = 'MEM', cutlineDSName=aoi, cropToCutline=True, xRes = pix_res, yRes = pix_res)
    dstSRS = g.GetProjectionRef()
    geo_t = g.GetGeoTransform()
    x_size, y_size = g.RasterXSize, g.RasterYSize     
    xmin, xmax = min(geo_t[0], geo_t[0] + x_size * geo_t[1]), max(geo_t[0], geo_t[0] + x_size * geo_t[1])  
    ymin, ymax = min(geo_t[3], geo_t[3] + y_size * geo_t[5]), max(geo_t[3], geo_t[3] + y_size * geo_t[5])
    outputBounds = [xmin, ymin, xmax, ymax]
    return dstSRS, outputBounds

def read_ele(dem, pix_res, dstSRS, outputBounds):
    g = gdal.Warp('', dem, format = 'MEM', outputBounds = outputBounds,
                             dstSRS = dstSRS, xRes = pix_res, yRes = pix_res, resampleAlg = gdal.GRA_Bilinear)
    data = g.ReadAsArray() / 1000.
    return data

def read_atmos_prior(cams_dir, pix_res, obs_time, dstSRS, outputBounds):
    time_ind    = np.abs((obs_time.hour  + obs_time.minute/60. + obs_time.second/3600.) - np.arange(0,25,3)).argmin()
    cams_names  = ['aod550', 'tcwv', 'gtco3'] 
    prior_uncs = [0.4, 0.5, 0.05]
    priors = []
    prior_scales = [1., 0.1, 46.698]
    for i in range(3):
        prior_f = cams_dir + '/'.join([datetime.datetime.strftime(obs_time, '%Y_%m_%d'), 
                                       datetime.datetime.strftime(obs_time, '%Y_%m_%d')+'_%s.tif'%cams_names[i]])
        var_g   = gdal.Open(prior_f)
        prior_g = gdal.Warp('', prior_f, format = 'MEM', outputBounds = outputBounds,
                             dstSRS = dstSRS, xRes = pix_res, yRes = pix_res, resampleAlg = gdal.GRA_Bilinear)
        g       = var_g.GetRasterBand(int(time_ind+1))
        offset  = g.GetOffset()            
        scale   = g.GetScale()             
        data    = prior_g.GetRasterBand(int(time_ind+1)).ReadAsArray() * scale + offset
        data[:] = np.nanmean(data)
        priors.append(data * prior_scales[i])
        #prior_uncs.append(np.ones_like(data)*prior_uncs[i])
    return priors, prior_uncs

def read_ang(ang, pix_res, dstSRS, outputBounds):
    g = gdal.Warp('', ang, format = 'MEM', outputBounds = outputBounds,
                             dstSRS = dstSRS, xRes = pix_res, yRes = pix_res, resampleAlg = 0)
    angs = g.ReadAsArray() / 100.  
    
    return angs

def prepare_aux(s2s, l8s, aero_res, dstSRS, outputBounds, dem, cams_dir, s2_toas, pix_res, s2_bands, l8_bands):
    s2_aux = namedtuple('s2_aux', 'sza vza raa priors prior_uncs ele')
    s2_auxs = []
    ele = read_ele(dem, aero_res, dstSRS, outputBounds)
    for s2 in s2s:
        saa, sza = read_ang(s2.sun_angs, aero_res, dstSRS, outputBounds)
        priors, prior_uncs = read_atmos_prior(cams_dir, aero_res, s2.obs_time, dstSRS, outputBounds) 
        vzas = []
        raas = []
        for band in s2_bands: #[0, 1, 2, 3, 8, 11, 12]:
            vaa, vza = read_ang(s2.view_angs[band], aero_res, dstSRS, outputBounds)
            raa      = vaa - saa
            vzas.append(np.cos(np.deg2rad(vza)))          
            raas.append(np.cos(np.deg2rad(raa)))                             
        s2_auxs.append(s2_aux(np.cos(np.deg2rad(sza)), vzas, raas, priors, prior_uncs, ele))

    if len(s2_auxs)>0:
        tcwvs = get_starting_tcwvs(s2_auxs, s2_toas, aero_res, pix_res)
        aots  = get_s2_aots(s2_auxs, s2_toas, aero_res, pix_res)                    
    else:
        tcwvs = []   
        aots  = []
    for _, s2_aux in enumerate(s2_auxs): 
                            
        tcwv = tcwvs[_] 
        aot  = aots[_]    
        median = np.nanmedian(tcwv)
        mask = (tcwv > 0) & (tcwv < 8)
        tcwv[~mask] = median
        prior_uncs    = s2_aux.prior_uncs
        priors        = s2_aux.priors
        
        if (median > 0) & (median<8):
            priors[1]     = tcwv    
            prior_uncs[1] = 0.1     

        median = np.nanmedian(aot)
        if (median > 0) & (median<2.5):
            priors[0]     = np.ones_like(tcwv)*median
            prior_uncs[0] = 0.05

        s2_aux        = s2_aux._replace(priors = priors, prior_uncs = prior_uncs )
        s2_auxs[_]    = s2_aux 
        
    l8_aux = namedtuple('l8_aux', 'sza vza raa priors prior_uncs ele')                     
    l8_auxs = [] 
    for l8 in l8s:                                                    
        saa, sza = read_ang(l8.sun_angs, aero_res, dstSRS, outputBounds)
        priors, prior_uncs = read_atmos_prior(cams_dir, aero_res, l8.obs_time, dstSRS, outputBounds) 
        vzas = []                                                     
        raas = []
        for band in l8_bands: #[0, 1, 2, 3, 4, 5, 6]:                          
            vaa, vza = read_ang(l8.view_angs[band], aero_res, dstSRS, outputBounds)
            raa      = vaa - saa
            vzas.append(np.cos(np.deg2rad(vza)))           
            raas.append(np.cos(np.deg2rad(raa)))                               
        l8_auxs.append(l8_aux(np.cos(np.deg2rad(sza)), vzas, raas, priors, prior_uncs, ele))
    return s2_auxs, l8_auxs

def read_xa_xb_xc(sensor, satellite, toa_bands):
    xps = namedtuple('xps', 'xap xbp xcp')
    xaps = []
    xbps = []
    xcps = []
    for band in toa_bands:
        band_name = 'B' + band.upper().split('/')[-1].split('B')[-1].split('.')[0]
        xap_emu = glob(file_path + '/emus/isotropic_%s_%s_%s_xap.npz'%(sensor, satellite, band_name))[0]
        xbp_emu = glob(file_path + '/emus/isotropic_%s_%s_%s_xbp.npz'%(sensor, satellite, band_name))[0]
        xcp_emu = glob(file_path + '/emus/isotropic_%s_%s_%s_xcp.npz'%(sensor, satellite, band_name))[0]
        xap = Two_NN(np_model_file=xap_emu)
        xbp = Two_NN(np_model_file=xbp_emu)
        xcp = Two_NN(np_model_file=xcp_emu)
        xaps.append(xap)
        xbps.append(xbp)
        xcps.append(xcp)
    return xps(xaps, xbps, xcps)


def load_emus(s2s, l8s, s2_bands, l8_bands):

    s2_toa_bands = np.array(['B01', 'B02', 'B03','B04','B05' ,'B06', 'B07', 'B08','B8A', 'B09', 'B10', 'B11', 'B12'])[s2_bands] #['B01', 'B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
    s2_emus      = []    
    for _, s2_toa in enumerate(s2s):
        satellite = s2_toa[2][0].split('_MSIL1C_')[0][-3:]
        s2_emu = read_xa_xb_xc('MSI', satellite, s2_toa_bands)
        s2_emus.append(s2_emu)

    l8_toa_bands = np.array(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'])[l8_bands]
    l8_emus = [] 
    l8_emu  = read_xa_xb_xc('OLI', 'L8', l8_toa_bands)
    for _, l8_toa in enumerate(l8s):
        l8_emus.append(l8_emu)
    return s2_emus, l8_emus
    
def get_obs(s2_files, l8_files):
    s2_obs = namedtuple('s2_obs', 'sun_angs view_angs toa cloud meta obs_time') 
    l8_obs = namedtuple('l8_obs', 'sun_angs view_angs toa cloud meta obs_time') 
    s2s = []                             
    l8s = []                             
    s2_times = []                        
    l8_times = []  
    for i in s2_files:                   
        obs_time = datetime.datetime.strptime(i[2][0].split('_MSIL1C_')[1][:15], '%Y%m%dT%H%M%S')
        s2_times.append(obs_time)        
        dat = np.array(i, dtype = np.object)[[0,1,2,3,-1]].tolist()
        s2s.append(s2_obs(dat[0], dat[1], dat[2], dat[3], dat[4], obs_time))
    for i in l8_files:                   
        with open(i[-1], 'r') as f:              
            for line in f:                   
                if 'DATE_ACQUIRED' in line:
                    date = line.split()[-1]  
                elif 'SCENE_CENTER_TIME' in line:
                    time = line.split()[-1]
        datetime_str = date + time
        obs_time     = datetime.datetime.strptime(datetime_str.split('.')[0], '%Y-%m-%d"%H:%M:%S')
        l8_times.append(obs_time)
        dat = np.array(i, dtype = np.object)[[0,1,2,3,-1]].tolist()
        l8s.append(l8_obs(dat[0], dat[1], dat[2], dat[3], dat[4], obs_time)) 
    return s2s, s2_times, l8s, l8_times

def combine_mask(target_mask, stable_targets):
    masks = []
    for stable_target in stable_targets:
        masks.append(stable_target & target_mask)
    return masks
        

def grid_conversion(array, new_shape):
    array[np.isnan(array)] = np.nanmean(array)
    rs, cs = array.shape
    x, y   = np.arange(rs), np.arange(cs)
    kx = 3 if rs > 3 else 1                       
    ky = 3 if cs > 3 else 1                       
    f      = interpolate.RectBivariateSpline(x, y, array, kx=kx, ky=ky, s=0)
    nx, ny = new_shape
    nx, ny = 1. * np.arange(nx) / nx * rs, 1. * np.arange(ny) / ny * cs
    znew   = f(nx, ny)
    return znew



# @jit(nopython=True)
# def object_jac(xap_H, xbp_H, xcp_H, xap_dH, xbp_dH, xcp_dH, toa, sur, band_weight):

#     band_weight = np.atleast_3d(band_weight).transpose(1,0,2)
#     toa     = np.atleast_3d(toa)
#     sur     = np.atleast_3d(sur)
#     boa_unc = 0.015

#     y        = xap_H * toa - xbp_H

#     sur_ref  = y / (1 + xcp_H * y) 

#     diff     = sur_ref - sur
    
#     unc_2 = boa_unc**2

#     full_J   = np.sum(0.5 * band_weight * (diff)**2 / unc_2, axis=0)

#     toa_xap_dH   = toa * xap_dH
#     toa_xap_H    = toa * xap_H
#     toa_xap_H_2  = toa_xap_H**2
#     xbp_H_xcp_dH = xbp_H * xcp_dH
#     xcp_H_2      = xcp_H**2
#     xbp_H_2      = xbp_H**2
    
#     #ddH       = -1 * (-toa * xap_dH - 2 * toa * xap_H * xbp_H * xcp_dH + 
#     #                 toa**2 * xap_H**2 * xcp_dH + xbp_dH + xbp_H**2 * xcp_dH) / (toa * xap_H * xcp_H - xbp_H * xcp_H + 1)**2 

#     # -1 * (-toa * xap_dH - 2 * toa * xap_H * xbp_H * xcp_dH + toa**2 * xap_H**2 * xcp_dH + xbp_dH + xbp_H**2 * xcp_dH)
#     above = (toa_xap_dH + 2 * toa_xap_H * xbp_H_xcp_dH - toa_xap_H_2 * xcp_dH -  xbp_dH - xbp_H_xcp_dH * xbp_H)

#     # (toa_xap_H * xcp_H - xbp_H * xcp_H + 1)**2
#     bottom = -2 * toa_xap_H * xbp_H * xcp_H_2 + toa_xap_H_2 * xcp_H_2 + 2 * toa_xap_H * xcp_H +  xbp_H_2 * xcp_H_2 - 2 * xbp_H * xcp_H + 1

#     dH = above / bottom
#     #if not np.allclose(dH, ddH):
#     #    raise
#     full_dJ  = np.sum(band_weight * dH * diff / (unc_2), axis=0)
#     return full_J, full_dJ

# @jit(nopython=True)
# def remap_J_dJ(J, dJ, pts, aero_res, shape):
#     full_J  = np.zeros(shape)
#     full_dJ = np.zeros((2,) + shape)
#     pixs    = int(np.ceil(500 / aero_res))
#     for i in range(len(pts[0])):
#         pt = pts[:,i]

#         sub = full_J [   pt[0] : pt[0] + pixs, pt[1] : pt[1] + pixs] 
#         full_J       [   pt[0] : pt[0] + pixs, pt[1] : pt[1] + pixs] = np.ones_like(sub) * J[i]

#         sub = full_dJ[0, pt[0] : pt[0] + pixs, pt[1] : pt[1] + pixs] 
#         full_dJ      [0, pt[0] : pt[0] + pixs, pt[1] : pt[1] + pixs] = np.ones_like(sub) * dJ[i][0]

#         sub = full_dJ[1, pt[0] : pt[0] + pixs, pt[1] : pt[1] + pixs] 
#         full_dJ      [1, pt[0] : pt[0] + pixs, pt[1] : pt[1] + pixs] = np.ones_like(sub) * dJ[i][1]

#     return full_J, full_dJ

@jit(nopython=True)
def prior_jac(pp, priors, prior_uncs):
    shape = priors[0].shape          
    # size  = priors[0].size 
    unc_2 = prior_uncs[:2]**2
    dif_1 = pp[0] - priors[0]
    dif_2 = pp[1] - priors[1]
    J     = 0.5 * (dif_1**2 / unc_2[0] + dif_2**2 / unc_2[1])
    dJ    = np.zeros((2,) + shape)
    dJ[0] = dif_1 / unc_2[0]
    dJ[1] = dif_2 / unc_2[1]
    return J, dJ

def prior_cost(p, auxs):
    Js  = 0
    dJs = []
    size = auxs[0].sza.size
    shape = auxs[0].sza.shape
    for i, aux in enumerate(auxs):
        pp    = p[i * size * 2 : (i+1) * size * 2].reshape(2, shape[0], shape[1])
        J, dJ = prior_jac(pp.astype(float), np.array(aux.priors).astype(float), np.array(aux.prior_uncs).astype(float))
        Js  +=  J
        dJs.append(dJ)
    return Js, np.array(dJs)

@jit(nopython=True) 
def smoothness (x, sigma_model_2):
    #hood = np.array([x[ :-2,  :-2], x[ :-2, 1:-1], 
    #                 x[ :-2,  2: ], x[1:-1,  :-2], 
    #                 x[1:-1,  2: ], x[2:  ,  :-2], 
    #                 x[ 2:,  1:-1], x[2:  , 2:  ]])

    #sigma_model_2 
    J  = np.zeros_like(x) 
    dJ = np.zeros_like(x)
    up    = slice(0, -2), slice(1, -1)
    down  = slice(1, -1), slice(0, -2)
    left  = slice(1, -1), slice(2, None)
    right = slice(2, None), slice(1, -1)
    center = slice(1,-1), slice(1,-1)
    #for sub in [x[up], x[down], x[left], x[right]]:
    for slic in [up, down, left, right]:
        diff       = x[slic] - x[center]
        J [slic]   = J [slic] + 0.5 * diff ** 2 / sigma_model_2
        dJ[slic]   = dJ[slic] + diff            / sigma_model_2 # this is essential..        
        dJ[center] = dJ[center] - diff / sigma_model_2 # this is essential as well..
    return J, dJ

# def smoothness(x, sigma_model_2):
#     n = x.shape[0]
#     DTD = np.eye(n)*2 - np.diag(np.ones(n),-1)[:n,:n] - np.diag(np.ones(n),1)[1:,1:]
#     DTD[0,0]   = 1
#     DTD[-1, -1] = 1
#     J1  = 0.5 * x.dot(DTD).dot(x.T)
#     dJ1 = DTD.dot(x.T)
#     J2  = 0.5 * x.T.dot(DTD).dot(x)
#     dJ2 = DTD.dot(x)
#     return (J1 + J2)/sigma_model_2, (dJ1.T + dJ2)/sigma_model_2
# def smoothness (x, sigma_model_2):
#     #n = x.shape[0]
#     #D = np.eye(n) - np.diag(np.ones(n),-1)[:n,:n]
#     #D[0,0] = 0
#     #D = np.matrix(D)
#     #dif = np.array((D.T * D) * np.matrix(x)) + np.array((D.T * D) * np.matrix(x).T).T
#     #mask = np.zeros_like(x).astype(bool)
#     #mask[1:-1,1:-1] = True
#     #dif[~mask] = 0
#     #J = 0.5 * x * dif
#     #return 2*J, 2*dif
#     # Build up the 8-neighbours
#     hood = np.array ( [  x[:-2, :-2], x[:-2, 1:-1], x[ :-2, 2: ], \
#                         x[ 1:-1,:-2], x[1:-1, 2:], \
#                         x[ 2:,:-2], x[ 2:, 1:-1], x[ 2:, 2:] ] )
#     j_model = np.zeros_like(x) 
#     der_j_model = np.zeros_like(x)
#     for i in [1,3,4,6]:
#         dif        = hood[i,:,:] - x[1:-1,1:-1] 
#         #dif[~mask[1:-1,1:-1]] = 0
#         j_model[1:-1,1:-1] = j_model[1:-1,1:-1] + 0.5 * dif **2/sigma_model_2
#         der_j_model[1:-1,1:-1] = der_j_model[1:-1,1:-1] - dif/sigma_model_2
    
#     return j_model, 2 * der_j_model

def smooth_cost(p, auxs, gamma1, gamma2):
    Js  = 0
    dJs = []
    inv_gamma1 = (1 / gamma1)**2
    inv_gamma2 = (1 / gamma2)**2
    for i, aux in enumerate(auxs):
        size  = aux.sza.size        
        shape = aux.sza.shape 
        pp    = p[i * size * 2 : (i+1) * size * 2].reshape(2, shape[0], shape[1])
        J_aot,  dJ_aot  = smoothness(pp[0],  inv_gamma1)
        J_tcwv, dJ_tcwv = smoothness(pp[1],  inv_gamma2)
        J, dJ      = J_aot + J_tcwv, np.array([dJ_aot, dJ_tcwv])
        Js  += J         
        #dJs +=dJ       
        dJs.append(dJ)
    return Js, np.array(dJs)


def predic_xps_all(X, emus):                                                                                                                                                  
    sza, vza, raa, aot, tcwv, tco3, ele = X
    xap_Hs  = []       
    xbp_Hs  = []       
    xcp_Hs  = []       
    xap_dHs = []       
    xbp_dHs = []       
    xcp_dHs = []       
    for i in range(vza.shape[0]): 
        xx    = np.array([sza, vza[i], raa[i], aot, tcwv, tco3, ele]).T
        H, dH = emus.xap[i].predict(xx, cal_jac=True)[0]
        xap_Hs.append(H)
        xap_dHs.append(dH[:,3:5])
                       
        H, dH = emus.xbp[i].predict(xx, cal_jac=True)[0]
        xbp_Hs.append(H)
        xbp_dHs.append(dH[:,3:5])
                       
        H, dH = emus.xcp[i].predict(xx, cal_jac=True)[0]
        xcp_Hs.append(H)
        xcp_dHs.append(dH[:,3:5])
                       
    return np.array(xap_Hs), np.array(xbp_Hs), np.array(xcp_Hs), np.array(xap_dHs), np.array(xbp_dHs), np.array(xcp_dHs)

# @jit(nopython=True)
# def cal_sur(xap_H, xbp_H, xcp_H, xap_dH, xbp_dH, xcp_dH, toa):
     
#     toa     = np.atleast_3d(toa)
#     y        = xap_H * toa - xbp_H
     
#     sur_ref  = y / (1 + xcp_H * y) 
     
#     toa_xap_dH   = toa * xap_dH
#     toa_xap_H    = toa * xap_H
#     toa_xap_H_2  = toa_xap_H**2
#     xbp_H_xcp_dH = xbp_H * xcp_dH
#     xcp_H_2      = xcp_H**2
#     xbp_H_2      = xbp_H**2
     
#     #ddH       = -1 * (-toa * xap_dH - 2 * toa * xap_H * xbp_H * xcp_dH + 
#     #                 toa**2 * xap_H**2 * xcp_dH + xbp_dH + xbp_H**2 * xcp_dH) / (toa * xap_H * xcp_H - xbp_H * xcp_H + 1)**2 
     
#     # -1 * (-toa * xap_dH - 2 * toa * xap_H * xbp_H * xcp_dH + toa**2 * xap_H**2 * xcp_dH + xbp_dH + xbp_H**2 * xcp_dH)
#     above = (toa_xap_dH + 2 * toa_xap_H * xbp_H_xcp_dH - toa_xap_H_2 * xcp_dH -  xbp_dH - xbp_H_xcp_dH * xbp_H)
     
#     # (toa_xap_H * xcp_H - xbp_H * xcp_H + 1)**2
#     bottom = -2 * toa_xap_H * xbp_H * xcp_H_2 + toa_xap_H_2 * xcp_H_2 + 2 * toa_xap_H * xcp_H +  xbp_H_2 * xcp_H_2 - 2 * xbp_H * xcp_H + 1

#     dH = above / bottom

#     #full_dJ  = np.sum(dH, axis=0)

#     return sur_ref, dH



# def obs_cost(p, obs, emus, auxs, pix_res, aero_res):
#     alpha = -1.6
#     band_weight = np.array([0.469, 0.555, 0.645, 0.859, 1.64 , 2.13 ])**alpha  
#     Js  = 0
#     dJs = 0
#     for _, ob in enumerate(obs):        
#         emu = emus[_]
#         aux = auxs[_]
#         pts = (ob.cors_pts * 500 / aero_res).astype(int)
    
#         size   = aux.sza.size
#         shape  = aux.sza.shape
#         pp     = p[_ * size * 2 : (_+1) * size * 2].reshape(2, shape[0], shape[1])
    
#         aot    = pp[0]               [   pts[0], pts[1]] 
#         tcwv   = pp[1]               [   pts[0], pts[1]] 
#         sza    = np.array(aux.sza)   [   pts[0], pts[1]] 
#         vza    = np.array(aux.vza)   [:, pts[0], pts[1]] 
#         raa    = np.array(aux.raa)   [:, pts[0], pts[1]] 
#         ele    = np.array(aux.ele)   [   pts[0], pts[1]]
#         tco3   = np.array(aux.priors)[2, pts[0], pts[1]]
#         X      = [sza, vza, raa, aot, tcwv, tco3, ele]
#         xap_Hs, xbp_Hs, xcp_Hs, xap_dHs, xbp_dHs, xcp_dHs  = predic_xps(X, emu)
        
#         J, dJ  = object_jac(xap_Hs, xbp_Hs, xcp_Hs, xap_dHs, xbp_dHs, xcp_dHs, ob.conv_toa, ob.cors_sur, band_weight)       
#         J, dJ = remap_J_dJ(J, dJ, pts, aero_res, shape)
#         Js  += J
#         dJs += dJ                                                                                                                                                         
#     return Js, dJs

@jit(nopython=True)
def choose_random_pixs(mask, pts, aero_res, pix_res, pix_area, choosen):

    # choosen = min(int(0.5 * (aero_res // pix_res)**2), 100)   
    sels = np.zeros((2, choosen, pts[0].shape[0]))
    #selys = np.zeros((4, pts[0].shape[0]))
    for i in range(len(pts[0])):
        pt         = pts[:,i]
        x_start    = pt[0] * pix_area
        y_start    = pt[1] * pix_area
        sub        = mask [x_start : x_start + pix_area,  y_start : y_start + pix_area]
        indx, indy = np.where(sub)
        ind        = np.arange(len(indx))
        np.random.shuffle(ind)
        sels[0, :, i] = int(pt[0] * pix_area) + indx[ind[:choosen]]
        sels[1, :, i] = int(pt[1] * pix_area) + indy[ind[:choosen]]
     
    return sels

def get_targets(obs, masks, pix_res, aero_res, target_mask, aero_shape, toa_shape):

    to_compute  = []
    coarse_inds = []
    fine_inds   = []
    fine_xys    = []
    #aero_shape = auxs[0].sza.shape                                           
    #toa_shape  = toas[0][0].shape
    pix_area   = int(np.ceil(aero_res / pix_res))                            
    # ratx, raty = int(aero_shape[0] * pix_area), int(aero_shape[1] * pix_area)

    valid_num      = np.array(masks).sum(axis=0)
    compute_mask   = valid_num>=2
    selxs , selys  = np.where(compute_mask)
    # ratio          = aero_res // pix_res
    proportion     = 0.75#(pix_area**2 - 32) / pix_area**2
    discard        = np.random.choice(len(selxs), int(proportion * len(selxs)), replace=False)
    compute_mask[selxs[discard], selys[discard]] = False
    # temp = np.zeros((ratx, raty)).astype(bool)                           
    # # to guarentee  consistent shape   
    # temp[:min(ratx, toa_shape[0]), :min(raty,toa_shape[1])] = compute_mask[:min(ratx, toa_shape[0]), :min(raty,toa_shape[1])]                                                                           
    # temp  = temp.reshape(aero_shape[0], pix_area, aero_shape[1], pix_area).sum(axis = (1,3))
    #ratio = aero_res // pix_res
    #pts   = np.array(np.where(temp>0))                                                  
    
    #selxs , selys = pts#choose_random_pixs(target_mask, pts, aero_res, pix_res, pix_area).astype(int)
    
    for _, ob in enumerate(obs):                                                     
        mask = masks[_] &  compute_mask                                                      
        #mm   = mask[selxs,  selys]   
        selx, sely = np.where(mask)     
        #selx = selxs[mm]                                                             
        #sely = selys[mm]      
        pts1 = (selx * pix_res / aero_res).astype(int), (sely * pix_res / aero_res).astype(int)
        mm = (pts1[0] < aero_shape[0]) & (pts1[1] < aero_shape[1])
        selx, sely = selx[mm], sely[mm]
        pts1 = pts1[0][mm], pts1[1][mm]
        pts2 = (ob.cors_pts * 500 / aero_res).astype(int)                          
        ppts = np.hstack([pts1, pts2])
        if ppts.shape[1]>0:
            inds, inv = np.unique(ppts, axis=1, return_inverse=True)
            to_compute.append(inds)
            fine_inds.append(  inv[:len(pts1[0])])
            coarse_inds.append(inv[len(pts1[0]):])
            fine_xys.append(np.array([selx, sely]))
        else:
            inds, inv = np.zeros((2,0)), np.zeros((0,))
            to_compute.append(inds)                
            fine_inds.append(  inv[:len(pts1[0])]) 
            coarse_inds.append(inv[len(pts1[0]):]) 
            fine_xys.append(np.array([selx, sely]))
    return to_compute, fine_inds, coarse_inds, fine_xys
     

# def get_xps(p, emus, auxs, to_compute):
#     xps    = []
#     for _, aux in enumerate(auxs):                 
#         emu = emus[_]                  
#         size   = aux.sza.size                    
#         shape  = aux.sza.shape                   
#         pp     = p[_ * size * 2 : (_+1) * size * 2].reshape(2, shape[0], shape[1])
#         pts    = to_compute[_]                                                 
#         if pts.shape[1] > 0:
#             ppts1, ppts2 = np.minimum(pts[0], pp[0].shape[0]-1), np.minimum(pts[1], pp[0].shape[1]-1)
#             aot    = pp[0]               [   ppts1, ppts2] 
#             tcwv   = pp[1]               [   ppts1, ppts2]
#             sza    = np.array(aux.sza)   [   ppts1, ppts2] 
#             vza    = np.array(aux.vza)   [:, ppts1, ppts2] 
#             raa    = np.array(aux.raa)   [:, ppts1, ppts2] 
#             ele    = np.array(aux.ele)   [   ppts1, ppts2]
#             tco3   = np.array(aux.priors)[2, ppts1, ppts2]
#             X      = [sza, vza, raa, aot, tcwv, tco3, ele]
#             xps.append(predic_xps_all(X, emu))
#         else:
#             xps.append([np.zeros((7, 0, 1)), np.zeros((7, 0, 1)), np.zeros((7, 0, 1)), 
#                         np.zeros((7, 0, 2)), np.zeros((7, 0, 2)), np.zeros((7, 0, 2)),])
        
#     return xps

# def coarse_cost(xps, obs, coarse_inds, to_compute, band_weight, aero_shape, aero_res):
#     Js  = 0
#     dJs = []
#     for _, ob in enumerate(obs):
#         ind = coarse_inds[_]
#         if ind.shape[0] > 0:
#             pts =  to_compute[_][:,ind]
#             xap_Hs, xbp_Hs, xcp_Hs, xap_dHs, xbp_dHs, xcp_dHs = xps[_]
#             xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH  = xap_Hs[1:, ind], xbp_Hs[1:, ind], xcp_Hs[1:, ind], \
#                                                                 xap_dHs[1:, ind], xbp_dHs[1:, ind], xcp_dHs[1:, ind]
#             J, dJ  = object_jac(xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH, ob.conv_toa, ob.cors_sur, band_weight)
#             J, dJ  = remap_J_dJ(J, dJ, pts, aero_res, aero_shape)
               
#             Js  += J
#             dJs.append(dJ)
#         else:
#             Js += np.zeros(aero_shape)
#             dJs.append(np.zeros((2,) + aero_shape))
#     return Js, np.array(dJs)

@jit(nopython=True)
def remap_J_dJ_coarse(J, dJ, pts, shape):
    full_J  = np.zeros(shape)
    full_dJ = np.zeros((2,) + shape)
    #pts = (ob.cors_pts * 500 / aero_res).astype(int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            mask = (pts[0] == i) & (pts[1] == j)
            full_J[i,j]  = J[mask].sum()
            full_dJ[0, i, j] = dJ[:,0][mask].sum()
            full_dJ[1, i, j] = dJ[:,1][mask].sum()
    return full_J, full_dJ

def get_xps(p, emus, auxs):
    xps    = []
    for _, aux in enumerate(auxs):                 
        emu = emus[_]                  
        size   = aux.sza.size                    
        shape  = aux.sza.shape                   
        pp     = p[_ * size * 2 : (_+1) * size * 2].reshape(2, shape[0], shape[1])
        #pts    = to_compute[_]                                                 
        if size > 0:
            #ppts1, ppts2 = np.minimum(pts[0], pp[0].shape[0]-1), np.minimum(pts[1], pp[0].shape[1]-1)
            aot    = pp[0].ravel()               #[   ppts1, ppts2] 
            tcwv   = pp[1].ravel()               #[   ppts1, ppts2]
            sza    = np.array(aux.sza).ravel()   #[   ppts1, ppts2] 
            vza    = np.array(aux.vza).reshape(len(aux.vza), -1)  #[:, ppts1, ppts2] 
            raa    = np.array(aux.raa).reshape(len(aux.raa), -1)   #[:, ppts1, ppts2] 
            ele    = np.array(aux.ele).ravel()   #[   ppts1, ppts2]
            tco3   = np.array(aux.priors)[2].ravel() #[2, ppts1, ppts2]
            X      = [sza, vza, raa, aot, tcwv, tco3, ele]
            xps.append(predic_xps_all(X, emu))
        else:
            xps.append([np.zeros((7, 0, 1)), np.zeros((7, 0, 1)), np.zeros((7, 0, 1)), 
                        np.zeros((7, 0, 2)), np.zeros((7, 0, 2)), np.zeros((7, 0, 2)),])
        
    return xps

@jit(nopython=True)
def cal_sur(xap_H, xbp_H, xcp_H, xap_dH, xbp_dH, xcp_dH, toa):
     
    toa     = np.atleast_3d(toa)
    y        = xap_H * toa - xbp_H
     
    sur_ref  = y / (1 + xcp_H * y) 
     
    toa_xap_dH   = toa * xap_dH
    toa_xap_H    = toa * xap_H
    toa_xap_H_2  = toa_xap_H**2
    xbp_H_xcp_dH = xbp_H * xcp_dH
    xcp_H_2      = xcp_H**2
    xbp_H_2      = xbp_H**2
     
    #ddH       = -1 * (-toa * xap_dH - 2 * toa * xap_H * xbp_H * xcp_dH + 
    #                 toa**2 * xap_H**2 * xcp_dH + xbp_dH + xbp_H**2 * xcp_dH) / (toa * xap_H * xcp_H - xbp_H * xcp_H + 1)**2 
     
    # -1 * (-toa * xap_dH - 2 * toa * xap_H * xbp_H * xcp_dH + toa**2 * xap_H**2 * xcp_dH + xbp_dH + xbp_H**2 * xcp_dH)
    above = (toa_xap_dH + 2 * toa_xap_H * xbp_H_xcp_dH - toa_xap_H_2 * xcp_dH -  xbp_dH - xbp_H_xcp_dH * xbp_H)
     
    # (toa_xap_H * xcp_H - xbp_H * xcp_H + 1)**2
    bottom = -2 * toa_xap_H * xbp_H * xcp_H_2 + toa_xap_H_2 * xcp_H_2 + 2 * toa_xap_H * xcp_H +  xbp_H_2 * xcp_H_2 - 2 * xbp_H * xcp_H + 1

    dH = above / bottom

    return sur_ref, dH

@jit(nopython=True)
def object_jac(xap_H, xbp_H, xcp_H, xap_dH, xbp_dH, xcp_dH, toa, sur, band_weight):

    band_weight = np.atleast_3d(band_weight).transpose(1,0,2)
    toa     = np.atleast_3d(toa)
    sur     = np.atleast_3d(sur)
    boa_unc = 0.05
    sur_ref, dH = cal_sur(xap_H, xbp_H, xcp_H, xap_dH, xbp_dH, xcp_dH, toa)
    diff     = sur_ref - sur
    unc_2 = boa_unc**2
    full_J   = np.sum(0.5 * band_weight * (diff)**2 / unc_2, axis=0)
    full_dJ  = np.sum(band_weight * dH * diff / (unc_2), axis=0)
    return full_J, full_dJ

def coarse_cost(xps, obs, band_weight, aero_shape, aero_res):
    # xps = get_xps(p, s2_emus + l8_emus, s2_auxs + l8_auxs, to_compute)
    Js  = 0
    dJs = []
    for _, ob in enumerate(obs):
        pts = (ob.cors_pts * 500 / aero_res).astype(int)
        if pts.size > 0:
            xap_Hs, xbp_Hs, xcp_Hs, xap_dHs, xbp_dHs, xcp_dHs = xps[_]
            xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH  = xap_Hs[1:].reshape((6,) + aero_shape + (1,)), xbp_Hs[1:].reshape((6,) + aero_shape + (1,)), xcp_Hs[1:].reshape((6,) + aero_shape + (1,)), \
                                                                xap_dHs[1:].reshape((6,) + aero_shape + (2,)), xbp_dHs[1:].reshape((6,) + aero_shape + (2,)), xcp_dHs[1:].reshape((6,) + aero_shape + (2,))
            
            
            J, dJ  = object_jac(xap_H[:, pts[0], pts[1]],  xbp_H[:, pts[0], pts[1]],  xcp_H[:, pts[0], pts[1]],  xap_dH[:, pts[0], pts[1]],  xbp_dH[:, pts[0], pts[1]],  xcp_dH[:, pts[0], pts[1]], ob.conv_toa, ob.cors_sur, band_weight)
            J, dJ  = remap_J_dJ_coarse(J, dJ, pts, aero_shape)
            Js  += J
            dJs.append(dJ)
        else:
            Js += np.zeros(aero_shape)
            dJs.append(np.zeros((2,) + aero_shape))
    return Js, np.array(dJs)

# def fcoarse_cost(p, obs, coarse_inds, to_compute, band_weight, aero_shape, aero_res,s2_emus, l8_emus, s2_auxs, l8_auxs):
#     # xps = get_xps(p, s2_emus + l8_emus, s2_auxs + l8_auxs, to_compute)
#     xps = get_xps(p, s2_emus + l8_emus, s2_auxs + l8_auxs) 
#     Js  = 0
#     dJs = []
#     for _, ob in enumerate(obs):
#         pts = (ob.cors_pts * 500 / aero_res).astype(int)
#         if pts.size > 0:
#             xap_Hs, xbp_Hs, xcp_Hs, xap_dHs, xbp_dHs, xcp_dHs = xps[_]
#             xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH  = xap_Hs[1:].reshape((6,) + aero_shape + (1,)), xbp_Hs[1:].reshape((6,) + aero_shape + (1,)), xcp_Hs[1:].reshape((6,) + aero_shape + (1,)), \
#                                                                 xap_dHs[1:].reshape((6,) + aero_shape + (2,)), xbp_dHs[1:].reshape((6,) + aero_shape + (2,)), xcp_dHs[1:].reshape((6,) + aero_shape + (2,))
            
            
#             J, dJ  = object_jac(xap_H[:, pts[0], pts[1]],  xbp_H[:, pts[0], pts[1]],  xcp_H[:, pts[0], pts[1]],  xap_dH[:, pts[0], pts[1]],  xbp_dH[:, pts[0], pts[1]],  xcp_dH[:, pts[0], pts[1]], ob.conv_toa, ob.cors_sur, band_weight)
#             J, dJ  = remap_J_dJ_coarse(J, dJ, aero_res, aero_shape, ob)
#             Js  += J.sum()
#             dJs.append(dJ)
#         else:
#             Js += np.zeros(aero_shape)
#             dJs.append(np.zeros((2,) + aero_shape))
#     return Js.sum()



def fine_cost_optimal(xps, toas, masks, band_weight, bands, aero_shape, aero_res, pix_res):
    
    all_surs, all_surs_dH, valid_masks = [], [], []
    
    valid_num      = np.array(masks).sum(axis=0)
    compute_mask   = valid_num>=len(toas)
    for _, toa in enumerate(toas):
        #fine_ind = fine_inds[_]
        mask  = (~toa.mask).all(axis=0) &  compute_mask 
        selx, sely = np.where(mask)
        pts = np.array([selx, sely])
        pts = (pts * pix_res / aero_res).astype(int)
        if pts.size > 0:
            xap_Hs, xbp_Hs, xcp_Hs, xap_dHs, xbp_dHs, xcp_dHs = xps[_]
            xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH  = xap_Hs.reshape((7,) + aero_shape + (1,)), xbp_Hs.reshape((7,) + aero_shape + (1,)), xcp_Hs.reshape((7,) + aero_shape + (1,)), \
                                                                xap_dHs.reshape((7,) + aero_shape + (2,)), xbp_dHs.reshape((7,) + aero_shape + (2,)), xcp_dHs.reshape((7,) + aero_shape + (2,))
            # pts = to_compute[_][:,fine_ind]
            # surs = 0       
            # dHs  = 0               
            band = bands[_]
            # num_pixels = min(selx.shape[0], 100)
            #for i in range(num_pixels):
            sur, dH  = cal_sur(xap_H[:, pts[0], pts[1]],  xbp_H[:, pts[0], pts[1]],  xcp_H[:, pts[0], pts[1]],  xap_dH[:, pts[0], pts[1]],  xbp_dH[:, pts[0], pts[1]],  xcp_dH[:, pts[0], pts[1]], toa[band][:, selx, sely].data)
            #sur, dH = cal_sur(xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH, toa[band][:, selx, sely].data)
            #     surs += sur
            #     dHs  += dH
            # surs /= num_pixels
            # dHs  /= num_pixels
        
            # instead we create a full image
            toa_shape = toa[0].shape
            temp =  np.zeros((7,) + toa_shape + (1,))
            temp[:, selx, sely] = sur
            all_surs.append(temp)   
            
            temp = np.zeros((7,) + toa_shape + (2,))
            temp[:, selx, sely] = dH
            all_surs_dH.append(temp)
            
            valid_mask = np.zeros(toa_shape)
            valid_mask[selx, sely] = 1
            
            valid_mask[0,  :] = 0
            valid_mask[:,  0] = 0
            valid_mask[-1, :] = 0
            valid_mask[:, -1] = 0

            valid_masks.append(valid_mask)       
        else:
            toa_shape = toa[0].shape
            temp = np.zeros((7,) + toa_shape + (1,))
            all_surs.append(temp)    
            temp = np.zeros((7,) + toa_shape + (2,))
            all_surs_dH.append(temp) 
            valid_mask = np.zeros(toa_shape)
            valid_mask[0,  :] = 0    
            valid_mask[:,  0] = 0    
            valid_mask[-1, :] = 0    
            valid_mask[:, -1] = 0    
                                     
            valid_masks.append(valid_mask)       
    

    #selxs , selys  = np.where(compute_mask)
    
    count_sum = np.sum(valid_masks, axis=0)[None, ..., None]
    mean_surs = np.sum(all_surs,    axis=0) / count_sum
    mean_dHs  = np.sum(all_surs_dH, axis=0) / count_sum / len(toas)
    
    # D[1/2*(a(x) - (a(x)+b(x)+g(x)+d(x))/4)^2 + 
    #   1/2*(b(x) - (a(x)+b(x)+g(x)+d(x))/4)^2 +  
    #   1/2*(g(x) - (a(x)+b(x)+g(x)+d(x))/4)^2 + 
    #   1/2*(d(x) - (a(x)+b(x)+g(x)+d(x))/4)^2, x]
    Js         = 0
    dJs        = []
    for i in range(len(all_surs)):
        # mask  = toas[i].mask
        # selx, sely = np.where(mask.any(axis=0))
        # pts = np.array([selx, sely])
        # pts = (pts * pix_res / aero_res).astype(int)
        diff = all_surs[i] - mean_surs 
        J    = 0.5 * diff**2
        dJ   = diff * (all_surs_dH[i] - mean_dHs)   
        J [:, valid_masks[i] == 0] = 0
        dJ[:, valid_masks[i] == 0] = 0
        J  =  J * band_weight[...,None, None, None]
        dJ = dJ * band_weight[...,None, None, None]
        #J, dJ = remap_J_dJ_coarse(J, dJ, pts, aero_shape)
        Js  += J
        dJs.append(dJ) 

    unc_2 = 0.025**2
    Js  = Js.sum(axis=(0,3)) / unc_2
    dJs = np.array(dJs).sum(axis=1) / unc_2

    pix_area   = int(np.ceil(aero_res / pix_res)) 
    ratx, raty = int(aero_shape[0] * pix_area), int(aero_shape[1] * pix_area)
    temp = np.zeros((ratx, raty))                         
    # to guarentee  consistent shape   
    temp[:min(ratx, toa_shape[0]), :min(raty,toa_shape[1])] = Js[:min(ratx, toa_shape[0]), :min(raty,toa_shape[1])]
    Js  = temp.reshape(aero_shape[0], pix_area, aero_shape[1], pix_area).sum(axis = (1,3))
    
    temp = np.zeros((len(all_surs), ratx, raty, 2))                          
    # to guarentee  consistent shape   
    temp[:, :min(ratx, toa_shape[0]), :min(raty,toa_shape[1]), :] = dJs[:, :min(ratx, toa_shape[0]), :min(raty,toa_shape[1]), :]
    dJs  = temp.reshape(len(all_surs), aero_shape[0], pix_area, aero_shape[1], pix_area, 2).sum(axis = (2,4))
    
    return  Js, dJs.transpose(0, 3, 1, 2)


def fine_cost(xps, toas, masks, band_weight, bands, aero_shape, aero_res, pix_res):
    #xps = get_xps(p, s2_emus + l8_emus, s2_auxs + l8_auxs)
    all_surs, all_surs_dH, valid_masks = [], [], []
    
    valid_num      = np.array(masks).sum(axis=0)
    compute_mask   = valid_num>=len(toas)
    
    selx, sely = np.where(compute_mask)
    selx, sely = np.minimum(selx, compute_mask.shape[0] - 1), np.minimum(sely, compute_mask.shape[1] - 1)
    pix_area   = int(np.ceil(aero_res / pix_res)) 
    
    for _, toa in enumerate(toas):
        #fine_ind = fine_inds[_]
        band = bands[_]
        toa_shape = toa[0].shape
        #mask  = (~toa.mask).all(axis=0) &  compute_mask 
        btoa = toa[band].data
        
        ratx, raty = int(aero_shape[0] * pix_area), int(aero_shape[1] * pix_area)
        
        temp = np.zeros((len(band), ratx, raty)) * np.nan

        temp[:, selx, sely] = btoa[:, selx, sely]
        
        btoa = np.nanmean(temp.reshape(len(band), aero_shape[0], pix_area, aero_shape[1], pix_area), axis = (2,4))
        
        mask = np.isfinite(btoa).all(axis=0)

        pts = np.array(np.where(mask))

        if pts.size > 0:
            xap_Hs, xbp_Hs, xcp_Hs, xap_dHs, xbp_dHs, xcp_dHs = xps[_]
            xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH  = xap_Hs.reshape((7,) + aero_shape + (1,)), xbp_Hs.reshape((7,) + aero_shape + (1,)), xcp_Hs.reshape((7,) + aero_shape + (1,)), \
                                                                xap_dHs.reshape((7,) + aero_shape + (2,)), xbp_dHs.reshape((7,) + aero_shape + (2,)), xcp_dHs.reshape((7,) + aero_shape + (2,))
            # pts = to_compute[_][:,fine_ind]
            # surs = 0       
            # dHs  = 0               
            
            # num_pixels = min(selx.shape[0], 100)
            #for i in range(num_pixels):
            sur, dH  = cal_sur(xap_H[:, pts[0], pts[1]],  xbp_H[:, pts[0], pts[1]],  xcp_H[:, pts[0], pts[1]],  xap_dH[:, pts[0], pts[1]],  xbp_dH[:, pts[0], pts[1]],  xcp_dH[:, pts[0], pts[1]], btoa[:, pts[0], pts[1]])
            #sur, dH = cal_sur(xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH, toa[band][:, selx, sely].data)
            #     surs += sur
            #     dHs  += dH
            # surs /= num_pixels
            # dHs  /= num_pixels
        
            # instead we create a full image

            temp =  np.zeros((7,) + aero_shape + (1,))
            temp[:, pts[0], pts[1]] = sur
            all_surs.append(temp)   
            
            temp = np.zeros((7,) + aero_shape + (2,))
            temp[:, pts[0], pts[1]] = dH
            all_surs_dH.append(temp)
            
            valid_mask = np.zeros(aero_shape)
            valid_mask[pts[0], pts[1]] = 1
            
            # valid_mask[0,  :] = 0
            # valid_mask[:,  0] = 0
            # valid_mask[-1, :] = 0
            # valid_mask[:, -1] = 0

            valid_masks.append(valid_mask)       
        else:
            temp = np.zeros((7,) + aero_shape + (1,))
            all_surs.append(temp)    
            temp = np.zeros((7,) + aero_shape + (2,))
            all_surs_dH.append(temp) 
            valid_mask = np.zeros(aero_shape)
            # valid_mask[0,  :] = 0    
            # valid_mask[:,  0] = 0    
            # valid_mask[-1, :] = 0    
            # valid_mask[:, -1] = 0    
                                     
            valid_masks.append(valid_mask)       
    

    #selxs , selys  = np.where(compute_mask)
    
    count_sum = np.sum(valid_masks, axis=0)[None, ..., None]
    #mean_surs = np.sum(all_surs,    axis=0) / len(toas)
    #mean_dHs  = np.nanmean(all_surs_dH, axis=0)
    mean_surs = np.nanmean(all_surs, axis=0)
    # D[1/2*(a(x) - (a(x)+b(x)+g(x)+d(x))/4)^2 + 
    #   1/2*(b(x) - (a(x)+b(x)+g(x)+d(x))/4)^2 +  
    #   1/2*(g(x) - (a(x)+b(x)+g(x)+d(x))/4)^2 + 
    #   1/2*(d(x) - (a(x)+b(x)+g(x)+d(x))/4)^2, x]
    Js         = 0
    dJs        = []
    for i in range(len(all_surs)):
        # mask  = toas[i].mask
        # selx, sely = np.where(mask.any(axis=0))
        # pts = np.array([selx, sely])
        # pts = (pts * pix_res / aero_res).astype(int)
        diff = all_surs[i] - mean_surs 
        J    = 0.5 * diff**2
        dJ   = diff * all_surs_dH[i]   
        J [:, valid_masks[i] == 0] = 0
        dJ[:, valid_masks[i] == 0] = 0
        J  =  J * band_weight[...,None, None, None]
        dJ = dJ * band_weight[...,None, None, None]
        #J, dJ = remap_J_dJ_coarse(J, dJ, pts, aero_shape)
        Js  += J
        dJs.append(dJ)      
    unc_2 = 0.05**2
    Js  = Js.sum(axis=(0,3)) / unc_2
    dJs = np.array(dJs).sum(axis=1) / unc_2

    # pix_area   = int(np.ceil(aero_res / pix_res)) 
    # ratx, raty = int(aero_shape[0] * pix_area), int(aero_shape[1] * pix_area)
    # temp = np.zeros((ratx, raty))                         
    # # to guarentee  consistent shape   
    # temp[:min(ratx, toa_shape[0]), :min(raty,toa_shape[1])] = Js[:min(ratx, toa_shape[0]), :min(raty,toa_shape[1])]
    # Js  = temp.reshape(aero_shape[0], pix_area, aero_shape[1], pix_area).sum(axis = (1,3))
    
    # temp = np.zeros((len(all_surs), ratx, raty, 2))                          
    # # to guarentee  consistent shape   
    # temp[:, :min(ratx, toa_shape[0]), :min(raty,toa_shape[1]), :] = dJs[:, :min(ratx, toa_shape[0]), :min(raty,toa_shape[1]), :]
    # dJs  = temp.reshape(len(all_surs), aero_shape[0], pix_area, aero_shape[1], pix_area, 2).sum(axis = (2,4))
    
    return  Js.sum(), dJs.transpose(0, 3, 1, 2)

# def fine_cost(xps, toas, fine_inds, to_compute, band_weight, bands, fine_xys, aero_shape, aero_res, pix_res):

#     all_surs, all_surs_dH, valid_masks = [], [], []

#     for _, toa in enumerate(toas):
#         fine_ind = fine_inds[_]

#         if fine_ind.shape[0]>0:
#             xap_Hs, xbp_Hs, xcp_Hs, xap_dHs, xbp_dHs, xcp_dHs = xps[_]
#             xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH  = xap_Hs[:, fine_ind], xbp_Hs[:, fine_ind], xcp_Hs[:, fine_ind], \
#                                                                 xap_dHs[:, fine_ind], xbp_dHs[:, fine_ind], xcp_dHs[:, fine_ind]
#             # pts = to_compute[_][:,fine_ind]
#             selx, sely = fine_xys[_]
#             # surs = 0       
#             # dHs  = 0               
#             band = bands[_]
#             # num_pixels = min(selx.shape[0], 100)
#             #for i in range(num_pixels):
#             sur, dH = cal_sur(xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH, toa[band][:, selx, sely].data)
#             #     surs += sur
#             #     dHs  += dH
#             # surs /= num_pixels
#             # dHs  /= num_pixels
        
#             # instead we create a full image
#             toa_shape = toa[0].shape
#             temp =  np.zeros((7,) + toa_shape + (1,))
#             temp[:, selx, sely] = sur
#             all_surs.append(temp)   
            
#             temp = np.zeros((7,) + toa_shape + (2,))
#             temp[:, selx, sely] = dH
#             all_surs_dH.append(temp)
            
#             valid_mask = np.zeros(toa_shape)
#             valid_mask[selx, sely] = 1
            
#             valid_mask[0,  :] = 0
#             valid_mask[:,  0] = 0
#             valid_mask[-1, :] = 0
#             valid_mask[:, -1] = 0

#             valid_masks.append(valid_mask)       
#         else:
#             toa_shape = toa[0].shape
#             temp = np.zeros((7,) + toa_shape + (1,))
#             all_surs.append(temp)    
#             temp = np.zeros((7,) + toa_shape + (2,))
#             all_surs_dH.append(temp) 
#             valid_mask = np.zeros(toa_shape)
#             valid_mask[0,  :] = 0    
#             valid_mask[:,  0] = 0    
#             valid_mask[-1, :] = 0    
#             valid_mask[:, -1] = 0    
                                     
#             valid_masks.append(valid_mask)       

#     count_sum = np.sum(valid_masks, axis=0)[None, ..., None]
#     mean_surs = np.sum(all_surs,    axis=0) / count_sum
#     mean_dHs  = np.sum(all_surs_dH, axis=0) / count_sum
#     # D[1/2*(a(x) - (a(x)+b(x)+g(x)+d(x))/4)^2 + 
#     #   1/2*(b(x) - (a(x)+b(x)+g(x)+d(x))/4)^2 + 
#     #   1/2*(g(x) - (a(x)+b(x)+g(x)+d(x))/4)^2 + 
#     #   1/2*(d(x) - (a(x)+b(x)+g(x)+d(x))/4)^2, x]
#     Js         = 0
#     dJs        = []
#     for i in range(len(all_surs)):
#         diff = all_surs[i] - mean_surs
#         J    = 0.5 * diff**2
#         dJ   = diff * (all_surs_dH[i] - mean_dHs)   
#         J [:, valid_masks[i] == 0] = 0
#         dJ[:, valid_masks[i] == 0] = 0
#         Js   += J * band_weight[...,None,None,None]
#         dJs.append(dJ * band_weight[...,None,None,None])
    
#     Js  = Js.sum(axis=(0,3)) /0.025**2
#     dJs = np.array(dJs).sum(axis=1) / 0.025**2

#     pix_area   = int(np.ceil(aero_res / pix_res)) 
#     ratx, raty = int(aero_shape[0] * pix_area), int(aero_shape[1] * pix_area)
#     temp = np.zeros((ratx, raty))                         
#     # to guarentee  consistent shape   
#     temp[:min(ratx, toa_shape[0]), :min(raty,toa_shape[1])] = Js[:min(ratx, toa_shape[0]), :min(raty,toa_shape[1])]
#     Js  = temp.reshape(aero_shape[0], pix_area, aero_shape[1], pix_area).sum(axis = (1,3))
    
#     temp = np.zeros((len(all_surs), ratx, raty, 2))                          
#     # to guarentee  consistent shape   
#     temp[:, :min(ratx, toa_shape[0]), :min(raty,toa_shape[1]), :] = dJs[:, :min(ratx, toa_shape[0]), :min(raty,toa_shape[1]), :]
#     dJs  = temp.reshape(len(all_surs), aero_shape[0], pix_area, aero_shape[1], pix_area, 2).sum(axis = (2,4))
    
#     return  Js, dJs.transpose(0, 3, 1, 2)

def get_starting_aots(obs, auxs, aero_res):
    iso_tnn = Two_NN(np_model_file= file_path + '/emus/Inverse_6S_AOT_iso.npz')
    aots  = []
    shape = auxs[0].priors[0].shape 
    for i in range(len(obs)):
        ob  = obs[i]
        aux = auxs[i]
        if ob.cors_sur.shape[1] < 10:
            aot = aux.priors[0]
        else:
            xx = np.vstack([ob.conv_toa[:2], ob.cors_sur[:2], np.ones_like(ob.cors_sur[:5]) * np.atleast_2d([aux.sza.mean(), aux.vza[0].mean(), aux.raa[0][0,0], aux.priors[2].mean(), aux.ele.mean()]).T]) 
            aot   = iso_tnn.predict(xx.T)[0].ravel() 
            # aot_min, aot_max = np.nanpercentile(aot, [15, 85])
            # indx, indy = (ob.cors_pts * 500 / aero_res).astype(int)
            # myInterpolator = NearestNDInterpolator((indx, indy), aot) 
            # grid_x, grid_y = np.mgrid[0:shape[0]:1, 0:shape[1]:1,]
            # aot = myInterpolator(grid_x, grid_y)
            # aot = smoothn(aot, isrobust=True, TolZ=1e-6, W = 100*((aot >= aot_min) & (aot <=aot_max)), s=10, MaxIter=1000)[0]
            aot = np.nanmedian(aot)*np.ones(shape)
        aots.append(aot.astype(float))
    return aots

def get_s2_aots(auxs, toas, aero_res, pix_res):
    # gbm = lgb.Booster(model_file = file_path + '/emus/gbm_aot.txt')
    gbm = joblib.load(file_path + '/emus/lgb.pkl') 
    aots = []
    area = np.ceil(aero_res / pix_res).astype(int)                                          
    kern = np.ones((area, area)) / (area*area)                                              
    shape = auxs[0].priors[0].shape                                                         
    points = np.repeat(np.arange(shape[0]), shape[1]), np.tile(np.arange(shape[1]), shape[0])
    points = (np.array(points) * aero_res / pix_res).astype(int).T 
    choosen_bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # The bounds for training samples of AERONET observations
    bands_min = [0.11572497, 0.08986528, 0.07280412, 0.05007033, 0.06228712, 0.06849915, 0.07015634, 0.06554198, 0.06809415, 0.02089167, 0.00080242, 0.04392302, 0.03012728]
    bands_max = [0.32072931, 0.30801709, 0.32420026, 0.38214899, 0.3951169 , 0.41631785, 0.44391895, 0.42444658, 0.46161492, 0.19927658, 0.00870671, 0.39533241, 0.32505772]
    for i in range(len(toas)):
        toa = toas[i]
        aux = auxs[i]
        aero_refs = []
        mask = True
        for _, band in enumerate(choosen_bands):
            temp   = toa[band].copy()                                                                                                                                            
            temp[toa[band].mask] = np.nan
            ref = points_convolve(temp, kern, points)
            mask = mask & (ref >= bands_min[_]) & (ref <= bands_max[_])
            aero_refs.append(ref) 
        mask = (mask & np.isfinite(aero_refs).any(axis=0))                  
        xx = np.vstack([np.array(aero_refs), np.array([aux.sza.ravel(), 
                        aux.vza[4].ravel(), aux.raa[4].ravel(), 
                        aux.priors[-1].ravel(), aux.ele.ravel()/10.])])
        #iX = np.vstack([np.array([b9, b8]).reshape(2,-1), cons.reshape(5, -1)])
        aot = np.exp(-1*gbm.predict(xx.T).astype(float))
        #aot = np.exp(-1*gbm.predict(xx.T).reshape(shape).astype(float))
        #aot[mask] = np.nan
        #if mask.sum()>3:
        #    aot_min, aot_median, aot_max = np.nanpercentile(aot[mask], [5, 50, 95])
        #    good_aot =  (aot >= aot_min) & (aot <= aot_max) 
        #    indx, indy = np.where(good_aot.reshape(shape))
            #myInterpolator = NearestNDInterpolator((indx, indy), aot[good_aot]) 
            #grid_x, grid_y = np.mgrid[0:shape[0]:1, 0:shape[1]:1,]
            #aot = myInterpolator(grid_x, grid_y)
            #aot = smoothn(aot, isrobust=True, TolZ=1e-6, W = 100*((aot >= aot_min) & (aot <=aot_max)), s=10, MaxIter=1000)[0]
        #else:
        aot = np.nanmedian(aot)*np.ones(shape)    
        aots.append(aot)
    return aots


def get_starting_tcwvs(auxs, toas, aero_res, pix_res):

    iso_tcwv = Two_NN(np_model_file = file_path + '/emus/S2_TCWV.npz')
    tcwvs = []
    area = np.ceil(aero_res / pix_res).astype(int)                                          
    kern = np.ones((area, area)) / (area*area)                                              
    shape = auxs[0].priors[0].shape                                                         
    points = np.repeat(np.arange(shape[0]), shape[1]), np.tile(np.arange(shape[1]), shape[0])
    points = (np.array(points) * aero_res / pix_res).astype(int).T 
    for i in range(len(toas)):
        toa = toas[i]
        aux = auxs[i]
        temp   = toa[9].copy()                                                                                                                                            
        temp[toa[9].mask] = np.nan
        b9   = points_convolve(temp, kern, points)
                                 
        temp   = toa[8].copy()   
        temp[toa[8].mask] = np.nan 
        b8   = points_convolve(temp, kern, points)
                                 
        xx = np.array([b9, b8, aux.sza.ravel(), aux.vza[4].ravel(), aux.raa[4].ravel(), aux.priors[-1].ravel(), aux.ele.ravel()/10.])
        #iX = np.vstack([np.array([b9, b8]).reshape(2,-1), cons.reshape(5, -1)])
        tcwv = iso_tcwv.predict(xx.T)[0].ravel()
        tcwvs.append(tcwv.reshape(shape).astype(float))
        
    return tcwvs


def cost_cost(p, s2_obs, l8_obs, s2_toas, s2_emus, s2_auxs, masks, l8_toas,l8_emus, l8_auxs, pix_res, aero_res, aero_shape,  gamma1, gamma2, alpha):
    
    band_weight = np.array([0.442, 0.469, 0.555, 0.645, 0.859, 1.64 , 2.13 ])**alpha 
    band_weight = band_weight
    bands   = [[0, 1, 2, 3, 8, 11, 12]] * len(s2_toas) + [[0, 1, 2, 3, 4, 5, 6]] * len(l8_toas)

    # aero_shape = s2_auxs[0].sza.shape
    # toa_shape  = s2_toas[0][0].shape
    # to_compute, fine_inds, coarse_inds, fine_xys = get_targets(s2_obs + l8_obs, masks, pix_res, aero_res, target_mask, aero_shape, toa_shape)

    xps = get_xps(p, s2_emus + l8_emus, s2_auxs + l8_auxs)
    smooth_J, smooth_dJ = smooth_cost(p, s2_auxs + l8_auxs, gamma1, gamma2)
    prior_J, prior_dJ   = prior_cost( p, s2_auxs + l8_auxs)
    fine_J, fine_dJ     = fine_cost(  xps, s2_toas + l8_toas, masks, band_weight, bands, aero_shape, aero_res, pix_res)
    coarse_J, coarse_dJ = coarse_cost(xps, s2_obs  + l8_obs,  band_weight[1:], aero_shape, aero_res)
    cost_mask = (fine_J==0) #& (coarse_J==0)
    prior_J [     cost_mask] = 0
    prior_dJ[:,:, cost_mask] = 0
    J  = coarse_J   + smooth_J   + prior_J + fine_J
    dJ = coarse_dJ  + smooth_dJ + prior_dJ + fine_dJ
    #J  = fine_J + prior_J
    #dJ = fine_dJ + prior_dJ
    return J.sum(), dJ.reshape(dJ.shape[0]*dJ.shape[1],-1).reshape(-1) 

def get_p0(s2_obs, l8_obs, s2_auxs, l8_auxs, s2_toas, aero_res, pix_res):
    p0 = [] 
    aots = get_starting_aots(s2_obs  + l8_obs, s2_auxs + l8_auxs, aero_res)
    if len(s2_auxs)>0:
        tcwvs = get_starting_tcwvs(s2_auxs, s2_toas, aero_res, pix_res)
    else:
        tcwvs = []
    for _, s2_aux in enumerate(s2_auxs): 

        tcwv = tcwvs[_]
        median = np.nanmedian(tcwv)
        if not ((median>0) & (median<8)):
            median = s2_aux.priors[1].mean()
        mask = (tcwv > 0) & (tcwv < 8)
        tcwv[~mask] = median

        # priors    = s2_aux.priors
        # priors[1] = tcwv
        # s2_aux    = s2_aux._replace(priors = priors)
        # s2_auxs[_] = s2_aux

        # aot = aots[_]
        # mask = (aot > 0) & (aot < 2.5)
        # if (not mask.any()) & (np.nanmax(aa)<=0):
        #     aot = np.ones_like(aot)*0.001
        # elif (not mask.any()) & (np.nanmax(aa)>=2.5):
        #     aot = np.ones_like(aot)*2.4       
        # elif not mask.any():
        #     aot = 
        # aot[~mask] = np.nan
        # median = np.nanmedian(aot)
        # aot[~mask] = median
        median = np.nanmedian(s2_aux.priors[0])
        if not ((median>0) & (median<2.5)):
            median = s2_aux.priors[0].mean()
        p0.append(np.ones_like(tcwv) * median) 
        p0.append(tcwv) 

    for _, l8_aux in enumerate(l8_auxs): 

        aot = aots[_ + len(s2_auxs)]
        mask = (aot > 0) & (aot < 2.5)
        aot[~mask] = np.nan
        median = np.nanmedian(aot)
        aot[~mask] = median 

        if not ((median>0) & (median<2.5)):
            median = l8_aux.priors[0].mean()

        p0.append(np.ones_like(aot) * median) 
        p0.append(l8_aux.priors[1]) 


    p0 = np.array(p0).ravel() 
    return p0

def grid_conversion(array, new_shape):
    #vmin, vmax = np.nanpercentile(array, (2.5, 97.5))
    #array[~((array < vmax) & (array > vmin))] = np.nanmedian(array)
    rs, cs = array.shape
    x, y   = np.arange(rs), np.arange(cs)
    kx = 3 if rs > 3 else 1                       
    ky = 3 if cs > 3 else 1
    if   (rs > 3 ) & (cs > 3):                    
        f      = interpolate.RectBivariateSpline(x, y, array, kx=kx, ky=ky, s=0)
        nx, ny = new_shape
        nx, ny = 1. * np.arange(nx) / nx * rs, 1. * np.arange(ny) / ny * cs
        znew   = f(nx, ny)
    else:
        znew = np.ones(new_shape) * array.mean()
    return znew

def sample_atmos(p, shape, new_shape):
    p_shape = p.shape
    new_p = np.zeros(p_shape[:2] + new_shape)
    for i in range(p_shape[0]):
        for j in range(p_shape[1]):
            new_p[i, j] = grid_conversion(p[i, j], new_shape)
    return new_p
        

def get_ranges(p_shape):
    vmin = np.zeros(p_shape)     
    vmax = np.zeros(p_shape)     
    aot_min_max  = [0.001, 2.5]
    tcwv_min_max = [0.001, 8] 
    for i in range(p_shape[0]):       
        #for j in range(p_shape[1]):   
        vmin[i, 0] = aot_min_max[0]
        vmax[i, 0] = aot_min_max[1]

        vmin[i, 1] = tcwv_min_max[0]
        vmax[i, 1] = tcwv_min_max[1]

    return np.array([vmin.ravel(), vmax.ravel()]).T

def get_correction_auxs(s2s, l8s, aero_res, dstSRS, outputBounds, dem, cams_dir, pix_res, s2_bands, l8_bands):
    s2_aux = namedtuple('s2_aux', 'sza vza raa priors prior_uncs ele')
    s2_auxs = []
    ele = read_ele(dem, aero_res, dstSRS, outputBounds)
    for s2 in s2s:
        saa, sza = read_ang(s2.sun_angs, aero_res, dstSRS, outputBounds)
        priors, prior_uncs = read_atmos_prior(cams_dir, aero_res, s2.obs_time, dstSRS, outputBounds) 
        vzas = []
        raas = []
        for band in s2_bands: #[0, 1, 2, 3, 8, 11, 12]:
            vaa, vza = read_ang(s2.view_angs[band], aero_res, dstSRS, outputBounds)
            raa      = vaa - saa
            vzas.append(np.cos(np.deg2rad(vza)))          
            raas.append(np.cos(np.deg2rad(raa)))                             
        s2_auxs.append(s2_aux(np.cos(np.deg2rad(sza)), vzas, raas, priors, prior_uncs, ele))

    l8_aux = namedtuple('l8_aux', 'sza vza raa priors prior_uncs ele')                     
    l8_auxs = [] 
    for l8 in l8s:                                                    
        saa, sza = read_ang(l8.sun_angs, aero_res, dstSRS, outputBounds)
        priors, prior_uncs = read_atmos_prior(cams_dir, aero_res, l8.obs_time, dstSRS, outputBounds) 
        vzas = []                                                     
        raas = []
        for band in l8_bands: #[0, 1, 2, 3, 4, 5, 6]:                          
            vaa, vza = read_ang(l8.view_angs[band], aero_res, dstSRS, outputBounds)
            raa      = vaa - saa
            vzas.append(np.cos(np.deg2rad(vza)))           
            raas.append(np.cos(np.deg2rad(raa)))                               
        l8_auxs.append(l8_aux(np.cos(np.deg2rad(sza)), vzas, raas, priors, prior_uncs, ele))
    return s2_auxs, l8_auxs

def do_correction_l8(toa, aux, aot, tcwv, emus, file_header, band_names, dstSRS, outputBounds, pix_res, aero_res):
    
    X          = [aux.sza.ravel(), np.array(aux.vza).reshape(len(toa), -1), 
                  np.array(aux.raa).reshape(len(toa), -1), aot.ravel(), tcwv.ravel(), aux.priors[2].ravel(), aux.ele.ravel()]
    xps        = predic_xps_all(X, emus[0])
    selx, sely = np.where(np.ones(toa[0].shape))
    pts            = (selx * pix_res / aero_res).astype(int), (sely * pix_res / aero_res).astype(int)
    inds, fine_ind = np.unique(pts, axis=1, return_inverse=True)
    
    xap_Hs, xbp_Hs, xcp_Hs, xap_dHs, xbp_dHs, xcp_dHs = xps
    xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH  = xap_Hs[:, fine_ind], xbp_Hs[:, fine_ind], xcp_Hs[:, fine_ind], \
                                                                xap_dHs[:, fine_ind], xbp_dHs[:, fine_ind], xcp_dHs[:, fine_ind]
    sur, dH = cal_sur(xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH, toa.data.reshape(len(toa), -1))
    fnames = [file_header + band + '.tif' for band in band_names]
    for _, fname in enumerate(fnames):
        ref = sur[_].reshape(toa[0].shape)
        ref = np.maximum((ref * 10000).astype(int), 0)
        g   = array_to_raster(fname, ref,  dstSRS, outputBounds[0], outputBounds[-1], pix_res, pix_res, gdal.GDT_UInt16)
        g.FlushCache()
    return sur, dH

def do_correction_s2(toa, aux, aot, tcwv, emus, file_header, band_names, dstSRS, outputBounds, pix_res, aero_res):
    
    X          = [aux.sza.ravel(), np.array(aux.vza).reshape(len(toa), -1), 
                  np.array(aux.raa).reshape(len(toa), -1), aot.ravel(), tcwv.ravel(), aux.priors[2].ravel(), aux.ele.ravel()]
    xps        = predic_xps_all(X, emus[0])
    xap_Hs, xbp_Hs, xcp_Hs, xap_dHs, xbp_dHs, xcp_dHs = xps
    pix_ress = [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20]
    xmax, ymax = aux.sza.shape
    surs = []
    for _, band in enumerate(band_names):
        toa_band = file_header + band + '.jp2'
        pix_res = pix_ress[_]
        data = gdal.Warp('', toa_band, format = 'MEM', outputBounds = outputBounds, 
                             dstSRS = dstSRS, xRes = pix_res, yRes = pix_res, resampleAlg = gdal.GRA_NearestNeighbour).ReadAsArray()/10000.
        selx, sely = np.where(np.ones(data.shape))
        fname = file_header + band + '.tif' 
        pts            = np.minimum((selx * pix_res / aero_res).astype(int), xmax-1), np.minimum((sely * pix_res / aero_res).astype(int), ymax-1)
        inds, fine_ind = np.unique(pts, axis=1, return_inverse=True)
        xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH  = xap_Hs[[_], fine_ind], xbp_Hs[[_], fine_ind], xcp_Hs[[_], fine_ind], \
                                                            xap_dHs[[_], fine_ind], xbp_dHs[[_], fine_ind], xcp_dHs[[_], fine_ind]
        sur, dH = cal_sur(xap_H,  xbp_H,  xcp_H,  xap_dH,  xbp_dH,  xcp_dH, data.reshape(1, -1))
        ref = sur.reshape(data.shape)
        ref = np.maximum((ref * 10000).astype(int), 0)
        g   = array_to_raster(fname, ref,  dstSRS, outputBounds[0], outputBounds[-1], pix_res, pix_res, gdal.GDT_UInt16)
        g.FlushCache()
        surs.append(sur)
    return surs, dH

def do_one_s2(fs, cams_dir, dem, mcd43_dir, mcd43_vrt_dir, jasmin):

    logger = create_logger()
    logger.propagate = False
    aoi, middle_file, file_index, s2_fnames, l8_fnames = fs
    site = aoi.split('/')[-1].split('_location')[0]
    f = np.load(file_path + '/sites_s2.npz', allow_pickle=True)
    sites_s2 = np.array(f.f.sites_s2)
    ind = sites_s2[:,0].tolist().index(site)
    s2_tiles = sites_s2[ind][1]

    temporal_window = 1
    pix_res = 30
    if 'MSIL1C' in middle_file:
        aoi = aoi.replace('/work/scratch/marcyin', '/data/nemesis')
        middle_file = middle_file.replace('/work/scratch/marcyin', '/data/nemesis')
        s2_files, l8_files = pre_processing([middle_file], [])
        dstSRS, outputBounds = get_bounds(aoi, s2_files[0][2][2], pix_res)
    else:
        s2_files, l8_files = pre_processing([], [middle_file])
        dstSRS, outputBounds = get_bounds(aoi, l8_files[0][2][2], pix_res)
    s2s, s2_times, l8s, l8_times = get_obs(s2_files, l8_files)
    jasmin_s2s = find_on_jasmin(s2_tiles, (s2s + l8s)[0].obs_time, outputBounds, dstSRS)
    s2_fs = [i.ID for i in jasmin_s2s]
    if middle_file.split('/')[-1] in s2_fs:
        ind = s2_fs.index(middle_file.split('/')[-1])
        del jasmin_s2s[ind]
    s2s += jasmin_s2s[:3]
    s2_times = [i.obs_time for i in s2s]
    obs_times = np.unique(s2_times + l8_times).tolist()
    vrt_dir = get_mcd43(aoi, obs_times, mcd43_dir, temporal_window = temporal_window , jasmin = False, vrt_dir=mcd43_vrt_dir)
    logger.info('Reading TOAs.')
    s2_toas, s2_surs, s2_uncs, s2_clds = read_s2(s2s, pix_res, dstSRS, outputBounds, vrt_dir, temporal_window = temporal_window )
    l8_toas, l8_surs, l8_uncs, l8_clds = read_l8(l8s, pix_res, dstSRS, outputBounds, vrt_dir, temporal_window = temporal_window )
    logger.info('Redo cloud mask.')
    s2_clouds, s2_shadows,s2_changes, l8_clouds, l8_shadows, l8_changes = redo_cloud_shadow(s2_toas, l8_toas, s2_clds, l8_clds)
    if 'MSIL1C' in middle_file:
        sur_x, sur_y = s2_surs[0][0].shape
    else:
        sur_x, sur_y = l8_surs[0][0].shape
    logger.info('Getting optimal PSF parameters.')  
    possible_x_y, struct, gaus, pointXs, pointYs = cal_psf_points(pix_res, sur_x, sur_y)
    s2_obs = do_s2_psf(s2s, s2_toas, s2_clouds, s2_shadows, s2_surs, possible_x_y, struct, gaus, pointXs, pointYs) 
    l8_obs = do_l8_psf(l8s, l8_toas, l8_clouds, l8_shadows, l8_surs, possible_x_y, struct, gaus, pointXs, pointYs)

    ret = []
    old_shape = None
    logger.info('MultiGrid solver in process...')
    s2_bands, l8_bands = [0, 1, 2, 3, 8, 11, 12], [0, 1, 2, 3, 4, 5, 6]
    s2_emus, l8_emus = load_emus(s2s, l8s, s2_bands, l8_bands)
    # [10000, 5000, 2500, 1000, 500, 240, 120]
    for _, aero_res in enumerate([4000, 2000, 1000, 500]):
        logger.info(bcolors.BLUE + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'+bcolors.ENDC)
        logger.info(bcolors.RED + 'Optimizing at resolution %d m' % (aero_res) + bcolors.ENDC)
        #aero_res = 250
        s2_auxs, l8_auxs = prepare_aux(s2s, l8s, aero_res, dstSRS, outputBounds, dem, cams_dir, s2_toas, pix_res, s2_bands, l8_bands)
        total_obs        = len(s2s) + len(l8s)
        obs_nums         = total_obs - np.array(s2_clouds + l8_clouds + s2_shadows + l8_shadows).astype(int).sum(axis=0)
        stable_targets   = get_stable_targets(s2_changes + l8_changes, s2_clouds + l8_clouds, s2_shadows + l8_shadows, s2_toas + l8_toas)
        target_mask      = (obs_nums >= 0.5*total_obs ) & (np.array(stable_targets).astype(int).sum(axis=0)>1)
        masks = combine_mask(target_mask, stable_targets) 
        if 'MSIL1C' in middle_file:
            new_shape = s2_auxs[0].ele.shape
        else:
            new_shape = l8_auxs[0].ele.shape
        if _ ==0:
            p0 = get_p0(s2_obs, l8_obs, s2_auxs, l8_auxs, s2_toas, aero_res, pix_res)
            p0_aot  = p0.reshape(len(s2_obs) + len(l8_obs), 2, -1)[:,0].mean(axis=1)
            p0_tcwv = p0.reshape(len(s2_obs) + len(l8_obs), 2, -1)[:,1].mean(axis=1)
            p0_aot  = ['%.02f'%i for i in p0_aot]
            p0_tcwv = ['%.02f'%i for i in p0_tcwv]
            logger.info('Starting mean AOTs : %s'%(', '.join(p0_aot)))
            logger.info('Starting mean TCWVs: %s'%(', '.join(p0_tcwv)))
        else:
            pp = ret[-1]
            p0 = sample_atmos(pp, old_shape, new_shape) 
            p0_aot  = p0.reshape(len(s2_obs) + len(l8_obs), 2, -1)[:,0].mean(axis=1)
            p0_tcwv = p0.reshape(len(s2_obs) + len(l8_obs), 2, -1)[:,1].mean(axis=1)
            p0_aot  = ['%.02f'%i for i in p0_aot]
            p0_tcwv = ['%.02f'%i for i in p0_tcwv]
            logger.info('Starting mean AOTs : %s'%(', '.join(p0_aot)))
            logger.info('Starting mean TCWVs: %s'%(', '.join(p0_tcwv))) 
            # for np.isnan(p0).any():
            #     p0 = get_p0(s2_obs, l8_obs, s2_auxs, l8_auxs, s2_toas, aero_res, pix_res)

            #p0[:len(s2_auxs),1] = np.array(tcwvs)
        if 'MSIL1C' in middle_file:
            aero_shape = s2_auxs[0].sza.shape
            toa_shape  = s2_toas[0][0].shape
        else:
            aero_shape = l8_auxs[0].sza.shape
            toa_shape  = l8_toas[0][0].shape
        #to_compute, fine_inds, coarse_inds, fine_xys = get_targets(s2_obs + l8_obs, masks, pix_res, aero_res, target_mask, aero_shape, toa_shape)
        bounds = get_ranges((total_obs, 2) + new_shape)
        psolve = optimize.minimize(cost_cost, p0.ravel(), jac=True, method='L-BFGS-B', bounds = bounds, 
                                options={'disp': 1, 'gtol': 1e-04, 'maxfun': 100, 'maxiter': 100, 'ftol': 1e-8, 'maxcor': 100}, 
                                args =(s2_obs, l8_obs, s2_toas, s2_emus,s2_auxs, masks, l8_toas,l8_emus, l8_auxs, pix_res, aero_res, aero_shape,  30, 10, -1.6))
        ret.append(psolve.x.reshape((total_obs, 2) + new_shape))
        old_shape = new_shape 
        
        logger.info(bcolors.GREEN + str(psolve['message'].decode()) + bcolors.ENDC)
        logger.info(bcolors.GREEN + 'Iterations: %d'%int(psolve['nit']) + bcolors.ENDC)
        logger.info(bcolors.GREEN + 'Function calls: %d'%int(psolve['nfev']) +bcolors.ENDC)  

    optimized_paras = ret[-1]
    s2_bands, l8_bands = range(13), range(7)
 
    if 'MSIL1C' in middle_file:
        para_index   = file_index
        example_file = s2s[file_index].toa[0] 
        file_header  = example_file.replace('B01.jp2', '')
        s2_auxs, _   = get_correction_auxs([s2s[file_index]], [], aero_res, dstSRS, outputBounds, dem, cams_dir,  pix_res, s2_bands, [])
        emus, _      = load_emus([s2s[file_index]], [], s2_bands, [])
        toa          = s2_toas[file_index]
        aux          = s2_auxs[0]
        cloud        = s2_clouds[file_index]
        shadow       = s2_shadows[file_index]
        logger.info('Atmospheric parameters retried and saving into local files...')
        cloud_shadow = cloud * 1. + shadow + 2.
        posterious_aot, posterious_tcwv = optimized_paras[para_index]
        
        aot_fname, tcwv_fname = file_header + 'aot.tif', file_header + 'tcwv.tif'
        cloud_fname           = file_header + 'cloud_shadow.tif'

        aot_g   = array_to_raster(aot_fname,   posterious_aot,  dstSRS, outputBounds[0], outputBounds[-1],  aero_res, aero_res, gdal.GDT_Float32)
        tcwv_g  = array_to_raster(tcwv_fname,  posterious_tcwv, dstSRS, outputBounds[0], outputBounds[-1],  aero_res, aero_res, gdal.GDT_Float32)
        cloud_g = array_to_raster(cloud_fname, cloud_shadow,    dstSRS, outputBounds[0], outputBounds[-1],  aero_res, aero_res, gdal.GDT_Byte)
        aot_g.FlushCache()                           
        tcwv_g.FlushCache()                           
        cloud_g.FlushCache()
        
        band_names   = np.array(['B01', 'B02', 'B03','B04','B05' ,'B06', 'B07', 'B08','B8A', 'B09', 'B10', 'B11', 'B12'])
        logger.info('Doing correction...')
        sur, dH = do_correction_s2(toa, aux, posterious_aot, posterious_tcwv, emus, file_header, band_names, dstSRS, outputBounds, pix_res, aero_res)
        
    else:
        para_index   = file_index + len(s2s)
        example_file = l8s[file_index].toa[0] 
        file_header  = example_file.replace('B1.TIF', '')
        _, l8_auxs   = get_correction_auxs([], [l8s[file_index]], aero_res, dstSRS, outputBounds, dem, cams_dir,  pix_res, [], l8_bands)
        _, emus      = load_emus([], [l8s[file_index]], [], l8_bands)
        toa          = l8_toas[file_index]
        aux          = l8_auxs[0]
        cloud        = l8_clouds[file_index]
        shadow       = l8_shadows[file_index]
        logger.info('Atmospheric parameters retried and saving into local files...')
        cloud_shadow = cloud * 1. + shadow + 2.
        posterious_aot, posterious_tcwv = optimized_paras[para_index]
        
        aot_fname, tcwv_fname = file_header + 'aot.tif', file_header + 'tcwv.tif'
        cloud_fname           = file_header + 'cloud_shadow.tif'

        aot_g   = array_to_raster(aot_fname,   posterious_aot,  dstSRS, outputBounds[0], outputBounds[-1],  aero_res, aero_res, gdal.GDT_Float32)
        tcwv_g  = array_to_raster(tcwv_fname,  posterious_tcwv, dstSRS, outputBounds[0], outputBounds[-1],  aero_res, aero_res, gdal.GDT_Float32)
        cloud_g = array_to_raster(cloud_fname, cloud_shadow,    dstSRS, outputBounds[0], outputBounds[-1],  aero_res, aero_res, gdal.GDT_Byte)
        aot_g.FlushCache()                           
        tcwv_g.FlushCache()                           
        cloud_g.FlushCache()
        band_names   = np.array(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
        logger.info('Doing correction...')
        sur, dH = do_correction_l8(toa, aux, posterious_aot, posterious_tcwv, emus, file_header, band_names, dstSRS, outputBounds, pix_res, aero_res)
    
    shutil.rmtree(vrt_dir)
    for i in s2s + l8s:
        try:
            ang_dir = os.path.dirname(os.path.dirname(i.sun_angs))
            shutil.rmtree(ang_dir)
        except:
            pass
    logger.info('Done!')
    return ret[-1], sur, dH

def array_to_raster(fname, array, dstSRS, xMin, yMax, xRes, yRes, dtype):    
    if array.ndim == 2:                             
        bands = 1                                   
    elif array.ndim ==3:                            
        bands = min(array.shape)                    
        t = np.argsort(array.shape)                 
        array = array.transpose(t)                  
    else:                                           
        raise IOError('Only 2 or 3 D array is supported.')    
    if os.path.exists(fname):
        os.remove(fname)                                                              
    driver = gdal.GetDriverByName('GTiff')          
    ds = driver.Create(fname, array.shape[-1], array.shape[-2], bands, dtype)                       
    ds.SetProjection(dstSRS)             
    geotransform    = [0., 0., 0., 0., 0., 0.]
    geotransform[0] = xMin
    geotransform[3] = yMax
    geotransform[1] =    xRes                                 
    geotransform[5] = -1*yRes                                                                                                          
    ds.SetGeoTransform(geotransform)                
    if array.ndim == 3:                             
        for i in range(bands):                      
            ds.GetRasterBand(i+1).WriteArray(array[i])                                                         
    else:                                           
         ds.GetRasterBand(1).WriteArray(array)                                                                                                           
    ds                                              
    return ds      
acix_2        = '/work/scratch/marcyin/acix_2/'
sites = [i.split('/')[-2] for i in glob(acix_2 + '/S2s_ground_30/*/')] 


cams_dir      = '/work/scratch/marcyin/CAMS/'
dem           = '/work/scratch/marcyin/DEM/global_dem.vrt'
mcd43_dir     = '/work/scratch/marcyin/MCD43/'
mcd43_vrt_dir = '/work/scratch/marcyin/MCD43_VRT/'
acix_2        = '/data/nemesis/acix_2'
cams_dir      = '/vsicurl/http://www2.geog.ucl.ac.uk/~ucfafyi/cams/'
dem           = '/vsicurl/http://www2.geog.ucl.ac.uk/~ucfafyi/eles/global_dem.vrt'
mcd43_dir     = '/home/ucfafyi/hep/MCD43/'
mcd43_vrt_dir = '/home/ucfafyi/MCD43_VRT/'
jasmin = True
#fss = find_around_files(sites[1])
# for fs in fss[-5:]:
#     ret = do_one_s2(fs, cams_dir, dem, mcd43_dir, mcd43_vrt_dir)
