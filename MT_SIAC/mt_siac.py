import os
import sys
import osr
import ogr
import mgrs
import gdal
import psutil
import requests
import datetime
import kernels
from io import StringIO
import pandas as pd
import numpy as np
from glob import glob
from functools import partial
from Get_S2 import get_s2_files
from get_MCD43 import get_mcd43
from multiprocessing import Pool
from scipy import ndimage, signal
from scipy.stats import linregress
from shapely.geometry import Point
from collections import namedtuple
from s2_preprocessing import s2_pre_processing
from l8_preprocessing import l8_pre_processing
from scipy.interpolate import NearestNDInterpolator
from skimage.morphology import disk, binary_dilation, binary_closing
procs =  psutil.cpu_count()
file_path = os.path.dirname(os.path.realpath(__file__))

sites = [i.split('/')[-2] for i in glob('/data/nemesis/acix_2/S2s_30/*/')] 

def do_one_site(site):
    s2_fnames = glob('/data/nemesis/acix_2/S2s_30/%s/*'%site)
    l8_fnames = glob('/data/nemesis/acix_2/L8s_30_LZW/%s/*'%site)
    #aoi = glob('/data/nemesis/acix_2/%s_location.json'%site)[0]
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
        fs = glob('/data/nemesis/acix_2/S2s_30/*/*' + tile + '*.SAFE/GR*/*/IMG_DATA/*B02.jp2')
        if len(fs) <= 0:
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
        ret = s2_pre_processing(s2_fname)    
        s2_files.append(ret[0])
    for l8_fname in l8_fnames:
        ret = l8_pre_processing(l8_fname)    
        l8_files.append(ret[0])
    return s2_files, l8_files

def _find_around_files(site):                                      
    f = np.load('S2_files_to_be_down.npz')
    fs = [i[0] for i in f.f.rets]
    s2_files, l8_files, s2_times, l8_times = do_one_site(site)
    res = []           
    tiles = np.unique([i.split('_')[-2] for i in s2_files]).tolist()
    for i in s2_times + l8_times:
        all_time  = [i+ datetime.timedelta(days=ii) for ii in range(-16, 16)]
        need_files = [j for j in fs if (datetime.datetime.strptime(j[11:19], '%Y%m%d') in all_time) & (j[38:44] in tiles)]
        res += need_files   
    return np.unique(res).tolist()


def find_around_files(site):
    s2_files, l8_files, s2_times, l8_times = do_one_site(site)
    aoi = glob('/data/nemesis/acix_2/%s_location.json'%site)[0] 
    res = [] 
    for i in s2_times: 
        s2_fnames = [] 
        l8_fnames = []
        for j in s2_times: 
            if abs((j - i).days)<16: 
                s2_fnames.append(s2_files[s2_times.index(j)]) 
        for k in l8_times: 
            if abs((k - i).days)<16: 
                l8_fnames.append(l8_files[l8_times.index(k)]) 
        res.append([aoi, s2_files[s2_times.index(i)], s2_fnames, l8_fnames])
    return res

def get_kk(angles):                                                                                                                                                                            
    vza ,sza,raa = angles
    kk = kernels.Kernels(vza ,sza,raa,\
                         RossHS=False,MODISSPARSE=True,\
                         RecipFlag=True,normalise=1,\
                         doIntegrals=False,LiType='Sparse',RossType='Thick')
    return kk



def warp_data(fname, outputBounds,  xRes, yRes,  dstSRS):
    g = gdal.Warp('',fname, format = 'MEM',  dstSRS = dstSRS, resampleAlg = 0, 
                  outputBounds=outputBounds, xRes = xRes, yRes = yRes, outputType = gdal.GDT_Int16) 
    return g.ReadAsArray() 

def read_MCD43(fnames, dstSRS, outputBounds):
    par = partial(warp_data, outputBounds = outputBounds, xRes = 500, yRes = 500, dstSRS = dstSRS) 
    p = Pool()
    p = Pool(procs)
    ret = p.map(par,  fnames)
    #ret  =list( map(par,  view_ang_name_gmls))
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
    if temporal_filling == True:
        temporal_filling = 16
    if temporal_filling:
        days   = [(obs_time - datetime.timedelta(days = int(i))) for i in np.arange(temporal_filling, 0, -1)] + \
                 [(obs_time + datetime.timedelta(days = int(i))) for i in np.arange(0, temporal_filling+1,  1)]
        fnames = []
        for temp in [da_temp, qa_temp]:                                                                                    
            for day in days:
                for band in boa_bands:
                    fname = mcd43_dir + '/'.join([day.strftime('%Y-%m-%d'), temp%(day.strftime('%Y%j'), band)])
                    fnames.append(fname) 
    else:
        fnames = []
        for temp in [da_temp, qa_temp]:
            for band in self.boa_bands:
                    fname = MCD43_dir + '/'.join([datetime.strftime(obs_time, '%Y-%m-%d'), temp%(doy, band)])
                    fnames.append(fname)
    das, ws = read_MCD43(fnames, dstSRS, outputBounds) 
    return das, ws


def smooth(da_w):
    da, w = da_w
    if (da.shape[-1]==0) | (w.shape[-1]==0):
        return da, w
    data  = np.array(smoothn(da, s=10., smoothOrder=1., axis=0, TolZ=0.001, verbose=False, isrobust=True, W = w))[[0, 3],]                                                                                                                          
    return data[0], data[1]


ret = find_around_files(sites[1])
fs = ret[0]

def read_l8(l8s, pix_res, dstSRS, outputBounds, vrt_dir):
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
        das, qas = get_boa(vrt_dir, boa_bands, i.obs_time, outputBounds, dstSRS, temporal_filling = 4)
        saa, sza = gdal.Warp('', i.sun_angs, format = 'MEM', outputBounds = outputBounds,
                             xRes = 500, yRes = 500, dstSRS = dstSRS, dstNodata=-32767, outputType = gdal.GDT_Int16).ReadAsArray() / 100
        new_shp  = saa.shape                 
        new_xys  = np.where(np.ones_like(sza))
        f_interp = NearestNDInterpolator(np.where(sza>=0), sza[sza>=0])
        sza = f_interp(new_xys).reshape(new_shp)
        f_interp = NearestNDInterpolator(np.where(saa>-180), saa[saa>-180]) 
        saa = f_interp(new_xys).reshape(new_shp)
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

def read_s2(s2s, pix_res, dstSRS, outputBounds, vrt_dir):
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
        das, qas = get_boa(vrt_dir, boa_bands, i.obs_time, outputBounds, dstSRS, temporal_filling = 4)
        saa, sza = gdal.Warp('', i.sun_angs, format = 'MEM', outputBounds = outputBounds, 
                             xRes = 500, yRes = 500, dstSRS = dstSRS, dstNodata=-32767, outputType = gdal.GDT_Int16).ReadAsArray() / 100
        new_shp  = saa.shape                    
        new_xys  = np.where(np.ones_like(sza))  
        f_interp = NearestNDInterpolator(np.where(sza>=0), sza[sza>=0]) 
        sza = f_interp(new_xys).reshape(new_shp)
        f_interp = NearestNDInterpolator(np.where(saa>-180), saa[saa>-180]) 
        saa = f_interp(new_xys).reshape(new_shp)
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
    month_obs = np.ma.vstack([np.ma.array(s2_toas)[:,[1, 2, 3, 8, 11, 12]], np.ma.array(l8_toas)[:,[1, 2, 3, 4, 5, 6]]])
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
        certain_cloud = binary_closing(certain_cloud)
        certain_cloud[vis_diff.mask] = (s2_clds[_]>40)[vis_diff.mask]
        nir_diff = diff[3:].mean(axis=0)        
        shadow   = nir_diff < -0.1              
        shadow   = binary_closing(shadow) & (~certain_cloud)
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
        certain_cloud  = binary_closing(certain_cloud)
        certain_cloud[vis_diff.mask] = (((l8_clds[_] >> 5) & 3) > 0)[vis_diff.mask] 
     
        nir_diff       = diff[3:].mean(axis=0)  
        certain_shadow = (((l8_clds[_] >> 7) & 3) > 1) & (nir_diff < -0.1)
        certain_shadow = binary_dilation(certain_shadow, selem = struct).astype(certain_cloud.dtype) & (nir_diff < -0.025)
        certain_shadow = binary_closing(certain_shadow) & (~certain_cloud)
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

from numba import jit

#@jit()
def cost(p, toa, sur, pxs, pys, pcxs, pcys, gaus):
    xshift, yshift = p
    point_xs     = pxs + xshift
    point_ys     = pys + yshift
    mask         = (point_xs<toa.shape[0]) & (point_xs>0) & (point_ys<toa.shape[1]) & (point_ys>0)
    point_xs     = point_xs[mask]
    point_ys     = point_ys[mask]
    points       = np.array([np.repeat(point_xs, len(point_ys)), np.tile(point_ys, len(point_xs))]).T
    mask         = (points[:,0]<toa.shape[0]) & (points[:,0]>0) & (points[:,1]<toa.shape[1]) & (points[:,1]>0) 
    conv_toa     = points_convolve(toa, gaus, points)
    mask         = mask & (conv_toa>=0.001)
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
    padx = int(np.ceil(kernel.shape[0]/2.))
    pady = int(np.ceil(kernel.shape[1]/2.)) 
    for _ in range(len(points)): 
        x, y    = points[_]
        batch   = data[x - kx + padx: x + kx + padx, y - ky + pady: y + ky + pady] 
        rets[_] = np.nansum(batch[:kernel.shape[0],:kernel.shape[1]]*kernel)
    return rets

@jit(nopython=True)
def points_convolve(im, kernel, points): 
    rows, cols     = im.shape
    k_rows, k_cols = kernel.shape
    padx = int(k_rows/2.)
    pady = int(k_cols/2.)
    data = np.zeros((rows + 2*k_rows, cols + 2*k_cols))
    #data = np.pad(im, (2*padx, 2*pady), mode='reflect') 
    data[k_rows: rows + k_rows, k_cols: cols + k_cols] = im
    return convolve(data, kernel, points) 

def do_s2_psf(s2_toas, s2_clouds, s2_shadows, s2_surs, possible_x_y, struct, gaus, pointXs, pointYs):
    s2_conv_toas = []                    
    s2_cors_surs = []
    s2_cors_pots = []
    for _, s2_toa in enumerate(s2_toas):      
        data       = s2_toa[12].data.copy()   
        border     = np.ones_like(data).astype(bool)
        border[1:-1, 1:-1] = False       
        bad_pix    = s2_clouds[_] | s2_shadows[_] | border
        bad_pix    = ndimage.binary_dilation(bad_pix, structure = struct, iterations=10)
        good_xy    = (~bad_pix)[pointXs, pointYs]
        p_xs, p_ys = pointXs[good_xy], pointYs[good_xy]
        p_corse_xs, p_corse_ys = np.array(range(60))[good_xy], np.array(range(60))[good_xy]
        p_corse_xs, p_corse_ys = np.repeat(p_corse_xs, len(p_corse_ys)), np.tile(p_corse_ys, len(p_corse_xs))
        par = partial(cost, toa = data, sur = s2_surs[_][-1], pxs = p_xs, pys = p_ys, pcxs = p_corse_xs, pcys = p_corse_ys, gaus = gaus)
        p   = Pool()                     
        ret = p.map(par, possible_x_y)        
        p.close()                        
        p.join()                         
        mind = np.argmin(np.array(ret, dtype=np.object)[:,0])
        shift_x, shift_y   = possible_x_y[mind]
        un_r, points, mask = ret[mind]   
        s2_conv_toa = []          
        s2_cors_sur = []       
        for band in [1, 2, 3, 8, 11, 12]:                                                                                                                                 
            data = s2_toa[band].data.copy()   
            data[s2_clouds[_]] = np.nan       
            data[s2_shadows[_]] = np.nan      
            conv_toa = points_convolve(data, gaus, points)
            conv_toa[conv_toa<=0.001] = np.nan
            s2_conv_toa.append(conv_toa)      
        s2_cors_sur = np.array(s2_surs[0])[:, p_corse_xs[mask], p_corse_ys[mask]]
        
        s2_conv_toas.append(s2_conv_toa) 
        s2_cors_surs.append(s2_cors_sur)
        s2_cors_pots.append(np.array([p_corse_xs[mask], p_corse_ys[mask]]))

    for _, s2_toa in enumerate(s2s):
        sensor = s2_toa[0].split('_MSIL1C_')[0][-3:]
        mask, s2_cors_sur = spectral_mapping(s2_cors_surs[_], s2_conv_toas[_], sensor)
        s2_cors_surs[_]   = np.array(s2_cors_sur)[:, mask]
        s2_cors_pots[_]   = np.array(s2_cors_pots[_])[:, mask]
        s2_conv_toas[_]   = np.array(s2_conv_toas[_])[:, mask]

    return s2_conv_toas, s2_cors_surs, s2_cors_pots

def do_l8_pdf(l8_toas, l8_clouds, l8_shadows, l8_surs, possible_x_y, struct, gaus, pointXs, pointYs):    
    l8_conv_toas = []                    
    l8_cors_surs = []
    l8_cors_pots = []
    for _, l8_toa in enumerate(l8_toas):      
        data       = l8_toa[6].data.copy()   
        border     = np.ones_like(data).astype(bool)
        border[1:-1, 1:-1] = False       
        bad_pix    = l8_clouds[_] | l8_shadows[_] | border
        bad_pix    = ndimage.binary_dilation(bad_pix, structure = struct, iterations=10)
        good_xy    = (~bad_pix)[pointXs, pointYs]
        p_xs, p_ys = pointXs[good_xy], pointYs[good_xy]
        p_corse_xs, p_corse_ys = np.array(range(60))[good_xy], np.array(range(60))[good_xy]
        p_corse_xs, p_corse_ys = np.repeat(p_corse_xs, len(p_corse_ys)), np.tile(p_corse_ys, len(p_corse_xs))
        par = partial(cost, toa = data, sur = s2_surs[_][-1], pxs = p_xs, pys = p_ys, pcxs = p_corse_xs, pcys = p_corse_ys, gaus = gaus)
        p   = Pool()                     
        ret = p.map(par, possible_x_y)        
        p.close()                        
        p.join()                         
        mind = np.argmin(np.array(ret, dtype=np.object)[:,0])
        shift_x, shift_y   = possible_x_y[mind]
        un_r, points, mask = ret[mind]   
        l8_conv_toa = []          
        for band in [1, 2, 3, 4, 5, 6]:                                                                                                                                 
            data = l8_toa[band].data.copy()   
            data[l8_clouds[_]] = np.nan       
            data[l8_shadows[_]] = np.nan      
            conv_toa = points_convolve(data, gaus, points)
            conv_toa[conv_toa<=0.001] = np.nan
            l8_conv_toa.append(conv_toa)      
        l8_cors_sur = np.array(l8_surs[0])[:, p_corse_xs[mask], p_corse_ys[mask]]
    
        l8_conv_toas.append(l8_conv_toa) 
        l8_cors_surs.append(l8_cors_sur)
        l8_cors_pots.append(np.array([p_corse_xs[mask], p_corse_ys[mask]]))

    for _, l8_toa in enumerate(l8s):
        sensor = 'L8'
        mask, l8_cors_sur = spectral_mapping(l8_cors_surs[_], l8_conv_toas[_], sensor)
        l8_cors_surs[_]   = np.array(l8_cors_sur)[:, mask]
        l8_cors_pots[_]   = np.array(l8_cors_pots[_])[:, mask]
        l8_conv_toas[_]   = np.array(l8_conv_toas[_])[:, mask]

    return l8_conv_toas, l8_cors_surs, l8_cors_pots

def cal_psf_points(pix_res, sur_x, sur_y):
    xstd, ystd   = 260/pix_res, 340/pix_res
    pointXs     = ((np.arange(sur_x) * 500) // pix_res ) 
    pointYs     = ((np.arange(sur_y) * 500) // pix_res ) 
    gaus         = gaussian(xstd, ystd, norm = True)
 
    possible_x_y = [(np.arange(0,50), np.arange(0,50) +i) for i in range(-10, 10)]
    possible_x_y = np.array(possible_x_y).transpose(1,0,2).reshape(2, -1).T
    struct       = disk(5) 
    return possible_x_y, struct, gaus, pointXs, pointYs

def spectral_mapping(sur, toa, sensor):
    pmins = [[ 0.81793009, -1.55666629,  0.03879234,  0.02664923],
             [ 0.50218134, -0.94398654, -0.36284911,  0.02876391],
             [ 0.61609484, -1.12717424, -0.24037129,  0.0239488 ],
             [ 0.67499803, -1.1988073 , -0.18331019,  0.02179141],
             [ 0.23458873, -0.4048219 , -0.56692888,  0.02484466],
             [ 0.08220874, -0.13492051, -0.74972003, -0.0331204 ]]
    pmaxs = [[-0.76916621,  1.8524333 , -1.43464388,  0.34984857],
             [-0.91464915,  1.96174322, -1.38302832,  0.28090987],
             [-0.9199249 ,  1.9681306 , -1.3704881 ,  0.28924671],
             [-0.87389258,  1.89261443, -1.30929285,  0.28807412],
             [-0.71647392,  1.34657557, -0.79536697,  0.13551599],
             [-0.34076349,  0.60544841, -0.34178543,  0.09669959]]
    spec_map     =  Two_NN(np_model_file=file_path+'/spectral_mapping/Aqua_%s_spectral_mapping.npz'%sensor)
    sur     = np.array(spec_map.predict(sur.T)).squeeze()
    mask = True                                 
    if sur.shape[1] > 3: 
        for i in range(len(toa)):           
            pmin = np.poly1d(pmins[i])
            pmax = np.poly1d(pmaxs[i])
            diff = toa[i] - sur[i]
            mas  = (diff >= pmin(sur[i])) & (diff <= pmax(sur[i]))
            mmin, mmax = np.percentile(toa[i][mas] - sur[i][mas], [5, 95])
            mas  = mas & (diff >= mmin) & (diff <= mmax)
            mask = mask & mas
    else:        
        mask = np.zeros(sur.shape[1]).astype(bool)
    return mask, sur

def get_stable_targets(changes, clouds, shadows):
    stable_targets = []
    for i in range(len(changes)):
        change = np.mean([abs(changes[i][-1]).data + abs(changes[i][-2].data)], axis=0)
        stable_target = (change < 0.015) & (~clouds[i]) & (~shadows[i]) 
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

def read_atmos_piro(cams_dir, pix_res, obs_time, dstSRS, outputBounds):
    time_ind    = np.abs((obs_time.hour  + obs_time.minute/60. + obs_time.second/3600.) - np.arange(0,25,3)).argmin()
    cams_names  = ['aod550', 'tcwv', 'gtco3'] 
    prior_uncs = [0.4, 0.1, 0.05]
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
        priors.append(data * prior_scales[i])
        #prior_uncs.append(np.ones_like(data)*prior_uncs[i])
    return priors, prior_uncs

def read_ang(ang, pix_res, dstSRS, outputBounds):
    g = gdal.Warp('', ang, format = 'MEM', outputBounds = outputBounds,
                             dstSRS = dstSRS, xRes = pix_res, yRes = pix_res, resampleAlg = 0)
    angs = g.ReadAsArray() / 100.  
    return angs

def prepare_aux(s2s, l8s, pix_res, obs_time, dstSRS, outputBounds, dem, cams_dir):
    s2_aux = namedtuple('s2_aux', 'sza vza raa priors prior_uncs ele')
    s2_auxs = []
    ele = read_ele(dem, pix_res, dstSRS, outputBounds)
    for s2 in s2s:
        saa, sza = read_ang(s2.sun_angs, pix_res, dstSRS, outputBounds)
        priors, prior_uncs = read_atmos_piro(cams_dir, pix_res, s2.obs_time, dstSRS, outputBounds) 
        vzas = []
        raas = []
        for band in [0, 1, 2, 3, 8, 11, 12]:
            vaa, vza = read_ang(s2.view_angs[band], pix_res, dstSRS, outputBounds)
            raa      = vaa - saa
            vzas.append(np.cos(np.deg2rad(vza)))          
            raas.append(np.cos(np.deg2rad(raa)))                             
        s2_auxs.append(s2_aux(np.cos(np.deg2rad(sza)), vzas, raas, priors, prior_uncs, ele))
        
    l8_aux = namedtuple('l8_aux', 'sza vza raa priors prior_uncs ele')                     
    l8_auxs = [] 
    for l8 in l8s:                                                    
        saa, sza = read_ang(l8.sun_angs, pix_res, dstSRS, outputBounds)
        priors, prior_uncs = read_atmos_piro(cams_dir, pix_res, l8.obs_time, dstSRS, outputBounds) 
        vzas = []                                                     
        raas = []
        for band in [0, 1, 2, 3, 4, 5, 6]:                          
            vaa, vza = read_ang(l8.view_angs[band], pix_res, dstSRS, outputBounds)
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


def load_emus(s2s, l8s):
    s2_toa_bands = ['B01', 'B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
    s2_emus      = []    
    for _, s2_toa in enumerate(s2s):
        satellite = s2_toa[0].split('_MSIL1C_')[0][-3:]
        s2_emu = read_xa_xb_xc('MSI', satellite, s2_toa_bands)
        s2_emus.append(s2_emu)

    l8_toa_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    l8_emus = [] 
    l8_emu  = read_xa_xb_xc('OLI', 'L8', l8_toa_bands)
    for _, l8_toa in enumerate(l8s):
        l8_emus.append(l8_emu)
    return s2_emus, l8_emus
    

def do_one_s2(fs):

    aoi, s2_file, s2_fnames, l8_fnames = fs
    s2_files, l8_files = pre_processing(s2_fnames, l8_fnames)
    s2_obs = namedtuple('s2_obs', 'sun_angs view_angs toa cloud meta obs_time') 
    l8_obs = namedtuple('l8_obs', 'sun_angs view_angs toa cloud meta obs_time') 
    s2s = []
    l8s = []
    s2_times = []
    l8_times = []

    pix_res = 30
    dstSRS, outputBounds = get_bounds(aoi, s2_files[0][2][2], pix_res)

    for i in s2_files:
        obs_time = datetime.datetime.strptime(i[0].split('_MSIL1C_')[1][:15], '%Y%m%dT%H%M%S')
        s2_times.append(obs_time)
        dat = np.array(i, dtype = np.object)[[0,1,2,3,-1]].tolist()
        s2s.append(s2_obs(dat[0], dat[1], dat[2], dat[3], dat[4], obs_time))
    for i in l8_files:
        with open(i[-1]) as f:              
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

    obs_times = np.unique(s2_times + l8_times).tolist()
    vrt_dir = get_mcd43(aoi, obs_times, '/home/ucfafyi/hep/MCD43/', temporal_window = 4, jasmin = True, vrt_dir='/home/ucfafyi/MCD43_VRT/')
    
    s2_toas, s2_surs, s2_uncs, s2_clds = read_s2(s2s, pix_res, dstSRS, outputBounds, vrt_dir)
    l8_toas, l8_surs, l8_uncs, l8_clds = read_l8(l8s, pix_res, dstSRS, outputBounds, vrt_dir)
    s2_clouds, s2_shadows,s2_changes, l8_clouds, l8_shadows, l8_changes = redo_cloud_shadow(s2_toas, l8_toas, s2_clds, l8_clds)

    sur_x, sur_y = s2_surs[0][0].shape
    possible_x_y, struct, gaus, pointXs, pointYs = cal_psf_points(pix_res, sur_x, sur_y)
    s2_conv_toas, s2_cors_surs, s2_cors_pots = do_s2_psf(s2_toas, s2_clouds, s2_shadows, s2_surs, possible_x_y, struct, gaus, pointXs, pointYs) 
    l8_conv_toas, l8_cors_surs, l8_cors_pots = do_l8_pdf(l8_toas, l8_clouds, l8_shadows, l8_surs, possible_x_y, struct, gaus, pointXs, pointYs)

    cams_dir  = '/vsicurl/http://www2.geog.ucl.ac.uk/~ucfafyi/cams/'
    dem       = '/vsicurl/http://www2.geog.ucl.ac.uk/~ucfafyi/eles/global_dem.vrt'

    s2_auxs, l8_auxs = prepare_aux(s2s, l8s, pix_res, obs_time, dstSRS, outputBounds, dem, cams_dir)
    s2_emus, l8_emus = load_emus(s2s, l8s)

    obs_nums         = len(s2s) + len(l8s) - np.array(s2_clouds + l8_clouds + s2_shadows + l8_shadows).astype(int).sum(axis=0)
    stable_targets   = get_stable_targets(s2_changes + l8_changes, s2_clouds + l8_clouds, s2_shadows + l8_shadows)
    target_mask      = (obs_nums >= 0.5*(len(s2s) + len(l8s))) & (np.array(stable_targets).astype(int).sum(axis=0)>1)




    
    




















