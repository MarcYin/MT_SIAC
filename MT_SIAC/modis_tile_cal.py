#!/usr/bin/env python  
import os
import osr
import ogr
import gdal
import numpy as np
from pyproj import Proj, transform

'''
This is a function used for the calculation of MODIS
tile names from lat and lon coordinates. get_raster_hv 
is used for the calculation of MODIS hv from a raster 
file and get_vector_hv form a raster file

'''

x_step = -463.31271653   
y_step = 463.31271653    
#m_y0, m_x0 = -20015109.354, 10007554.677

tile_width = 1111950.5196666666
# m_x0, m_y0 = -20015109.35579742, -10007554.677898709
m_x0, m_y0 = -20015109.35579742, -10007554.677898709

def get_raster_hv(example_file):
    try:
        g = gdal.Open(example_file)
    except:
        try:
            g = example_file
            g.GetGeoTransform()[0]
        except:
            raise IOError('aoi has to be raster file or a gdal object.')
    geo_t = g.GetGeoTransform()
    x_size, y_size = g.RasterYSize, g.RasterXSize
 
    # wgs84 = osr.SpatialReference( ) # Define a SpatialReference object
    # wgs84.ImportFromEPSG( 4326 ) # And set it to WGS84 using the EPSG code
    # H_res_geo = osr.SpatialReference( )
    
    # H_res_geo.ImportFromWkt(raster_wkt)
    # tx = osr.CoordinateTransformation(H_res_geo, wgs84)
    
    modisProj= Proj('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs')
    wgs84Proj = Proj(init='epsg:4326')
    raster_wkt = Proj(g.GetProjection())
    xs = [geo_t[0], geo_t[0] + geo_t[1]*x_size, geo_t[0], geo_t[0] + geo_t[1]*x_size]
    ys = [geo_t[3], geo_t[3] + geo_t[5]*y_size, geo_t[3] + geo_t[5]*y_size, geo_t[3]]

    lon, lat = transform(raster_wkt, wgs84Proj, np.array(xs).ravel(), np.array(ys).ravel())
    hh, vv = mtile_cal(lat, lon)
    tiles  = ['h%02dv%02d'%(hh[i], vv[i]) for i in range(len(hh))]
    tiles  = np.unique(tiles)
    # so we need the four corners coordiates to check whether they are within the same modis tile
    # (ul_lon, ul_lat, ulz ) = tx.TransformPoint( geo_t[0], geo_t[3])
 
    # (lr_lon, lr_lat, lrz ) = tx.TransformPoint( geo_t[0] + geo_t[1]*x_size, \
    #                                       geo_t[3] + geo_t[5]*y_size )
 
    # (ll_lon, ll_lat, llz ) = tx.TransformPoint( geo_t[0] , \
    #                                       geo_t[3] + geo_t[5]*y_size )
 
    # (ur_lon, ur_lat, urz ) = tx.TransformPoint( geo_t[0] + geo_t[1]*x_size, \
    #                                       geo_t[3]  )
    # a0, b0 = None, None
    # corners = [(ul_lon, ul_lat), (lr_lon, lr_lat), (ll_lon, ll_lat), (ur_lon, ur_lat)]
    # tiles = []
    # for i,j  in enumerate(corners):
    #     h, v = mtile_cal(j[1], j[0])
    #     tiles.append('h%02dv%02d'%(h,v))
    # unique_tile = np.unique(np.array(tiles))
    return tiles #unique_tile


def get_vector_hv(aoi):
    try:
        og = ogr.Open(aoi)
    except:
        try:
            og = aoi
            l = og.GetLayer(0)
        except:
            raise IOError('aoi has to be vector file or a ogr object')
    feature = og.GetLayer(0).GetFeature(0)
    coordinates = feature.geometry().GetGeometryRef(-0).GetPoints()
    lon, lat  = np.array(coordinates).T
    hh, vv = mtile_cal(lat, lon)
    tiles  = ['h%02dv%02d'%(hh[i], vv[i]) for i in range(len(hh))]
    tiles  = np.unique(tiles)
    # for coordinate in coordinates:
    #     h, v = mtile_cal(coordinate[1], coordinate[0])       
    #     tiles.append('h%02dv%02d'%(h,v))
    # unique_tile = np.unique(np.array(tiles))                                                                                                                                             
    return tiles #unique_tile   

def mtile_cal(lat, lon):
    outProj= Proj('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs')
    inProj = Proj(init='epsg:4326')
    ho, vo = transform(inProj,outProj, np.array(lon).ravel(), np.array(lat).ravel())
    h = ((ho-m_x0)/tile_width).astype(int)
    v = 17 - ((vo-m_y0)/tile_width).astype(int)
    return h, v

if __name__ == '__main__':
    aoi          = '~/DATA/S2_MODIS/l_data/LC08_L1TP_014034_20170831_20170915_01_T1/AOI.json'
    example_file = '~/DATA/S2_MODIS/l_data/LC08_L1TP_014034_20170831_20170915_01_T1/aot.tif'
    print(get_vector_hv(aoi))
    print(get_vector_hv(ogr.Open(aoi)))
    print(get_raster_hv(example_file))
    print(get_raster_hv(gdal.Open(example_file)))

