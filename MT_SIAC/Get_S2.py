import os 
import sys
import zipfile
import requests
import numpy as np
from glob import glob
from datetime import datetime
from os.path import expanduser
import get_scihub_pass
from query import query_sen2
from multiprocessing import Pool, cpu_count
from parse_aoi import parse_aoi
from create_logger import create_logger

home = expanduser("~")
file_path = os.path.dirname(os.path.realpath(__file__))
logger = create_logger()
logger.propagate = True

username, password = np.loadtxt(file_path + '/.scihub_auth', dtype=str)
auth = tuple([username, password])


now = datetime.now().strftime('%Y-%m-%d')
def find_S2_files(aoi = '', tiles = '', latlon = None, start = '2015-12-01', end = now, cloud_cover = 100, one_by_one=True, band = None):
    wkts = []
    rets = []
    if len(aoi) > 0:
        wkts += parse_aoi(aoi = aoi)
    if latlon is not None:
        wkts += parse_aoi(latlon = latlon)
    for wkt in wkts:
        rets += query_sen2(wkt, search_by_tile=True, start=start, end=end, cloud_cover=100, band=None, one_by_one=True) 

    if len(tiles) > 0:
        tiles = np.atleast_1d(tiles).tolist()
        for tile in tiles:
            rets += query_sen2('', search_by_tile=tile, start=start, end=end, cloud_cover=100, band=None, one_by_one=True)
    return rets

def down_s2_file(rets, s2_file_dir = home + '/S2_data'):
    url_fnames = [[j, s2_file_dir + '/' + i] for [i, j] in rets]
    p = Pool(2)
    ret = p.map(downloader, url_fnames)

def downloader(url_fname):
    url, fname = url_fname
    logger.info('Try to download %s'%fname.split('/')[-1])
    if os.path.exists(fname + '.SAFE'):
        logger.info('%s exists, skip downloading'%fname.split('/')[-1])
    else:
        r  = requests.get(url, stream = False, headers={'user-agent': 'My app'}, auth = auth)
        remote_size = int(r.headers['Content-Length'])
        if r.ok:
            data = r.content
            if len(data) == remote_size:
                with open(fname, 'wb') as f:
                    f.write(data)
            else:           
                raise IOError('Failed to download the whole file.')
        else: 
            logger.error(r.content)

def get_s2_files(aoi = '', tiles = '', latlon = None, start = '2015-12-01', end = now, cloud_cover = 100, s2_file_dir = home + '/S2_data'):
    #logger.propagate = False
    rets = find_S2_files(aoi = aoi, tiles = tiles, latlon = latlon, start = start, end =end, cloud_cover = cloud_cover, one_by_one=True, band = None)
    if os.path.realpath(s2_file_dir) in os.path.realpath(home + '/S2_data'):
        if not os.path.exists(home + '/S2_data'):
            os.mkdir(home + '/S2_data') 
    logger.info('Start downloading all the files..')
    #down_s2_file(rets, s2_file_dir = s2_file_dir)
    fnames = [s2_file_dir + '/' + i[0]  for i in rets]
    for fname in fnames:
        if not os.path.exists(fname + '.SAFE'):
            logger.info('Start unziping..')
            logger.info('Unziping %s' %fname.split('/')[-1])
            zip_ref = zipfile.ZipFile(fname, 'r')
            zip_ref.extractall(s2_file_dir)
            zip_ref.close()
            os.remove(fname)
    fnames =  [s2_file_dir + '/' + i[0] + '.SAFE' for i in rets]
    return fnames

