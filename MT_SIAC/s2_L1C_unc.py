import gdal
import datetime
import s2_rut_algo
import numpy as np
import s2_l1_rad_conf as rad_conf
import xml.etree.cElementTree as ET
from reproject import reproject_data




product_meta_file    = '/data/nemesis/S2_data/S2A_MSIL1C_20181209T031111_N0207_R075_T50SMG_20181209T045755.SAFE/MTD_MSIL1C.xml' 
datasctrip_meta_file = '/data/nemesis/S2_data/S2A_MSIL1C_20181209T031111_N0207_R075_T50SMG_20181209T045755.SAFE/DATASTRIP/DS_SGS__20181209T045755_S20181209T031113/MTD_DS.xml' 
#product_meta_file    = '/data/nemesis/S2_data/S2B_MSIL1C_20180611T110619_N0206_R137_T30UYD_20180611T170311.SAFE/MTD_MSIL1C.xml' 
#datasctrip_meta_file = '/data/nemesis/S2_data/S2B_MSIL1C_20180611T110619_N0206_R137_T30UYD_20180611T170311.SAFE/DATASTRIP/DS_SGS__20180611T170311_S20180611T110704/MTD_DS.xml'

def get_coefs(product_meta_file, datasctrip_meta_file):
    time_init = {'Sentinel-2A': datetime.datetime(2015, 6, 23, 10, 00),'Sentinel-2B': datetime.datetime(2017, 3, 7, 10, 00)}
    u_diff_temp_rate = {'Sentinel-2A': [0.15, 0.09, 0.04, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        'Sentinel-2B': [0.15, 0.09, 0.04, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}




    datastrip_meta_tree = ET.parse(datasctrip_meta_file)
    datastrip_meta      = datastrip_meta_tree.getroot()
    alpha  = []
    beta   = []
    bandid = []
    for child in datastrip_meta.findall('{https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Datastrip_Metadata.xsd}Quality_Indicators_Info'):
        for i in child.findall('Radiometric_Info'):
            for j in i.findall('Radiometric_Quality_List'):
                for k in j:
                    for l in k.findall('Noise_Model'):
                        for m in l.findall('BETA'):
                            beta.append(float(m.text))
                        for m in l.findall('ALPHA'):
                            alpha.append(float(m.text))

                    bandid.append(int(k.attrib['bandId']))
    alpha = sorted(alpha, key = lambda i: bandid[alpha.index(i)])
    beta  = sorted(beta,  key = lambda i: bandid[beta.index(i)]) 


    gains  = []
    bandid = []
    for child in datastrip_meta.findall('{https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Datastrip_Metadata.xsd}Image_Data_Info'):
        for i in child.findall('Sensor_Configuration'):   
            for j in i.findall('Acquisition_Configuration'):
                for k in j.findall('Spectral_Band_Info'):                           
                    for l in k:
                        for m in l.findall('PHYSICAL_GAINS'):
                            gains.append(float(m.text))
                        bandid.append(int(l.attrib['bandId']))

    gains  = sorted(gains,  key = lambda i: bandid[gains.index(i)])

    u_diff_temp = None
    spacecraft  = None
    for child in datastrip_meta.findall('{https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Datastrip_Metadata.xsd}General_Info'):
        for l in child.findall('Datatake_Info'):
            for m in l.findall('SPACECRAFT_NAME'):
                spacecraft = m.text
        for i in child.findall('Datastrip_Time_Info'):   
            for j in i.findall('DATASTRIP_SENSING_START'):
                time_start  = datetime.datetime.strptime(j.text, '%Y-%m-%dT%H:%M:%S.%fZ')
                u_diff_temp = (time_start - time_init[spacecraft]).days / 365.25 * np.array(u_diff_temp_rate[spacecraft])

    product_meta_tree = ET.parse(product_meta_file)                                                          
    product_meta      = product_meta_tree.getroot()  

    quant = None
    u_sun = None
    bandid = [] 
    solar_irradiance = []
    for child in product_meta.findall('{https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-1C.xsd}General_Info'):
        for i in child.findall('Product_Image_Characteristics'):
            for j in i.findall('QUANTIFICATION_VALUE'):
                quant = float(j.text)
            for k in i.findall('Reflectance_Conversion'):
                for l in k.findall('U'):
                    u_sun = float(l.text)
                for m in k.findall('Solar_Irradiance_List'):
                    for n in m:
                        solar_irradiance.append(float(n.text))
                        bandid.append(int(n.attrib['bandId']))

    solar_irradiance  = sorted(solar_irradiance,  key = lambda i: bandid[solar_irradiance.index(i)])
    bandid = sorted(bandid)
    return alpha, beta, gains, u_diff_temp, quant, u_sun, solar_irradiance, bandid, spacecraft



def cal_unc(toa, sza, toa_band_id, alpha, beta, gains, u_diff_temp, quant, u_sun, solar_irradiance, bandid, spacecraft):
    rut_algo = s2_rut_algo.S2RutAlgo()
    rut_algo.a     = gains[toa_band_id]
    rut_algo.beta  = beta[toa_band_id]
    rut_algo.alpha = alpha[toa_band_id]
    rut_algo.tecta = sza
    rut_algo.e_sun = solar_irradiance[toa_band_id]
    rut_algo.u_sun = u_sun
    rut_algo.quant = quant
    rut_algo.u_diff_temp = u_diff_temp[toa_band_id]
    unc = rut_algo.unc_calculation(toa, toa_band_id, spacecraft)
    return unc

b2 = '/data/nemesis/S2_data/S2A_MSIL1C_20181209T031111_N0207_R075_T50SMG_20181209T045755.SAFE/GRANULE/L1C_T50SMG_A018090_20181209T031113/IMG_DATA/T50SMG_20181209T031111_B01.jp2'  
sza             = '/data/nemesis/S2_data/S2A_MSIL1C_20181209T031111_N0207_R075_T50SMG_20181209T045755.SAFE/GRANULE/L1C_T50SMG_A018090_20181209T031113/ANG_DATA/SAA_SZA.tif'
#b2 = '/data/nemesis/S2_data/S2B_MSIL1C_20180611T110619_N0206_R137_T30UYD_20180611T170311.SAFE/GRANULE/L1C_T30UYD_A006598_20180611T110704/IMG_DATA/T30UYD_20180611T110619_B12.jp2'
#sza = '/data/nemesis/S2_data/S2B_MSIL1C_20180611T110619_N0206_R137_T30UYD_20180611T170311.SAFE/GRANULE/L1C_T30UYD_A006598_20180611T110704/ANG_DATA/SAA_SZA.tif'
sza = reproject_data(sza, b2, outputType= gdal.GDT_Float32).data[1]/100.
b2  = gdal.Open(b2).ReadAsArray()

coefs = get_coefs(product_meta_file, datasctrip_meta_file)
unc = cal_unc(b2, sza, 12, *coefs)
