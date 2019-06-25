# A sensor invariant Atmospheric Correction (SIAC)
### Feng Yin
### Department of Geography, UCL
### ucfafyi@ucl.ac.uk


MT-SIAC means multi-temporal SIAC, is a expansion of [SIAC](https://github.com/MarcYin/SIAC/). In this version, we want to exploit the temporal information from both S2 and L8 observtions, under the aims of having better performance in terms of atmospheric parameters retieval spatially and further constain our surface reflectance stabalebility. 

## Data needed:
* MCD43 : 16 days before and 16 days after the Sentinel 2 / Landsat 8 sensing date
* ECMWF CAMS Near Real Time prediction: a time step of 3 hours with the start time of 00:00:00 over the date, and data from 01/04/2015 are mirrored in UCL server at: http://www2.geog.ucl.ac.uk/~ucfafyi/cams/
* Global DEM: Global DEM VRT file built from ASTGTM2 DEM, and most of the DEM over land are mirrored in UCL server at: http://www2.geog.ucl.ac.uk/~ucfafyi/eles/
* Emulators: emulators for atmospheric path reflectance, total transmitance and single scattering Albedo, and the emulators for Sentinel 2, Landsat 8 and MODIS trained with 6S.V2 can be found at: http://www2.geog.ucl.ac.uk/~ucfafyi/emus/

## Installation:

1. Directly from github 

```bash
pip install https://github.com/MarcYin/MT_SIAC/archive/master.zip
```

To save your time for installing GDAL:

```bash
conda install -c conda-forge gdal>2.1
```
## Examples and Map:

A [page](http://www2.geog.ucl.ac.uk/~ucfafyi/Atmo_Cor/index.html) shows some correction samples.

A [map](http://www2.geog.ucl.ac.uk/~ucfafyi/map) for comparison between TOA and BOA.

## Citation:

Yin, F., Lewis, P. E., Gomez-Dans, J., & Wu, Q. (2019, February 21). A sensor-invariant atmospheric correction method: application to Sentinel-2/MSI and Landsat 8/OLI. https://doi.org/10.31223/osf.io/ps957

### LICENSE
GNU GENERAL PUBLIC LICENSE V3
