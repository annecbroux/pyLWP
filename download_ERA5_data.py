#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: billault

Code to download ERA5 reanalysis data from copernicus server.
Data is downloaded for a small square bounded with +/- 1 degree of lat/lon from the desired location.py
The parameters (lat, lon, time frame etc.) should be set in the "__main__" part of the script (end). 
"""

import cdsapi
import os
import numpy as np
from netCDF4 import Dataset
import glob


def download_ERA(lat,lon,year,month_str_list, day_str_list, time_str_list, save_to_nc_file):
    # lat and lon must be in degrees (decimal)
    # month_str_list (resp. day_str_list, time_str_list) contain the list of months which we want to download (in strings)
    N = lat+.5
    W = lon-.5
    S = lat-.5
    E = lon+.5
    c = cdsapi.Client()
    c.retrieve('reanalysis-era5-single-levels',{
        'product_type':'reanalysis',
        'variable':['total_column_cloud_liquid_water','total_column_rain_water',
                    'total_column_water','total_column_water_vapour'],
        'year':str(year),
        'month': month_str_list,
        'day': day_str_list,
        'area':[N,W,S,E],
        'time': time_str_list,
        'format':'netcdf'
        },save_to_nc_file)
     
    

if __name__=='__main__':
    year = 2021
    month_str_list = ['12'] #,'11','12']
    day_str_list = ['01','02','03', '04','05','06','07','08','09','10','11','12','13','14','15',
                   '16','17','18','19','20','21', '22','23','24','25','26','27','28','29','30','31']
    time_str_list = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00',
                    '12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
    lat = 38.0
    lon = 22.2
    save_to_nc_file = '/home/billault/Documents/In_progress/HELMOS_2021/ERA5_LWP_2021_12.nc'
    download_ERA(lat,lon,year,month_str_list, day_str_list, time_str_list, save_to_nc_file)
