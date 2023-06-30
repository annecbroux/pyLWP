#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: billault

Defines a few useful functions for the implementation of the IWV/ LWP retrieval.

"""

from netCDF4 import Dataset
import datetime
import numpy as np
from scipy.interpolate import interp1d

def Ps_to_sea_level(Ps, Ts, alt):
    """ SI units """
    Ps = Ps * np.power((1 - 0.0065*alt/(Ts+0.0065*alt)),-5.257)
    return Ps


def Ps_to_sea_level_old(Ps, Ts, alt):
    """ SI units """
    To = np.zeros(len(Ts))
    To[0] = Ts[0]
    To[1:] = .5*(Ts[:-1]+Ts[1:])
    Rd = 287 # J/(kg.K)
    g = 9.807 # m/s²
    H = Rd/g*To
    Ps = Ps*np.exp(450/H)   
    return Ps

def load_ERA_variables(erapath, t):
    start = datetime.datetime(1900,1,1,0)     
    if type(erapath)== str:
        eradat = Dataset(erapath)    
        dt = [start+datetime.timedelta(hours=h.item()) for h in eradat.variables['time'][:].data]
        tclw = eradat['tclw'][:,3,3]
        tcrw = eradat['tcrw'][:,3,3]
        tcwv = eradat['tcwv'][:,3,3]
        eradat.close()
    elif type(erapath)== list:
        dt = []
        tclw = []
        tcrw = []
        tcwv = []
        for erap in sorted(erapath):
            eradat = Dataset(erap)
            dt += [start+datetime.timedelta(hours=h.item()) for h in eradat.variables['time'][:].data]
            tclw += eradat['tclw'][:,3,3].tolist()
            tcrw += eradat['tcrw'][:,3,3].tolist()
            tcwv += eradat['tcwv'][:,3,3].tolist()
            eradat.close()
        tclw = np.array(tclw)
        tcrw = np.array(tcrw)
        tcwv = np.array(tcwv)
    else:
        raise TypeError('erapath must be a string or a list of strings')
    t_era = np.array([datetime.datetime.timestamp(ddt.replace(tzinfo=datetime.timezone.utc)) for ddt in dt])
    lwp = (tclw+tcrw)*10**3 #LWP is in g/m²
    pwv = tcwv #keep PWV in kg/m²

    f_lwp_interp = interp1d(t_era,lwp,bounds_error=False,fill_value=-1)
    f_pwv_interp = interp1d(t_era,pwv,bounds_error=False,fill_value=-1)

    LWPera = f_lwp_interp(t)
    PWVera = f_pwv_interp(t)
    
    return (LWPera, PWVera)

def load_WProf_variables(path):
    if type(path)== str:
        nc = Dataset(path)
        t =nc.variables['Time'][:].tolist()
        Ts =nc.variables['Environment-temp'][:].tolist()
        RHs =nc.variables['Rel-humidity'][:].tolist()
        Ps =nc.variables['Barometric-pressure'][:].tolist()
        TB =nc.variables['Direct-detection-brightness-temp'][:].tolist()
        LWPrpg =nc.variables['Liquid-water-path'][:].tolist()
    elif type(path)== list:
        t = []
        Ts = []
        TB = []
        RHs = []
        Ps = []
        LWPrpg = []
        
        for f in path_list:
            nc = Dataset(f)
            t+=nc.variables['Time'][:].tolist()
            Ts+=nc.variables['Environment-temp'][:].tolist()
            RHs+=nc.variables['Rel-humidity'][:].tolist()
            Ps+=nc.variables['Barometric-pressure'][:].tolist()
            TB+=nc.variables['Direct-detection-brightness-temp'][:].tolist()
            LWPrpg +=nc.variables['Liquid-water-path'][:].tolist()
    return (np.array(t), np.array(Ts), np.array(Ps), np.array(RHs), np.array(TB),np.array(LWPrpg))