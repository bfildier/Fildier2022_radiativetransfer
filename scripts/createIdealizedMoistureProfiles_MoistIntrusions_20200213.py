#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create idealized moisture profiles, by fitting parametric profiles to EUREC4A
sounding data on 2020-01-13, and remove moist intrusions for radiative calculations.

Created on Thu Nov 25 11:31:59 2021

@author: bfildier
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os,sys,glob
from math import *
import xarray as xr
import pickle
import pytz
from datetime import datetime as dt
from datetime import timedelta, timezone
from scipy import optimize
from scipy.stats import linregress
from scipy.interpolate import interp1d
import argparse

# Constants - could use a module if that's better
#
hPatoPa = 100.
epsilon = 0.6223 # Ratio of molar mass of water to dry air
CtoK    = 273.15 # Celsius to Kelvin
gtokg   = 1.e-3
ghgs    = ["co2", "ch4", "n2o", "o3", "o2", "n2", "co"]

# Load own modules
projectname = 'EUREC4A_organization'
#workdir = '/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_organization/'
workdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.dirname(workdir)
while os.path.basename(rootdir) != projectname:
    rootdir = os.path.dirname(rootdir)
repodir = rootdir
moduledir = os.path.join(repodir,'functions')
resultdir = os.path.join(repodir,'results','idealized_calculations')
figdir = os.path.join(repodir,'figures','idealized_calculations')
inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
resultinputdir = os.path.join(repodir,'results','radiative_features')
scriptsubdir = 'observed_moist_intrusions'


# current environment
thismodule = sys.modules[__name__]

##-- Own modules
sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

#- Parameters & constants
from thermoConstants import *
from radiativefeatures import *


## Graphical parameters
plt.style.use(os.path.join(matplotlib.get_configdir(),'stylelib/presentation.mplstyle'))


def getProfiles(rad_features, data_day, z_min, z_max):

    #- Mask
    # |qrad| > 5 K/day
    qrad_peak = np.absolute(rad_features.qrad_lw_peak)
    keep_large = qrad_peak > 5 # K/day
    # in box
    lon_day = data_day.longitude[:,50]
    lat_day = data_day.latitude[:,50]
    keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
    # high-level peak
    keep_high =  np.logical_and(rad_features.z_net_peak < z_max, # m
                                rad_features.z_net_peak > z_min)
    # combined
    k = np.logical_and(np.logical_and(keep_large,keep_box),keep_high)

    # temperature
    temp = data_day.temperature.values[k,:]
    # specific humidity
    qv = data_day.specific_humidity.values[k,:]
    # relative humidity    
    rh = data_day.relative_humidity.values[k,:]
    # saturated specific humidity
    qvstar = qv/rh

    return temp, qv, qvstar, rh


def computeMedianProfiles(varids):

    for varid in varids:
    
        # get variable
        var = getattr(thismodule,varid)
        # compute quartiles
        var_med = np.nanpercentile(var,50,axis=0)
        # reassign in current environment with correct variable name
        setattr(thismodule,'%s_med'%(varid),var_med)


##-- idealization of thermodynamic profiles

def computeLinearTemperatureFit():
    """Linear regression of T(z)"""
    
    mask = ~np.isnan(z) & ~np.isnan(temp_med)
    slope, intercept, r, p, se = linregress(z[mask],temp_med[mask])
    # print('T(z) = %2.1f z + %3.1f, r = %1.2f'%(slope,intercept,r))
    temp_id = slope*z + intercept
    
    return temp_id


def computePowerFitQvstar():
    """Power fit of qvstar(p)"""
    
    mask = ~np.isnan(pres) & ~np.isnan(qvstar_med)
    slope, intercept, r, p, se = linregress(np.log(pres[mask]),np.log(qvstar_med[mask]))
    # print('ln(qvstar) = %2.1f ln(p) + %3.1f, r = %1.2f'%(slope,intercept,r))
    # print('alpha = %1.2f, qvstar(1000hPa) = %1.2f'%(slope,slope*np.log(1000)+intercept))
    qvstar_id = np.exp(slope*np.log(pres) + intercept)
    
    return qvstar_id

def piecewise_linear(z:np.array,z_breaks:list,rh_breaks:list):
    """
    Define piecewise linear RH shape with constant value at top and bottom.

    Args:
        z (np.array): z coordinate
        z_breaks (list): z values of break points
        rh_breaks (list): rh values of break points

    Returns:
        np.array: piecewize rh
        
    """
    
    cond_list = [z <= z_breaks[0]]+\
                [np.logical_and(z > z_breaks[i-1],z <= z_breaks[i]) for i in range(1,6)]+\
                [z > z_breaks[5]]
    def make_piece(k):
        def f(z):
            return rh_breaks[k-1]+(rh_breaks[k]-rh_breaks[k-1])/(z_breaks[k]-z_breaks[k-1])*(z-z_breaks[k-1])
        return f 
    func_list = [lambda z: rh_breaks[0]]+\
                [make_piece(k) for k in range(1,6)]+\
                [lambda z: rh_breaks[5]]
                
    return np.piecewise(z,cond_list,func_list)

def piecewise_fit(z:np.array,rh:np.array,z_breaks_0:list,rh_breaks_0:list):    
    """
    Compute piecewise-linear fit of RH(z).

    Args:
        z (np.array): z coordinate
        rh (np.array): rh profile
        z_breaks_0 (list): initial z values of break points
        rh_breaks_0 (list): initial rh values of break points

    Returns:
        z_breaks (list): fitted z values of break points
        rh_breaks (list): fitted rh values of break points
        rh_id (np.array): piecewize rh fit

    """

    N_breaks = len(z_breaks_0)
    
    def piecewise_fun(z,*p):
        return piecewise_linear(z,p[0:N_breaks],p[N_breaks:2*N_breaks])

    mask = ~np.isnan(z) & ~np.isnan(rh)
    p , e = optimize.curve_fit(piecewise_fun, z[mask], rh[mask],p0=z_breaks_0+rh_breaks_0)

    rh_id = piecewise_linear(z,p[0:N_breaks],p[N_breaks:2*N_breaks])
    z_breaks= list(p[0:N_breaks])
    rh_breaks = list(p[N_breaks:2*N_breaks])

    return z_breaks,rh_breaks,rh_id



##--- Removing moisture intrusions to idealized and flattened profiles

def removeIntrusion(which:str,z_breaks:np.array,rh_breaks:np.array):
    """
    Compute rh profile after removing a moisture intrusion

    Args:
        which (str): 'upper' or 'lower' intrusion
        z_breaks (list): initial z values of break points
        rh_breaks (list): initial rh values of break points

    Returns:
        z_breaks_new (list): adjusted z values of break points
        rh_breaks_new (list): adjusted rh values of break points
        rh_new (np.array): output rh profile

    """
    
    if which == 'upper':
        i_peaks = [4]
    elif which == 'lower':
        i_peaks = [2]
    elif which == 'both':
        i_peaks = [2,4]
    
    # update params
    z_breaks_new = z_breaks.copy()
    rh_breaks_new = rh_breaks.copy()
    for i_peak in i_peaks:
        rh_breaks_new[i_peak] = rh_breaks_new[i_peak+1] # set rh peak value to that of the one just above

    # rh
    rh_new = piecewise_linear(z,z_breaks_new,rh_breaks_new)

    return z_breaks_new,rh_breaks_new,rh_new
            
    
##---- Merge profiles for radiative calculation
    
def mergeProfile(index:int, name:str, z:np.array, pres:np.array, temp:np.array, qv:np.array, rh:np.array,
                  deltaP:float=100, sfc_emis:float=.98, sfc_alb:float=0.07,
                  mu0:float=1.,background_file:str=None,h2o_only:bool=False):
    """
    Create thermodynamic profile for radiative calculation.

    Args:
        index (int): profile index
        name (str): name of profile
        z (np.array): z coordinate (km)
        pres (np.array): pressure coordinate (Pa)
        temp (np.array): temperature profile (K)
        qv (np.array): specific humidity profile (kg/kg)
        rh (np.array): relative humidity profile
        deltaP (float, optional): pressure increment in final grid (Pa). Defaults to 100.
        sfc_emis (float, optional): surface emissivity. Defaults to .98.
        sfc_alb (float, optional): surface albedo. Defaults to 0.07.
        mu0 (float, optional): cosine of solar zenith angle. Defaults to 1.
        background_file (str, optional): reference background profile to merge. Defaults to None.
        h2o_only (boolean, optional): only treat radiative effect of water vapor (set CO2, CH4, N20 to zero).

    Returns:
        profile (xr.array): output .

    """

    pres_sfc = 1e5 # Pa
    i_sfc = np.where(pres<=pres_sfc)[0][0]
    temp_sfc = temp[i_sfc] # K

    #- background file
    back = xr.open_dataset(background_file)
    
    #- pressure coordinates
    play = np.flipud(np.linspace(deltaP,pres_sfc,int(pres_sfc/deltaP)))
    print("play shape:",play.shape)
    # destination layer pressures for background sounding in increasing order
    play_switch = np.ceil(np.nanmin(pres))
    whereto_back = play <= play_switch
    whereto_id = np.logical_not(whereto_back)
    # Interface pressures: mostly the average of the two neighboring layer pressures
    plev = np.append(np.append(play.max() + deltaP/2.,0.5 * (play[1:] + play[:-1])), play.min() - deltaP/2.)
    # # revert
    # plev = np.flipud(plev)
    
    #- z coordinate
    z_lay = np.full(play.shape,np.nan)
    # interpolate
    z_lay = interp1d(pres,z,fill_value='extrapolate')(play)
    # interface heights
    diffZ = np.diff(z_lay)
    z_lev = np.append(np.append(z_lay.min() - diffZ[0]/2.,0.5 * (z_lay[1:] + z_lay[:-1])), z_lay.max() + diffZ[-1]/2.)

    #- temperature        
    temp_lay = np.full(play.shape,np.nan)
    # interpolate idealized profile
    temp_lay[whereto_id] = interp1d(pres,temp)(play[whereto_id])
    # interpolate background profile
    temp_lay[whereto_back] = interp1d(back.p_lay.data,back.t_lay.data)(play[whereto_back])

    #- volume mixing ratio
    h2o_lay = np.full(play.shape,np.nan)
    # interpolate idealized profile
    h2o = qv/(1-qv) / epsilon
    h2o_lay[whereto_id] = interp1d(pres,h2o)(play[whereto_id])
    # interpolate background profile
    h2o_lay[whereto_back] = interp1d(back.p_lay.data,back.vmr_h2o.data)(play[whereto_back])

    #- specific humidity
    qv_lay = np.full(play.shape,np.nan)
    # interpolate idealized profile
    qv_lay[whereto_id] = interp1d(pres,qv)(play[whereto_id])
    
    #- relative humidity
    rh_lay = np.full(play.shape,np.nan)
    # interpolate idealized profile
    rh_lay[whereto_id] = interp1d(pres,rh)(play[whereto_id])

    #- store all
    profile = xr.Dataset({"index":([], index), \
                      "name":([],name),\
                      "tlay":(["play"], temp_lay), \
                      "play":(["play"], play), \
                      "zlay":(["play"], z_lay), \
                      "h2o":(["play"], h2o_lay), \
                      "qv":(["play"], qv_lay), \
                      "rh":(["play"], rh_lay), \
                      "plev"   :(["plev"], plev), \
                      "zlev"   :(["plev"], z_lev), \
                      "sfc_emis":([], sfc_emis), \
                      "sfc_alb":([], sfc_alb), \
                      "sfc_t":([], temp_sfc), \
                      "cos_sza":([], mu0), \
                      "lw_dn"  :(["plev"], np.repeat(np.nan, plev.size)),\
                      "lw_up"  :(["plev"], np.repeat(np.nan, plev.size)),\
                      "lw_net" :(["plev"], np.repeat(np.nan, plev.size)),\
                      "sw_dn"  :(["plev"], np.repeat(np.nan, plev.size)),\
                      "sw_up"  :(["plev"], np.repeat(np.nan, plev.size)),\
                      "sw_net" :(["plev"], np.repeat(np.nan, plev.size))})

    # Add the other greenhouse gases
    lowest = back.p_lay.argmax()
    back_on_p = back.swap_dims({'lay':'p_lay'}).reset_coords() # Background sounding on pressure layers
    back_on_p = back_on_p.rename({'p_lay':'play'}) # Rename p_lay into play
    for g in ghgs:
        
        profile[g] = back_on_p["vmr_" + g].interp(play=play).fillna(back["vmr_" + g].isel(lay=lowest))

        if h2o_only == 'True' and g in ['co2','ch4','n2o']:
            print('replace %s concentration with 0'%g)
            profile[g][:] = 0
            
    back.close()

        
    return profile
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Builds a netCDF file suitable for computing radiative fluxes by merging a background sounding with an idealized thermodynamic profile")
    parser.add_argument("--background_file", type=str, default="../input/tropical-atmosphere.nc",
                        help="Directory where reference values are")
    parser.add_argument("--deltaP", type=int, default=100,
                        help="Pressure discretization of sonde (Pa, integer)")
    parser.add_argument("--sfc_emissivity", type=float, default=0.98, dest="emis",
                        help="Surface emissivity (spectrally constant)")
    parser.add_argument("--sfc_albedo", type=float, default=0.07, dest="alb",
                        help="Surface albedo (spectrally constant, same for direct and diffuse)")
    parser.add_argument("--mu_0", type=float, default=1, dest="mu0",
                        help="Cosine of solar zenith angle, default is to compute from sonde file (someday)")
    parser.add_argument("--out_dir", type=str, default="../output/rad_profiles/idealized_profiles",
                        help="Directory where the output files should be saved")
    parser.add_argument("--out_file", type=str, default=None,
                        help="Output file name")
    parser.add_argument("--h2o_only", type=str, default='False',
                        help="Only use H2O")
    args = parser.parse_args()

    output_dir  = args.out_dir
    


    ##-- Let's get started
    
    day = '20200213'
    
    # box of analysis
    lat_box = 11,16
    lon_box = -60,-52
    
    dim_t = 0
    
    
    ##--- Load data 
    
    # Profiles
    radprf = xr.open_dataset(os.path.join(inputdir,'rad_profiles_CF.nc'))
    # choose profiles for that day that start at bottom
    data_all = radprf.where(radprf.z_min<=50,drop=True)
    data_day = data_all.sel(launch_time=day)
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    
    # coordinates
    z = data_all.alt.values/1e3 # km
    pres = np.nanmean(data_all.pressure.data,axis=dim_t)/100 # hPa
    
    #-- Radiative features
    features_filename = 'rad_features.pickle'
    print('loading %s'%features_filename)
    # load
    features_path = os.path.join(resultinputdir,day,features_filename)
    rad_features = pickle.load(open(features_path,'rb'))
    
    
    # height range for radiative cooling peaks
    z_min = 5000 # m
    z_max = 9000 # m
    
    
    # -- get profiles
    temp, qv, qvstar, rh = getProfiles(rad_features, data_day, z_min, z_max)
    
    # -- compute median profiles
    varids = 'temp','qv','qvstar','rh'
    Nv = len(varids)
    computeMedianProfiles(varids)
        
        
    ##--- Idealization profiles of median profiles
    
    # linear temperature fit T(z)
    temp_id = computeLinearTemperatureFit()
    
    # power fit of qvstar(p)
    qvstar_id = computePowerFitQvstar()
    
    # piecewise linear fit for RH(z)
    z_breaks_0 = [1.8,2,4,5,6.5,6.8]
    rh_breaks_0 = [0.8,0.1,0.65,0.1,0.65,0.05]
    z_breaks_id,rh_breaks_id,rh_id = piecewise_fit(z,rh_med,z_breaks_0,rh_breaks_0)
    
    # resulting piecewise_power qv
    qv_id = rh_id*qvstar_id
    
    
    ##--- Flattening of moist intrusions
    
    # flatten rh
    z_breaks_idf = z_breaks_id.copy()
    rh_breaks_idf = rh_breaks_id.copy()
    z_breaks_idf[1] = z_breaks_id[0] # flatten boundary layer
    z_breaks_idf[3] = z_breaks_id[2] # flatten lower intrusion
    z_breaks_idf[5] = z_breaks_id[4] # flatten upper intrusion
    
    rh_idf = piecewise_linear(z,z_breaks_idf,rh_breaks_idf)
    
    # qvstar
    qvstar_idf = qvstar_id
    
    # resulting piecewise_power qv
    qv_idf = rh_idf*qvstar_id
    
    # temp
    temp_idf = temp_id
        
        
    ##--- Removing moisture intrusions to idealized and flattened profiles
    
    for id_suff in 'id','idf':
        
        for which,w_suff in zip(['lower','upper','both'],['rl','ru','rul']):
            
            # fetch
            z_breaks_in = getattr(thismodule,'z_breaks_%s'%id_suff)
            rh_breaks_in = getattr(thismodule,'rh_breaks_%s'%id_suff)
            
            # rh
            z_breaks_new,rh_breaks_new,rh_new = removeIntrusion(which,z_breaks_in,rh_breaks_in)
        
            # qvstar
            qvstar_new = qvstar_id
        
            # qv
            qv_new = qvstar_new*rh_new
        
            # temp
            temp_new = temp_id
        
            # save
            for varid in 'z_breaks','rh_breaks','rh','qv','qvstar','temp':
                
                var = getattr(thismodule,"%s_new"%varid)
                setattr(thismodule,'%s_%s_%s'%(varid,id_suff,w_suff),var)
                
    
    ##--- Merge profiles with background file
    
    all_profiles = []
    
    # profile counter
    i = 0
    
    #- First, add median profile
    prof = mergeProfile(0,'reference',z*1e3,pres*1e2,temp_med,qv_med,rh_med,
                     deltaP=args.deltaP,
                     sfc_emis=args.emis,
                     sfc_alb=args.alb,
                     mu0=args.mu0,
                     background_file=args.background_file)
    
    all_profiles.append(prof)
    i += 1
    
    #- Add all idealized profiles
    # loop
    for id_suff in 'id','idf':
        
        for which,w_suff in zip(['none','lower','upper','both'],['','_rl','_ru','_rul']):
            
            name = "%s%s"%(id_suff,w_suff)
            
            # fetch variables
            for varid in 'rh','qv','temp':
                
                var = getattr(thismodule,'%s_%s%s'%(varid,id_suff,w_suff))
                setattr(thismodule,varid,var)

            # merge profiles
            prof = mergeProfile(i,name,z*1e3,pres*1e2,temp,qv,rh,
                                 deltaP=args.deltaP,
                                 sfc_emis=args.emis,
                                 sfc_alb=args.alb,
                                 mu0=args.mu0,
                                 background_file=args.background_file,
                                 h2o_only=args.h2o_only)
            
            # store
            all_profiles.append(prof)

        i += 1

    
    ##-- Save profiles to disk
    
    N_prof = len(all_profiles)

    for i,profile in zip(range(N_prof),all_profiles):
                         
        output_file = "rrtmgp_" + '{:>04}'.format(i)+'.nc'
        print(output_file)
        profile.to_netcdf(os.path.join(output_dir, output_file))
    
        
        
    
