#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create idealized moisture profiles: use a moist adiabats with various temperatures
and a given relative humidity profile (step) provided as input to provide qv

Created on Tue Jan 11 09:18:33 2022

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
from scipy.io import loadmat

# Constants - could use a module if that's better
#
hPa_to_Pa = 100.
epsilon = 0.6223 # Ratio of molar mass of water to dry air
CtoK    = 273.15 # Celsius to Kelvin
gtokg   = 1.e-3
ghgs    = ["co2", "ch4", "n2o", "o3", "o2", "n2", "co"]

# Load own modules
projectname = 'EUREC4A_rad_profiles'
workdir = '/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_rad_profiles/scripts/'
# workdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.dirname(workdir)
while os.path.basename(rootdir) != projectname:
    rootdir = os.path.dirname(rootdir)
repodir = rootdir
moduledir = os.path.join(os.path.dirname(repodir),'EUREC4A_organization/functions')
resultdir = os.path.join(repodir,'results','idealized_calculations')
inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
resultinputdir = os.path.join(os.path.dirname(repodir),'EUREC4A_organization/results','radiative_features')
radinputdir = os.path.join(os.path.dirname(repodir),'EUREC4A_organization/input')
scriptsubdir = 'warming'
tempinputdir = os.path.join(os.path.dirname(repodir),'EUREC4A_organization/input','moistadiabat')

# current environment
thismodule = sys.modules[__name__]

##-- Own modules
sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

#- Parameters & constants
from thermoConstants import *
from thermoFunctions import *
from matrixoperators import *

mo = MatrixOperators()

## Graphical parameters
plt.style.use(os.path.join(matplotlib.get_configdir(),'stylelib/presentation.mplstyle'))


# def getIdealizedProf(radprf_id,varids,i_id):

#     z_id = radprf_id.zlay[i_id]/1e3

#     temp_id = radprf_id.tlay[i_id]
#     qv_id = radprf_id.qv[i_id]
#     rh_id = radprf_id.rh[i_id]
#     qvstar_id = qv_id/rh_id

#     return z_id,temp_id,qv_id,qvstar_id,rh_id

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
    # lw cooling
    qradlw = rad_features.qrad_lw_smooth[k,:]

    return temp, qv, rh, qradlw


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
    
    N_breaks = len(z_breaks)
    
    cond_list = [z <= z_breaks[0]]+\
                [np.logical_and(z > z_breaks[i-1],z <= z_breaks[i]) for i in range(1,N_breaks)]+\
                [z > z_breaks[N_breaks-1]]
    def make_piece(k):
        def f(z):
            return rh_breaks[k-1]+(rh_breaks[k]-rh_breaks[k-1])/(z_breaks[k]-z_breaks[k-1])*(z-z_breaks[k-1])
        return f 
    func_list = [lambda z: rh_breaks[0]]+\
                [make_piece(k) for k in range(1,N_breaks)]+\
                [lambda z: rh_breaks[N_breaks-1]]
                
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



##---- Merge profiles for radiative calculation
    
def mergeProfile(index:int, name:str, z_rh:np.array, pres_rh:np.array, rh:np.array,
                 z_temp:np.array, temp:np.array,
                  deltaP:float=100, sfc_emis:float=.98, sfc_alb:float=0.07,
                  mu0:float=1.,background_file:str=None,h2o_only:bool=False):
    """
    Create thermodynamic profile for radiative calculation.

    Args:
        index (int): profile index
        name (str): name of profile
        z_rh (np.array): z coordinate (km)
        pres_rh (np.array): pressure coordinate (Pa)
        rh (np.array): relative humidity ()
        z_temp (np.array): z coordinate for temperature profile (km)
        temp (np.array): temperature profile (K)
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
    temp_sfc = temp[0]+1 # K

    #- background file
    back = xr.open_dataset(background_file)
    
    #- pressure coordinates
    play = np.flipud(np.linspace(deltaP,pres_sfc,int(pres_sfc/deltaP)))
    # destination layer pressures for background sounding in increasing order
    play_switch = np.ceil(np.nanmin(pres_rh))
    whereto_back = play <= play_switch
    whereto_id = np.logical_not(whereto_back)
    # Interface pressures: mostly the average of the two neighboring layer pressures
    plev = np.append(np.append(play.max() + deltaP/2.,0.5 * (play[1:] + play[:-1])), play.min() - deltaP/2.)
    
    #- z coordinate
    z_lay = np.full(play.shape,np.nan)
    # interpolate
    z_lay = interp1d(pres_rh,z_rh,fill_value='extrapolate')(play)
    # interface heights
    diffZ = np.diff(z_lay)
    z_lev = np.append(np.append(z_lay.min() - diffZ[0]/2.,0.5 * (z_lay[1:] + z_lay[:-1])), z_lay.max() + diffZ[-1]/2.)

    #- temperature
    temp_lay = np.full(play.shape,np.nan)
    # interpolate idealized profile
    temp_lay[whereto_id] = interp1d(z_temp,temp)(z_lay[whereto_id])
    # interpolate background profile
    temp_lay[whereto_back] = interp1d(back.p_lay.data,back.t_lay.data)(play[whereto_back])
    
    #- saturated specific humidity from temp and pres profiles #keeping pres in Pa
    qvstar_lay = saturationSpecificHumidity(temp_lay,play)
    
    #- relative humidity
    rh_lay = np.full(play.shape,np.nan)
    # interpolate idealized profile
    rh_lay[whereto_id] = interp1d(pres_rh,rh)(play[whereto_id])
    rh_lay[whereto_back] = np.nanmin(rh)

    #- specific humidity
    qv_lay = qvstar_lay*rh_lay

    #- volume mixing ratio
    h2o_lay = qv_lay/(1-qv_lay) / epsilon
    # interpolate background profile
    h2o_lay[whereto_back] = interp1d(back.p_lay.data,back.vmr_h2o.data)(play[whereto_back])
    

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
    
    day = '20200126'
    z_min = 1000 # m
    z_max = 2200 # m
    
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
    
    
    #-- Radiative features
    features_filename = 'rad_features.pickle'
    print('loading %s'%features_filename)
    # load
    features_path = os.path.join(resultinputdir,day,features_filename)
    rad_features = pickle.load(open(features_path,'rb'))
    

    # coordinates
    z = data_all.alt.values/1e3 # km
    pres = np.nanmean(data_all.pressure.data,axis=dim_t)/100 # hPa
    

    #-- temperature moist adiabat
    temp_ad_files = glob.glob(os.path.join(tempinputdir,'DATA*'))
    
    temp_ad_data = {}
    for file in temp_ad_files:
        SST_str = file[-7:-4]
        temp_ad_data[SST_str] = loadmat(file)

    # to use like
    # temp_ad_data['302']['Tma_'][:,0]
    
    #-- RH profiles from 2020-01-26 fitted piecewise linear
    
    # all profiles
    temp, qv, rh, qradlw = getProfiles(rad_features, data_day, z_min, z_max)
    
    # median profile
    rh_med = np.nanpercentile(rh,50,axis=0)

    # piecewise linear fit for RH(z)
    z_breaks_0 = [1.8,2]
    rh_breaks_0 = [0.8,0.1]
    z_breaks_id,rh_breaks_id,rh_id = piecewise_fit(z,rh_med,z_breaks_0,rh_breaks_0)
    
    print('- Fitted parameters, for %s, peaks below %1.1fkm'%(day,z_max/1e3))
    print('z_breaks :',z_breaks_id)
    print('rh_breaks :',rh_breaks_id)


    ##-- Put them all together    

    print("- list RH-temp profiles combinations")
    
    rh_profiles = []
    temp_profiles = []
    profile_names = []
    
    #-- First, add the profiles with the original data
     
    #- Add the reference RH profile
    rh_profiles.append(rh_med)
    temp_profiles.append(temp_ad_data['300']['Tma_'][:,0])
    profile_names.append('ref')
    
    #- Add profiles for all temperatures

    for SST_str in temp_ad_data.keys():
        
        rh_profiles.append(rh_id)
        temp_profiles.append(temp_ad_data[SST_str]['Tma_'][:,0])
        profile_names.append('RHid_SST%s'%SST_str)

    
    ##--- Merge profiles with background file
    
    all_profiles = []
    N_prof = len(rh_profiles)
    
    for i,name in zip(range(N_prof),profile_names):
        
        rh = rh_profiles[i]
        z_rh = z
        pres_rh = pres*1e2
        temp = temp_profiles[i]
        z_temp = temp_ad_data['300']['zgrd'][:,0]/1e3

        
        #- First, add median profile
        prof = mergeProfile(i,name,z_rh,pres_rh,rh,z_temp,temp,
                             deltaP=args.deltaP,
                             sfc_emis=args.emis,
                             sfc_alb=args.alb,
                             mu0=args.mu0,
                             background_file=args.background_file,
                             h2o_only=args.h2o_only)
        
        all_profiles.append(prof)

    
    ##-- Save profiles to disk

    for i,profile in zip(range(N_prof),all_profiles):
                         
        output_file = "rrtmgp_" + '{:>04}'.format(i)+'.nc'
        print(output_file)
        profile.to_netcdf(os.path.join(output_dir, output_file))
    
        
        
    