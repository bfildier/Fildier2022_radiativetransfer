#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create idealized moisture profiles, by fitting parametric profiles to EUREC4A
sounding data on 2020-01-13, and remove moist intrusions for radiative calculations.

Use lower intrusion.

Created on Wed Mar  2 12:39:01 2022

@author: bfildier
"""

from pathlib import WindowsPath
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
workdir = '/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_organization/'
# workdir = os.path.dirname(os.path.realpath(__file__))
rootdir = os.path.dirname(workdir)
print('rootdir:',rootdir)
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
# from radiativefeatures import *
from matrixoperators import *
mo = MatrixOperators()

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
    
    Nb = len(z_breaks)
    
    cond_list = [z <= z_breaks[0]]+\
                [np.logical_and(z > z_breaks[i-1],z <= z_breaks[i]) for i in range(1,Nb)]+\
                [z > z_breaks[Nb-1]]
    def make_piece(k):
        def f(z):
            return rh_breaks[k-1]+(rh_breaks[k]-rh_breaks[k-1])/(z_breaks[k]-z_breaks[k-1])*(z-z_breaks[k-1])
        return f 
    func_list = [lambda z: rh_breaks[0]]+\
                [make_piece(k) for k in range(1,Nb)]+\
                [lambda z: rh_breaks[Nb-1]]
    
    return np.piecewise(z,cond_list,func_list)

def computeWPaboveZ(qv,pres,p_top):
    """Calculates the integrated water path above each level.

    Arguments:
        - qv: specific humidity in kg/kg, Nz-vector
        - pres: pressure coordinate in hPa, Nz vector
        - p_top: pressure of upper integration level

    returns:
        - wp_z: water path above each level, Nz-vector"""

    Np = qv.shape[0]
    wp_z = np.full(Np,np.nan)

    p_increasing = np.diff(pres)[0] > 0
    
    if p_increasing:
        
        i_p_top = np.where(pres >= p_top)[0][0]
        
        for i_p in range(i_p_top,Np):
        # self.wp_z[:,i_z] = self.mo.pressureIntegral(arr=data.specific_humidity[:,i_z:],pres=pres[i_z:],p_levmin=pres[i_z],p_levmax=pres[-1],z_axis=z_axis)

            arr = qv
            p = pres
            p0 = p_top
            p1 = p[i_p]
            i_w = i_p
            
            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    else:
        
        i_p_top = np.where(pres >= p_top)[0][-1]

        for i_p in range(i_p_top):
            
            arr = np.flip(qv)
            p = np.flip(pres)
            p0 = p_top
            p1 = pres[i_p]
            i_w = i_p

            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    return wp_z

def waterPath(qv,pres,p_bottom,p_top):
    
    p_increasing = np.diff(pres)[0] > 0

    if p_increasing:

        arr = qv
        p = pres

    else:

        arr = np.flip(qv)
        p = np.flip(pres)

    p0 = p_top
    p1 = p_bottom

    return mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

def saturatedWaterPath(temp,pres,p_bottom,p_top):
    
    p_increasing = np.diff(pres)[0] > 0

    if p_increasing:

        p = pres
        arr = saturationSpecificHumidity(temp,p*hPa_to_Pa)

    else:

        p = np.flip(pres)
        arr = saturationSpecificHumidity(np.flip(temp),p*hPa_to_Pa)
        
    p0 = p_top
    p1 = p_bottom

    return mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

def z2p(z_0,z,pres):
    """Assume z is increasing"""
    
    i_z = np.where(z>=z_0)[0][0]
    
    return pres[i_z]

def p2z(p_0,pres,z):
    """Assume p is decreasing"""
    
    i_p = np.where(pres<=p_0)[0][0]
    
    return z[i_p]

# Add rectangle intrusion to constant RH -- can center on prescribed level
def addRectangleIntrusionToProfile(pres,qvstar,rh_prof,W_int,p_int,rh_max,where='below'):
    """Add moisture intrusion as a rectangle in RH"""
    
    #-- find vertical extent of intrusion
    # q deficit (RH to RHmax)
    delta_qv_prof = (rh_max-rh_prof)*qvstar
    i_int = np.where(pres>p_int)[0][-1]

    # masks on each side of reference level
    mask_above_lev = (pres < p_int).data
    mask_below_lev = (pres > p_int).data

    # water paths on each side of reference level
    delta_qv_below_lev = delta_qv_prof.copy()
    delta_qv_below_lev[mask_above_lev] = 0

    delta_qv_above_lev = delta_qv_prof.copy()
    delta_qv_above_lev[mask_below_lev] = 0

    Wdqv_below_lev = computeWPaboveZ(delta_qv_below_lev,pres,-1)
    Wdqv_above_lev = computeWPaboveZ(delta_qv_above_lev,pres,-1)
    Wdqv_below_lev[np.isnan(Wdqv_below_lev)] = 0
    Wdqv_above_lev[np.isnan(Wdqv_above_lev)] = 0

    if where == 'below':

        # levels where W is below water path of intrusion
        mask_below_intrusion = (Wdqv_below_lev >= W_int).data
        # merge masks
        mask_intrusion = np.logical_not(np.logical_or(mask_below_intrusion,
                                                      mask_above_lev))

    elif where == 'above':

        # levels where W is below water path of intrusion
        mask_above_intrusion = (Wdqv_above_lev[i_int]-Wdqv_above_lev >= W_int).data
        # merge masks
        mask_intrusion = np.logical_not(np.logical_or(mask_above_intrusion,
                                                      mask_below_lev))

    elif where =='centered':

        # half water path below -- mask beyond lower half
        mask_below_intrusion = (Wdqv_below_lev >= W_int/2).data
        # half water path above -- madk beyond upper half
        mask_above_intrusion = (Wdqv_above_lev[i_int]-Wdqv_above_lev >= W_int/2).data
        # merge masks
        mask_intrusion = np.logical_not(np.logical_or(mask_above_intrusion,
                                                      mask_below_intrusion))
    
    #-- create resulting RH profile
    rh_out = rh_prof.copy()
    rh_out[mask_intrusion] = rh_max

    return rh_out

##---- Merge profiles for radiative calculation
    
def mergeProfile(index:int, name:str, z:np.array, pres:np.array, temp:np.array, qv:np.array, rh:np.array,
                  temp_ref:float, pres_ref:float,
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

    #- uniform temp and pres for fixed kappa
    tlay_ref = temp_ref*np.ones(temp_lay.shape)
    play_ref = pres_ref*np.ones(play.shape)

    #- store all
    profile = xr.Dataset({"index":([], index), \
                      "name":([],name),\
                      "tlay":(["play"], temp_lay), \
                      "play":(["play"], play), \
                      "zlay":(["play"], z_lay), \
                      "tlay_ref":(["play"], tlay_ref), \
                      "play_ref":(["play"], play_ref), \
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
    parser.add_argument("--position", type=str, default='centered',
                        help="Position of intrusions (below, centered, above)")
    parser.add_argument("--temp_ref", type=float, default=290,
                        help="Reference temperature for kappa")
    parser.add_argument("--pres_ref", type=float, default=80000,
                        help="Reference pressure for kappa")
    args = parser.parse_args()

    output_dir  = args.out_dir
    
    print("C\'est parti")

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
    
    # #-- Radiative features
    # features_filename = 'rad_features.pickle'
    # print('loading %s'%features_filename)
    # # load
    # features_path = os.path.join(resultinputdir,day,features_filename)
    # rad_features = pickle.load(open(features_path,'rb'))
    
    ## -- Idealized moist intrusion profiles
    mi_path = os.path.join(resultdir,scriptsubdir,'moist_intrusions.pickle')
    moist_intrusions = pickle.load(open(mi_path,'rb'))
    day_label = '20200213, lower'

    # coordinates
    pres = moist_intrusions[day_label]['profiles']['pres']
    z = moist_intrusions[day_label]['profiles']['z']
    
    #-- median profiles
    temp_med = moist_intrusions[day_label]['profiles']['temp']['med']
    qvstar_med = moist_intrusions[day_label]['profiles']['qvstar']['med'] 
    rh_med = moist_intrusions[day_label]['profiles']['rh']['med']
    qv_med = rh_med*qvstar_med
    
    #-- idealized temperature and moisture profiles
    # Linear fit T(z)
    print('> Linear temperature fit')
    mask = ~np.isnan(z) & ~np.isnan(temp_med)
    slope, intercept, r, p, se = linregress(z[mask],temp_med[mask])
    print('T(z) = %2.1f z + %3.1f, r = %1.2f'%(slope,intercept,r))
    temp_id = slope*z + intercept

    # power fit qvstar(p)
    print("> power qvstar fit in pressure")
    mask = ~np.isnan(pres) & ~np.isnan(qvstar_med)
    slope, intercept, r, p, se = linregress(np.log(pres[mask]),np.log(qvstar_med[mask]))
    print('ln(qvstar) = %2.1f ln(p) + %3.1f, r = %1.2f'%(slope,intercept,r))
    print('alpha = %1.2f, qvstar(1000hPa) = %1.2f'%(slope,slope*np.log(1000)+intercept))
    qvstar_id = np.exp(slope*np.log(pres) + intercept) 
    
    #-- create idealized RH profiles
    print("> Create idealized RH profiles")
    
    z_breaks_id = moist_intrusions[day_label]['fit']['z_breaks_id']
    z_jump_FT = z_breaks_id[1]
    i_jump_FT = np.where(z >= z_jump_FT)[0][0]
    
    # i_levmax = np.where(np.isnan(qvstar_id))[0][0] # assuming data is ordered from bottom to top
    # p_levmax = pres[i_levmax]
    p_levmax = 200 # hPa
    print('p_levmax =', p_levmax)
    
    rh_id = moist_intrusions[day_label]['fit']['rh_id']
    rh_remint = moist_intrusions[day_label]['fit']['rh_remint'] 
    rh_breaks_id = moist_intrusions[day_label]['fit']['rh_breaks_id']
    W_int = moist_intrusions[day_label]['stats']['W_int']
    p_int = moist_intrusions[day_label]['stats']['p_int_center']

    #1 reference profile (constant above gradual transition)
    print('1. reference profile')
    z_breaks_ref = z_breaks_id[:2]
    rh_breaks_ref = [rh_breaks_id[0],rh_breaks_id[-1]]
    rh_ref = piecewise_linear(z,z_breaks_ref,rh_breaks_ref)
    #2 reference + MI
    print('2. moist intrusion')
    rh_delta_int = rh_id - rh_remint
    rh_mi = rh_ref + rh_delta_int
    #3 reference + rectangle MI
    print("3. rectangle moist intrusion")
    rh_mi_rect = addRectangleIntrusionToProfile(pres,qvstar_med,rh_ref,W_int,p_int,rh_ref[0],where='centered')
    #4 reference + uniformly-redistributed MI
    print('4. vertically-redistributed intrusion')
    crh_above = computeWPaboveZ(qvstar_med*rh_mi,pres,p_levmax)[i_jump_FT]/computeWPaboveZ(qvstar_med,pres,p_levmax)[i_jump_FT]
    z_breaks_mi_uniform = z_breaks_ref 
    rh_breaks_mi_uniform = [rh_breaks_id[0],crh_above]
    rh_mi_uniform = piecewise_linear(z,z_breaks_mi_uniform,rh_breaks_mi_uniform)
    #5 RH uniform = RH_{BL}
    print('5. vertically-uniform profile')
    rh_uniform = rh_breaks_id[0]*np.ones(rh_ref.shape)
    
    
    
    ##--- Now automate creation of rectangle moist intrusions (and equivalent uniform RH)
    
    print("- Generate array of intrusion parameters (height and W)")
    print("for each height, also calculate the profiles with uniform RH")
    
    N_sample = 20
    
    # heights
    Hs = np.linspace(3,10,N_sample) # km
    
    # water paths
    p_jump_FT = pres[i_jump_FT]
    W_ref = waterPath(qvstar_id*rh_ref,pres,p_jump_FT,p_levmax)
    W_sat_FT = computeWPaboveZ(qvstar_med,pres,p_levmax)[i_jump_FT] 
    
    Wmin = 0.5 # mm
    Wmax = 6.5 # mm
    Ws = np.linspace(Wmin,Wmax,N_sample)
    
    # 1D arrays with all parameter combinations
    Hs_2D,Ws_2D = np.meshgrid(Hs,Ws)
    Hs_all = Hs_2D.flatten() # height of intrusion center of mass 
    Ws_all = Ws_2D.flatten() # water path within intrusion only
    
    ##--- Merge profiles with background file
    print('> merge all profiles with background')

    #- Initialize lists and start with median profile    
    temp_profiles = [temp_med]
    qv_profiles = [qv_med]
    rh_profiles = [rh_med]
    profile_names = ['median']
    
    #- Add all idealized profile
    for name in 'ref','mi','mi_rect','mi_uniform','uniform':
                
        rh = getattr(thismodule,'rh_%s'%(name))
        qv = qvstar_id*rh

        qv_profiles.append(qv)
        temp_profiles.append(temp_id)
        rh_profiles.append(rh)
        profile_names.append(name)

    #- Add constant RH profiles for all sample water paths
    
    for W in Ws:
        
        print('create constant RH profile for W = %2.2f'%W)
        
        # rh
        rh_min = (W+W_ref)/W_sat_FT
        z_breaks_new = z_breaks_ref
        rh_breaks_new = [rh_breaks_id[0],rh_min]
        rh = piecewise_linear(z,z_breaks_new,rh_breaks_new)
        rh_profiles.append(rh)
        # temp
        temp_profiles.append(temp_id)
        # qv
        qv = rh*qvstar_id
        qv_profiles.append(qv)
        # name
        profile_names.append('W_%2.2fmm_uniform_RH'%W)
    
    #- Add RH profiles with rectangle moist intrusions for all sample parameters
    
    for H,W in zip(Hs_all,Ws_all):
        
        print('create rectangular RH intrusion for W = %2.2f and H = %1.2f'%(W,H))
        
        # rh
        p_i = z2p(H,z,pres)
        # rh = addRectangleIntrusionToProfile(pres,qvstar_id,rh_ref,W,p_i,rh_ref[0],where='centered')
        rh = addRectangleIntrusionToProfile(pres,qvstar_id,rh_ref,W,p_i,rh_ref[0],where=args.position)
        rh_profiles.append(rh)
        # temp 
        temp_profiles.append(temp_id)
        # qv
        qv = rh*qvstar_id
        qv_profiles.append(qv)
        # name
        profile_names.append('W_%2.2fmm_H_%1.2fkm'%(W,H))
        

    #- loop and merge with background profile
    all_profiles = []
    N_prof = len(rh_profiles)
    
    for i in range(N_prof):

        # merge profiles
        prof = mergeProfile(i,profile_names[i],
                            z*1e3,pres*1e2,
                            temp_profiles[i],
                            qv_profiles[i],
                            rh_profiles[i],
                            deltaP=args.deltaP,
                            sfc_emis=args.emis,
                            sfc_alb=args.alb,
                            mu0=args.mu0,
                            background_file=args.background_file,
                            h2o_only=args.h2o_only,
                            temp_ref=args.temp_ref,
                            pres_ref=args.pres_ref)
        
        # store
        all_profiles.append(prof)

    
    ##-- Save profiles to disk
    
    N_prof = len(all_profiles)

    for i,profile in zip(range(N_prof),all_profiles):
                         
        output_file = "rrtmgp_" + '{:>04}'.format(i)+'.nc'
        print(output_file)
        profile.to_netcdf(os.path.join(output_dir, output_file))
    
        
        
    
