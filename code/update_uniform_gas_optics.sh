#!/bin/bash


make clean
cp sonde_radiation_uniform_gas_optics.F90 sonde_radiation.F90
make
cp sonde_radiation sonde_radiation_uniform_gas_optics

