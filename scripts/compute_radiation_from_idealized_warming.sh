#!/bin/bash

# Uses radiative transfer code by Robert Pincus to calculate
# radiative profiles from idealized moisture profiles

#conda activate pyLMD

wdir=${PWD%/*}
cdir=${wdir}/code

#- First select the right radiation script
rad_script=sonde_radiation_default

cd $cdir
cp ${rad_script} sonde_radiation
cd -

#- Then create profiles and compute radiative profile

odir=${wdir}/output/rad_profiles/idealized_profiles_idealized_warming
h2o_only=True

echo "-- generate idealized moisture profiles"
python createIdealizedMoistureProfiles_IdealizedWarming.py --out_dir=${odir}/${subdir} --h2o_only=${h2o_only}

echo
echo "-- Compute radiative profiles"

for ofile in `ls ${odir}/rrtmgp_0???.nc`; do
    echo $ofile
    ${cdir}/sonde_radiation $ofile
    #echo " "
done

echo 
echo "-- Post processing"

python post_processing_idealized_profiles.py --in_dir=${odir} --out_dir=${odir} --comp_qrad=True --interp=False

echo


echo "Calculation completed!"

exit 0
