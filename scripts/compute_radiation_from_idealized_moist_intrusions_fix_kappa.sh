#!/bin/bash

# Uses RRTMGP code by Robert Pincus to calculate
# radiative profiles from idealized moisture profiles

#conda activate pyLMD

wdir=${PWD%/*}
cdir=${wdir}/code

#- First select the right radiation script
rad_script=sonde_radiation_uniform_gas_optics

cd $cdir
cp ${rad_script} sonde_radiation
cd -

#- Then create profiles and compute radiative profile

# odir=${wdir}/output/rad_profiles/idealized_profiles_moist_intrusions_20200213
odir=${wdir}/output/rad_profiles/idealized_profiles_moist_intrusions_20200213lower_fix_k
h2o_only=False

echo "-- generate idealized moisture profiles"
# python createIdealizedMoistureProfiles_MoistIntrusions_20200213.py --out_dir=${odir}/${subdir} --h2o_only=${h2o_only}
python createIdealizedMoistureProfiles_MoistIntrusions_20200213lower.py --temp_ref=290 --pres_ref=80000 --out_dir=${odir}/${subdir} --h2o_only=${h2o_only}

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
