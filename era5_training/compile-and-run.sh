FC=ifort
FFLAGS=""

module --force purge
# these come from the environment listed in software_environment.txt in the CESM Case directory
module load cesmdev/1.0 ncarenv/23.06 craype/2.7.20 intel/2023.0.0 mkl/2023.0.0 ncarcompilers/1.0.0
module load cmake/3.26.3 cray-mpich/8.1.25 hdf5-mpi/1.12.2 netcdf-mpi/4.9.2 parallel-netcdf/1.12.3
module load parallelio/2.6.2 esmf/8.6.0b04

source ../.nlgw/bin/activate

FTORCH_ROOT="${HOME}/fresh/ftorch-install"
NETCDF_LIB="${NETCDF}/lib"
export LD_LIBRARY_PATH="${NETCDF_LIB}:${FTORCH_ROOT}/lib64:${LD_LIBRARY_PATH}"

COMMAND="${FC} \
    -O2 \
    ${FFLAGS} \
    -I${FTORCH_ROOT}/include/ftorch \
    $(pkg-config --cflags-only-I netcdf-fortran) \
    -g infer.f90 -o infer.exe \
    $(pkg-config --libs netcdf-fortran) \
    -L${FTORCH_ROOT}/lib64 -lftorch"

echo $COMMAND

${COMMAND}

./infer.exe attention test-data/ .
echo
echo "========================================="
echo
./infer.exe ann test-data/ .
