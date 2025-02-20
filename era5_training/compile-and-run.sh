COMP=$1

if [[ ${COMP} == "intel" ]]; then
    FC=ifort
    FFLAGS=""

    # source /glade/u/home/tmeltzer/cam-test/debug_env.sh

    module purge
    module load cesmdev/1.0 ncarenv/23.06 craype/2.7.20 linaro-forge/23.0 intel/2023.0.0 mkl/2023.0.0
    module load ncarcompilers/1.0.0 cmake/3.26.3 cray-mpich/8.1.25 hdf5-mpi/1.12.2
    module load netcdf-mpi/4.9.2 parallel-netcdf/1.12.3 parallelio/2.6.2-debug esmf/8.6.0b04-debug
elif [[ ${COMP} == "gcc" ]]; then

    FC=gfortran
    FFLAGS="-ffree-line-length-none"

    module purge
    module load ncarenv/24.12 gcc/12.4.0 cmake cuda/12.3.2 netcdf/4.9.3
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    NC='\033[0m' # No Color
    echo -e "${RED}ERROR:${YELLOW} required option missing. Please specify [${GREEN}gcc${YELLOW}] or [${GREEN}intel${YELLOW}] as compiler.${NC}"
    exit 1
fi

source ../.nlgw/bin/activate

FTORCH_ROOT="/glade/u/home/tmeltzer/FTorch/bin/ftorch_${COMP}"
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

# gdb -q --args ./infer.exe attention test-data/ .
./infer.exe attention test-data/ .
echo
echo "========================================="
echo
./infer.exe ann test-data/ .
