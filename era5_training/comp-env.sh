module --force purge
module load cesmdev/1.0 ncarenv/23.06 craype/2.7.20 linaro-forge/23.0 intel/2023.0.0 mkl/2023.0.0
module load ncarcompilers/1.0.0 cmake/3.26.3 cray-mpich/8.1.25 hdf5-mpi/1.12.2
module load netcdf-mpi/4.9.2 parallel-netcdf/1.12.3 parallelio/2.6.2-debug esmf/8.6.0b04-debug

source ~/nonlocal_gwfluxes/.nlgw/bin/activate

export FTORCH_ROOT="$HOME/FTorch/bin/ftorch_intel"
export LD_LIBRARY_PATH="${FTORCH_ROOT}/lib64:${LD_LIBRARY_PATH}"
