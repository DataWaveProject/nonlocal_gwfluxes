source ~/.envs/nlgw/bin/activate
spack env activate -p gpu-cuda

NETCDFF_LIB=$(pkg-config --libs-only-L netcdf-fortran | sed "s/-L//")
FTORCH_ROOT="/software/ftorch/gfortran/"
NVRT_LIB="/software/spack/opt/spack/linux-ubuntu24.04-skylake/gcc-13.3.0/cuda-12.6.3-dtxqxh2shhyofstcjzt5aoetovxk4yg6/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="${FTORCH_ROOT}/lib:${NETCDFF_LIB}:${NVRT_LIB}:LD_LIBRARY_PATH"
# echo LD_LIBRARY_PATH="${FTORCH_ROOT}/lib:${NETCDFF_LIB}:LD_LIBRARY_PATH"

gfortran \
    -O2 \
    -ffree-line-length-none \
    -I${FTORCH_ROOT}/include/ftorch \
    $(pkg-config --cflags-only-I netcdf-fortran) \
    -g infer.f90 -o infer.exe \
    $(pkg-config --libs netcdf-fortran) \
    -L${FTORCH_ROOT}/lib/ -lftorch

./infer.exe attention test-data/ .
echo
echo "========================================="
echo
./infer.exe ann test-data/ .
