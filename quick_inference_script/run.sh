NETCDFF_LIB=$(pkg-config --libs-only-L netcdf-fortran | sed "s/-L//")
FTORCH_ROOT="/software/ftorch/gfortran/"
export LD_LIBRARY_PATH="${FTORCH_ROOT}/lib:${NETCDFF_LIB}:LD_LIBRARY_PATH"

gfortran \
    -I${FTORCH_ROOT}/include/ftorch \
    $(pkg-config --cflags-only-I netcdf-fortran) \
    -g infer.f90 -o infer.exe \
    $(pkg-config --libs netcdf-fortran) \
    -L${FTORCH_ROOT}/lib/ -lftorch

./infer.exe
