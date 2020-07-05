mkdir build

cd build

cmake .. -G "Visual Studio 16 2019" -DPARDISO_LIBRARIES="C:/local/pardiso/libpardiso600-WIN-X86-64.lib" -DLAPACK_LIBRARIES="C:/local/LAPACK/lib/liblapack.dll.a" -DBLAS_LIBRARIES="C:/local/LAPACK/lib/libblas.dll.a"

cmake --build .

cmake --build . --config release