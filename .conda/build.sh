#!/usr/bin/env sh

cd $RECIPE_DIR/../ || exit 1
mkdir -p build-umem-c
cd build-umem-c
export PATH=$BUILD_PREFIX/bin:$PATH
echo PREFIX=$PREFIX
echo BUILD_PREFIX=$BUILD_PREFIX
echo CONDA_PREFIX=$CONDA_PREFIX
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX ../c
make -j$CPU_COUNT
make test
ctest -D ExperimentalMemCheck -E test_cuda
make install

