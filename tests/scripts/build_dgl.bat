@ECHO OFF
SETLOCAL EnableDelayedExpansion

CALL "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
CALL mkvirtualenv --system-site-packages %BUILD_TAG%
DEL /S /Q build
DEL /S /Q _download
MD build

SET _MSPDBSRV_ENDPOINT_=%BUILD_TAG%
SET TMP=%WORKSPACE%\tmp
SET TEMP=%WORKSPACE%\tmp
SET TMPDIR=%WORKSPACE%\tmp

PUSHD build
cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DUSE_OPENMP=ON -DBUILD_TORCH=ON -Dgtest_force_shared_crt=ON -DDMLC_FORCE_SHARED_CRT=ON -DBUILD_CPP_TEST=1 -DCMAKE_CONFIGURATION_TYPES="Release" -DTORCH_PYTHON_INTERPS=python -DBUILD_SPARSE=ON .. -G "Visual Studio 16 2019" || EXIT /B 1
msbuild dgl.sln /m /nr:false || EXIT /B 1
COPY /Y Release\runUnitTests.exe .
POPD

CALL workon %BUILD_TAG%

PUSHD python
DEL /S /Q build *.egg-info dist
pip install -e . || EXIT /B 1
POPD

ENDLOCAL
EXIT /B
