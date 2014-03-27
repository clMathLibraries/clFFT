# ########################################################################
# Copyright 2013 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################


# Locate the FFTW (http://www.fftw.org/) Framework.
#
# Defines the following variables:
#
#   FFTW_FOUND - Found the FFTW framework
#   FFTW_INCLUDE_DIRS - Include directories
#
# Also defines the library variables below as normal
# variables.  These contain debug/optimized keywords when
# a debugging library is found.
#
#   FFTW_LIBRARIES - libfftw
#
# Accepts the following variables as input:
#
#   FFTW_ROOT - (as a CMake or environment variable)
#                The root directory of the fftw install prefix
#
#   FIND_LIBRARY_USE_LIB64_PATHS - Global property that controls whether 
#               findFFTW should search for 64bit or 32bit libs
#-----------------------------------------------
# Example Usage:
#
#    find_package(FFTW REQUIRED)
#    include_directories(${FFTW_INCLUDE_DIRS})
#
#    add_executable(foo foo.cc)
#    target_link_libraries(foo ${FFTW_LIBRARIES})
#
#-----------------------------------------------

find_path(FFTW_INCLUDE_DIRS
    NAMES fftw3.h
    HINTS
        ${FFTW_ROOT}/include
        ${FFTW_ROOT}/api
        ${FFTW_ROOT}
        $ENV{FFTW_ROOT}/include
        $ENV{FFTW_ROOT}/api
        ENV FFTW_ROOT
    PATHS
        /usr/include
        /usr/local/include
)
mark_as_advanced( FFTW_INCLUDE_DIRS )

find_library( FFTW_SINGLE_PRECISION_LIBRARIES
    NAMES fftw3f libfftw3f-3
    HINTS
        ${FFTW_ROOT}/lib
        ${FFTW_ROOT}/.libs
        ${FFTW_ROOT}
        $ENV{FFTW_ROOT}/lib
        $ENV{FFTW_ROOT}/.libs
        ENV FFTW_ROOT
    PATHS
        /usr/lib
        /usr/local/lib
    DOC "FFTW dynamic library"
)
mark_as_advanced( FFTW_SINGLE_PRECISION_LIBRARIES )

find_library( FFTW_DOUBLE_PRECISION_LIBRARIES
    NAMES fftw3 libfftw3-3
    HINTS
        ${FFTW_ROOT}/lib
        ${FFTW_ROOT}/.libs
        ${FFTW_ROOT}
        $ENV{FFTW_ROOT}/lib
        $ENV{FFTW_ROOT}/.libs
        ENV FFTW_ROOT
    PATHS
        /usr/lib
        /usr/local/lib
    DOC "FFTW dynamic library"
)
mark_as_advanced( FFTW_DOUBLE_PRECISION_LIBRARIES )

set( FFTW_LIBRARIES ${FFTW_SINGLE_PRECISION_LIBRARIES} ${FFTW_DOUBLE_PRECISION_LIBRARIES} )
mark_as_advanced( FFTW_LIBRARIES )

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( FFTW DEFAULT_MSG FFTW_LIBRARIES FFTW_INCLUDE_DIRS )

if( NOT FFTW_FOUND )
    message( STATUS "FindFFTW looked for single precision libraries named: fftw3f or libfftw3f-3" )
    message( STATUS "FindFFTW looked for double precision libraries named: fftw3 or libfftw3-3" )
endif()
