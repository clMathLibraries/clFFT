# - Find clFFT

# Defines the following variables:
# CLFFT_INCLUDE_DIRS    - Location of clFFT's include directory.
# CLFFT_LIBRARIES       - Location of clFFT's libraries
# CLFFT_FOUND           - True if clFFT has been located
#
# You may provide a hint to where clFFT's root directory may be located
# by setting CLFFT_ROOT_HINT before calling this script. This should point to
# the directory which contains 'include/clFFT.h'
#
# Variables used by this module, they can change the default behaviour and
# need to be set before calling find_package:
#
#=============================================================================
# Copyright 2014 Brian Kloppenborg
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

IF(CLFFT_INCLUDE_DIRS)
  # Already in cache, be silent
  set (CLFFT_FIND_QUIETLY TRUE)
ENDIF (CLFFT_INCLUDE_DIRS)

FIND_PATH(CLFFT_ROOT_DIR
    NAMES include/clFFT.h
    HINTS /usr/local/ ${CLFFT_ROOT_HINT} 
    DOC "clFFT root directory.")
    
FIND_PATH(_CLFFT_INCLUDE_DIRS
    NAMES clFFT.h
    HINTS ${CLFFT_ROOT_DIR}/include
    DOC "clFFT Include directory")

FIND_LIBRARY(_CLFFT_LIBRARY
    NAMES clFFT
    HINTS ${CLFFT_ROOT_DIR}/lib ${CLFFT_ROOT_DIR}/lib64)

SET(CLFFT_INCLUDE_DIRS ${_CLFFT_INCLUDE_DIRS})
SET(CLFFT_LIBRARIES ${_CLFFT_LIBRARY})

# handle the QUIETLY and REQUIRED arguments and set CLFFT_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(CLFFT DEFAULT_MSG CLFFT_LIBRARIES CLFFT_INCLUDE_DIRS)

MARK_AS_ADVANCED(CLFFT_LIBRARIES CLFFT_INCLUDE_DIRS)


