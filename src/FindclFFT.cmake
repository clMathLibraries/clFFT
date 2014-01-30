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
#  This code is licensed under the MIT License.  See the FindclFFT.cmake script
#  for the text of the license.
#
# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#=============================================================================

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


