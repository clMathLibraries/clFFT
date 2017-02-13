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


#  client
set( Client.Source	client.cpp
                    openCL.misc.cpp
                    stdafx.cpp )

set( Client.Headers client.h
                    openCL.misc.h
                    ../statTimer/statisticalTimer.extern.h
                    ../include/unicode.compatibility.h
                    ../include/stdafx.h
                    ../include/targetver.h
                    ../include/clFFT.h )

set( Client.Files ${Client.Source} ${Client.Headers} )

set( RT_LIB "" )
if( WIN32 )
	add_definitions( "/D_CONSOLE" )
elseif( NOT APPLE )
	# To use the dlopen() and dlclose() functions, we should link with libdl
	set( RT_LIB "-lrt" )
endif( )

# Include standard OpenCL headers
include_directories( ${Boost_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS} ../../../common ${PROJECT_BINARY_DIR}/include ../include )

add_executable( clFFT-client ${Client.Files} )

target_link_libraries( clFFT-client clFFT ${Boost_LIBRARIES} ${OpenCL_LIBRARIES} ${CMAKE_DL_LIBS} ${RT_LIB} )

set_target_properties( clFFT-client PROPERTIES VERSION ${CLFFT_VERSION} )
set_target_properties( clFFT-client PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
if( APPLE )
    # properly deal with RPATH on mac
    set_target_properties( clFFT-client PROPERTIES INSTALL_RPATH "@loader_path/../lib${SUFFIX_LIB}")
endif()

install( TARGETS clFFT-client
        RUNTIME DESTINATION bin${SUFFIX_BIN}
        LIBRARY DESTINATION lib${SUFFIX_LIB}
        ARCHIVE DESTINATION lib${SUFFIX_LIB}/import
        )
