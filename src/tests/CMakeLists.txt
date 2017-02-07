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


# List the names of the files to compile
set( clFFT.Test.Source
	 test_constants.cpp
	 buffer_memory.cpp
	 buffer.cpp
	 unit_test.cpp
	 accuracy_test_common.cpp
	 accuracy_test_pow2.cpp
	 accuracy_test_pow3.cpp
	 accuracy_test_pow5.cpp
	 accuracy_test_pow7.cpp
	 accuracy_test_mixed_radices.cpp
	 accuracy_test_random.cpp
	 accuracy_test_directed.cpp
	 accuracy_test_mixed_callback.cpp
	 accuracy_test_pow2_precallback.cpp
	 accuracy_test_pow3_precallback.cpp
	 accuracy_test_pow5_precallback.cpp
	 accuracy_test_pow7_precallback.cpp
	 accuracy_test_postcallback.cpp
	 gtest_main.cpp
	 ${PROJECT_SOURCE_DIR}/client/openCL.misc.cpp
	 c-compliance.c
   )

set( clFFT.Test.Headers
	${PROJECT_SOURCE_DIR}/include/clFFT.h
	${PROJECT_SOURCE_DIR}/include/unicode.compatibility.h
	${PROJECT_SOURCE_DIR}/include/convenienceFunctions.h
	${PROJECT_SOURCE_DIR}/library/private.h
	${PROJECT_SOURCE_DIR}/client/openCL.misc.h
	accuracy_test_common.h
	test_constants.h
	buffer_memory.h
	buffer.h
	cl_transform.h
	fftw_transform.h
	typedefs.h
  )

set( clFFT.Test.Files ${clFFT.Test.Source} ${clFFT.Test.Headers} )

set( LD_PTHREAD "" )
if( MINGW )
    # -std=c++0x causes g++ to go into strict ANSI mode, which doesn't declare non-standard functions
    # Googletest for mingw appears to have a dependency on _stricmp and off64_t
	set( CMAKE_CXX_FLAGS "-std=gnu++0x ${CMAKE_CXX_FLAGS}" )
elseif( CMAKE_COMPILER_IS_GNUCXX )
	set( CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}" )
	set( LD_PTHREAD "-lpthread" )
elseif( APPLE )
	set( CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++ ${CMAKE_CXX_FLAGS}" )
	add_definitions( -DGTEST_USE_OWN_TR1_TUPLE )
endif( )

# Include standard OpenCL headers
include_directories( ${Boost_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}  ${PROJECT_BINARY_DIR}/include ${PROJECT_SOURCE_DIR}/include )

add_executable( Test ${clFFT.Test.Files} )

# If the runtime is being built by the project, use it, otherwise link to a runtime library specified in the install prefix
if( BUILD_RUNTIME )
	target_link_libraries( Test clFFT ${Boost_LIBRARIES} ${GTEST_LIBRARIES} ${FFTW_LIBRARIES} ${OpenCL_LIBRARIES} ${LD_PTHREAD} ${CMAKE_DL_LIBS})
else( )
	# Search for 64bit libs if FIND_LIBRARY_USE_LIB64_PATHS is set to true in the global environment, 32bit libs else
	get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )
	if( LIB64 )
		set( clFFT.library "${CMAKE_INSTALL_PREFIX}/lib64" )
	else( )
		set( clFFT.library "${CMAKE_INSTALL_PREFIX}/lib32" )
	endif( )

	if( WIN32 )
		set( clFFT.library "${clFFT.library}/import/clFFT${CMAKE_STATIC_LIBRARY_SUFFIX}" )
	else( )
		set( clFFT.library "${clFFT.library}/${CMAKE_SHARED_LIBRARY_PREFIX}clFFT${CMAKE_SHARED_LIBRARY_SUFFIX}" )
	endif( )

	target_link_libraries( Test ${clFFT.library} ${Boost_LIBRARIES} ${GTEST_LIBRARIES} ${FFTW_LIBRARIES} ${OpenCL_LIBRARIES} ${LD_PTHREAD} ${CMAKE_DL_LIBS})

endif( )

# The following set_target_properties is to get around a bug in cmake 2.8.2, where the suffix after the first '.' is dropped
IF( (MSVC_VERSION VERSION_EQUAL 1600) AND (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION} VERSION_LESS 2.8.3) )
	message( STATUS "Detected vs2010 and Cmake version less than 2.8.3; renaming Test with underscores " )
	set_target_properties( Test PROPERTIES OUTPUT_NAME "clFFT_Test" )
ENDIF( )

set_target_properties( Test PROPERTIES VERSION ${CLFFT_VERSION} )
set_target_properties( Test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

