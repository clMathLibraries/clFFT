# ########################################################################
# Copyright 2015 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)

#  client
set( cuFFT-client.Source	cuFFT-client.cpp )

set( cuFFT-client.Files ${cuFFT-client.Source} )

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_20,code=sm_20; -gencode arch=compute_30,code=sm_30; -gencode arch=compute_35,code=sm_35; -gencode arch=compute_37,code=sm_37; -gencode arch=compute_50,code=sm_50; -gencode arch=compute_52,code=sm_52)

include_directories( ./ ../common/inc/ )

CUDA_ADD_EXECUTABLE( cuFFT-client ${cuFFT-client.Files} )

CUDA_ADD_CUFFT_TO_TARGET( cuFFT-client )

target_link_libraries( cuFFT-client ${CUDA_LIBRARIES})

# Set output directory to bin
if( MSVC )
	set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BITNESS})
else()
	set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BITNESS}/${CMAKE_BUILD_TYPE})
endif()
