/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


#pragma once
#if !defined( CLIENT_H )
#define CLIENT_H

//	Boost headers that we want to use
//	#define BOOST_PROGRAM_OPTIONS_DYN_LINK
#include <boost/program_options.hpp>
#include "stdafx.h"
#include "../statTimer/statisticalTimer.extern.h"
#include "../include/unicode.compatibility.h"

#include <fftw3.h>

typedef unsigned char uint24_t[3]; 

#define CALLBCKSTR(...) #__VA_ARGS__
#define STRINGIFY(...) 	CALLBCKSTR(__VA_ARGS__)

#define BATCH_LENGTH 1024 

#define ConvertToFloat typedef unsigned char uint24_t[3]; \n \
						float convert24To32bit(__global void* in, uint inoffset, __global void* userdata) \n \
						{ \n \
						__global uint24_t* inData =  (__global uint24_t*)in; \n \
						float val = inData[inoffset][0] << 16 | inData[inoffset][1] << 8 | inData[inoffset][2] ; \n \
						return val;  \n \
						}

#define ConvertToFloat_KERNEL typedef unsigned char uint24_t[3]; \n \
							__kernel void convert24To32bit (__global void *input, __global void *output) \n \
							 { \n \
								uint inoffset = get_global_id(0); \n \
								__global uint24_t* inData =  (__global uint24_t*)input; \n \
								float val = inData[inoffset][0] << 16 | inData[inoffset][1] << 8 | inData[inoffset][2] ; \n \
								*((__global float*)output + inoffset) = val;  \n \
							} \n


template < typename T >
void R2C_transform(std::auto_ptr< clfftSetupData > setupData, size_t* inlengths, size_t batchSize, 
				   clfftDim dim, clfftPrecision precision,  cl_uint profile_count);

template < typename T >
void runR2CPrecallbackFFT(std::auto_ptr< clfftSetupData > setupData, cl_context context, cl_command_queue commandQueue,
						size_t* inlengths, clfftDim dim, clfftPrecision precision,
						size_t batchSize, size_t vectorLength, size_t fftLength, cl_uint profile_count);

template < typename T >
void runR2CPreprocessKernelFFT(std::auto_ptr< clfftSetupData > setupData, cl_context context, 
							cl_command_queue commandQueue, cl_device_id device_id,
							size_t* inlengths, clfftDim dim, clfftPrecision precision,
							size_t batchSize, size_t vectorLength, size_t fftLength, cl_uint profile_count);

fftwf_complex* get_R2C_fftwf_output(size_t* lengths, size_t fftbatchLength, int batch_size,
									clfftLayout in_layout, clfftDim dim);

template < typename T1, typename T2>
bool compare(T1 *refData, std::vector< std::complex< T2 > > data,
             size_t length, const float epsilon = 1e-6f);

#ifdef WIN32

struct Timer
{
    LARGE_INTEGER start, stop, freq;

public:
    Timer() { QueryPerformanceFrequency( &freq ); }

    void Start() { QueryPerformanceCounter(&start); }
    double Sample()
    {
        QueryPerformanceCounter  ( &stop );
        double time = (double)(stop.QuadPart-start.QuadPart) / (double)(freq.QuadPart);
        return time;
    }
};

#else

#include <time.h>
#include <math.h>

struct Timer
{
    struct timespec start, end;

public:
    Timer() { }

    void Start() { clock_gettime(CLOCK_MONOTONIC, &start); }
    double Sample()
    {
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time = 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
        return time * 1E-9;
    }
};

#endif

#endif
