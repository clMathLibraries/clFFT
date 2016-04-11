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
#if !defined( CLFFT_TESTCONSTANTS_H )
#define CLFFT_TESTCONSTANTS_H

#include "clFFT.h"
#include <string>
#include <stdexcept>

//Pre-callback function strings
#define PRE_MULVAL float2 mulval_pre(__global void* in, uint offset, __global void* userdata)\n \
				{ \n \
				float scalar = *((__global float*)userdata + offset); \n \
				float2 ret = *((__global float2*)in + offset) * scalar; \n \
				return ret; \n \
				}

#define PRE_MULVAL_UDT typedef struct USER_DATA  \
					   {  \
						float scalar1;  \
						float scalar2;  \
						} USER_DATA; \n \
					float2 mulval_pre(__global void* in, uint offset, __global void* userdata)\n \
					{ \n \
					__global USER_DATA *data = ((__global USER_DATA *)userdata + offset); \n \
					float scalar = data->scalar1 * data->scalar2; \n \
					float2 ret = *((__global float2*)in + offset) * scalar; \n \
					return ret; \n \
					}

#define PRE_MULVAL_DP double2 mulval_pre(__global void* in, uint offset, __global void* userdata)\n \
				{ \n \
				double scalar = *((__global double*)userdata + offset); \n \
				double2 ret = *((__global double2*)in + offset) * scalar; \n \
				return ret; \n \
				}

#define PRE_MULVAL_PLANAR float2 mulval_pre(__global void* inRe, __global void* inIm, uint offset, __global void* userdata)\n \
				{ \n \
				float scalar = *((__global float*)userdata + offset); \n \
				float2 ret; \n \
				ret.x = *((__global float*)inRe + offset) * scalar; \n \
				ret.y = *((__global float*)inIm + offset) * scalar; \n \
				return ret; \n \
				}

#define PRE_MULVAL_PLANAR_DP double2 mulval_pre(__global void* inRe, __global void* inIm, uint offset, __global void* userdata)\n \
				{ \n \
				double scalar = *((__global double*)userdata + offset); \n \
				double2 ret; \n \
				ret.x = *((__global double*)inRe + offset) * scalar; \n \
				ret.y = *((__global double*)inIm + offset) * scalar; \n \
				return ret; \n \
				}

#define PRE_MULVAL_REAL float mulval_pre(__global void* in, uint offset, __global void* userdata)\n \
				{ \n \
				float scalar = *((__global float*)userdata + offset); \n \
				float ret = *((__global float*)in + offset) * scalar; \n \
				return ret; \n \
				}

#define PRE_MULVAL_REAL_DP double mulval_pre(__global void* in, uint offset, __global void* userdata)\n \
				{ \n \
				double scalar = *((__global double*)userdata + offset); \n \
				double ret = *((__global double*)in + offset) * scalar; \n \
				return ret; \n \
				}

//Precallback test for LDS - works when 1 WI works on one input element
#define PRE_MULVAL_LDS float2 mulval_pre(__global void* in, uint offset, __global void* userdata, __local void* localmem)\n \
				{ \n \
				uint lid = get_local_id(0); \n \
				__local float* lds = (__local float*)localmem + lid; \n \
				lds[0] = *((__global float*)userdata + offset); \n \
				barrier(CLK_LOCAL_MEM_FENCE); \n \
				float prev = offset <= 0 ? 0 : *(lds - 1); \n \
				float next = offset >= get_global_size(0) ? 0 : *(lds + 1); \n \
				float avg = (prev + *lds + next)/3.0f;\n \
				float2 ret = *((__global float2*)in + offset) * avg; \n \
				return ret; \n \
				}

//Post-callback function strings
#define POST_MULVAL void mulval_post(__global void *output, uint outoffset, __global void *userdata, float2 fftoutput )\n \
				{ \n \
				float scalar = *((__global float*)userdata + outoffset); \n \
				*((__global float2*)output + outoffset) = fftoutput * scalar; \n \
				}

#define POST_MULVAL_DP void mulval_post(__global void *output, uint outoffset, __global void *userdata, double2 fftoutput )\n \
				{ \n \
				double scalar = *((__global double*)userdata + outoffset); \n \
				*((__global double2*)output + outoffset) = fftoutput * scalar; \n \
				}

#define POST_MULVAL_PLANAR void mulval_post(__global void *outputRe, __global void *outputIm, size_t outoffset, __global void *userdata, float fftoutputRe, float fftoutputIm )\n \
				{ \n \
				float scalar = *((__global float*)userdata + outoffset); \n \
				*((__global float*)outputRe + outoffset) = fftoutputRe * scalar; \n \
				*((__global float*)outputIm + outoffset) = fftoutputIm * scalar; \n \
				}

#define POST_MULVAL_PLANAR_DP void mulval_post(__global void *outputRe, __global void *outputIm, size_t outoffset, __global void *userdata, double fftoutputRe, double fftoutputIm )\n \
				{ \n \
				double scalar = *((__global double*)userdata + outoffset); \n \
				*((__global double*)outputRe + outoffset) = fftoutputRe * scalar; \n \
				*((__global double*)outputIm + outoffset) = fftoutputIm * scalar; \n \
				}

//Postcallback test for LDS - works when 1 WI works on one element. 
//Assumes 1D FFT of length 64.
#define POST_MULVAL_LDS void mulval_post(__global void *output, uint outoffset, __global void *userdata, float2 fftoutput, __local void* localmem)\n \
				{ \n \
				uint lid = get_local_id(0); \n \
				__local float* lds; \n \
				if (outoffset < 16) \n \
				{ \n \
				lds  = (__local float*)localmem + lid*4; \n \
				lds[0] = *((__global float*)userdata + lid*4); \n \
				lds[1] = *((__global float*)userdata + lid*4 + 1); \n \
				lds[2] = *((__global float*)userdata + lid*4 + 2); \n \
				lds[3] = *((__global float*)userdata + lid*4 + 3); \n \
				} \n \
				barrier(CLK_LOCAL_MEM_FENCE); \n \
				lds  = (__local float*)localmem + outoffset; \n \
				float prev = outoffset <= 0 ? 0 : *(lds - 1); \n \
				float next = outoffset >= (get_global_size(0) - 1) ? 0 : *(lds + 1); \n \
				float avg = (prev + *lds + next)/3.0f; \n \
				*((__global float2*)output + outoffset) = fftoutput * avg; \n \
				}

#define POST_MULVAL_REAL void mulval_post(__global void *output, uint outoffset, __global void *userdata, float fftoutput )\n \
				{ \n \
				float scalar = *((__global float*)userdata + outoffset); \n \
				*((__global float*)output + outoffset) = fftoutput * scalar; \n \
				}

#define POST_MULVAL_REAL_DP void mulval_post(__global void *output, uint outoffset, __global void *userdata, double fftoutput )\n \
				{ \n \
				double scalar = *((__global double*)userdata + outoffset); \n \
				*((__global double*)output + outoffset) = fftoutput * scalar; \n \
				}

typedef struct USER_DATA
				{
				float scalar1;
				float scalar2;
				} USER_DATA;

#define CALLBCKSTR(...) #__VA_ARGS__
#define STRINGIFY(...) 	CALLBCKSTR(__VA_ARGS__)

enum { REAL=0, IMAG=1 };
enum { dimx=0, dimy=1, dimz=2 };
enum fftw_dim { one_d=1, two_d=2, three_d=3 };
enum { one_interleaved_buffer=1, separate_real_and_imaginary_buffers=2 };
const bool use_explicit_intermediate_buffer = true;
const bool autogenerate_intermediate_buffer = false;
const bool pointwise_compare = true;
const bool root_mean_square = false;
extern bool comparison_type;
extern bool suppress_output;

// this thing is horrible. horrible! i am not proud.
extern size_t super_duper_global_seed;

const size_t small2 = 32;
const size_t normal2 = 1024;
const size_t large2 = 8192;
const size_t dlarge2 = 4096;

const size_t small3 = 9;
const size_t normal3 = 729;
const size_t large3 = 6561;
const size_t dlarge3 = 2187;

const size_t small5 = 25;
const size_t normal5 = 625;
const size_t large5 = 15625;
const size_t dlarge5 = 3125;

const size_t small7 = 49;
const size_t normal7 = 343;
const size_t large7 = 16807;
const size_t dlarge7 = 2401;

const size_t large_batch_size = 2048;
const size_t do_not_output_any_mismatches = 0;
const size_t default_number_of_mismatches_to_output = 10;
const size_t max_dimension = 3;

const double magnitude_lower_limit = 1.0E-100;

extern float tolerance;
extern double rmse_tolerance;

extern cl_device_type g_device_type;
extern cl_int g_device_id;
extern cl_int g_platform_id;

extern size_t number_of_random_tests;
extern time_t random_test_parameter_seed;
extern bool verbose;

void handle_exception( const std::exception& except );
size_t max_mem_available_on_cl_device(size_t device_index);

// Creating this template function and specializations to control the length inputs to the tests;
// these should be removed once the size restriction on transfrom lengths (SP 2^24 and DP 2^22)
// is removed; the dlarge* constants can then be removed

template <typename T>
inline size_t MaxLength2D(size_t rad)
{
	return 0;
}

template <>
inline size_t MaxLength2D<float>(size_t rad)
{
	switch(rad)
	{
	case 2:		return large2;
	case 3:		return large3;
	case 5:		return large5;
	case 7:		return large7;
	default:	return 0;
	}
}

template <>
inline size_t MaxLength2D<double>(size_t rad)
{
	switch(rad)
	{
	case 2:		return dlarge2;
	case 3:		return dlarge3;
	case 5:		return dlarge5;
	case 7:		return dlarge7;
	default:	return 0;
	}
}


#endif
