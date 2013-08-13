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

const size_t large_batch_size = 2048;
const size_t do_not_output_any_mismatches = 0;
const size_t default_number_of_mismatches_to_output = 10;
const size_t max_dimension = 3;

const double magnitude_lower_limit = 1.0E-100;

extern float tolerance;

extern cl_device_type device_type;
extern cl_uint device_gpu_list;

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
	default:	return 0;
	}
}


#endif
