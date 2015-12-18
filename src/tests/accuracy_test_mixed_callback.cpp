/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
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

#include <gtest/gtest.h>
#include<math.h>

#include "test_constants.h"
#include "fftw_transform.h"
#include "cl_transform.h"
#include "typedefs.h"
#include "accuracy_test_common.h"
#include <stdexcept>
#include <vector>

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_callback_single : public ::testing::Test {
protected:
	accuracy_test_callback_single(){}
	virtual ~accuracy_test_callback_single(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

class accuracy_test_precallback_double : public ::testing::Test {
protected:
	accuracy_test_precallback_double(){}
	virtual ~accuracy_test_precallback_double(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

class mixed_radix_precallback : public ::testing::TestWithParam<size_t> {
	protected:
		mixed_radix_precallback(){}
		virtual ~mixed_radix_precallback(){}
		virtual void SetUp(){}
		virtual void TearDown(){}
};

class mixed_radix_postcallback : public ::testing::TestWithParam<size_t> {
	protected:
		mixed_radix_postcallback(){}
		virtual ~mixed_radix_postcallback(){}
		virtual void SetUp(){}
		virtual void TearDown(){}
};

class Supported_Fft_Sizes_Callback
{
public:
	std::vector<size_t> sizes;
	const size_t max_mixed_radices_to_test;

	Supported_Fft_Sizes_Callback()
	: max_mixed_radices_to_test( 4096 )
	{
		size_t i=0, j=0, k=0, l=0;
		size_t sum, sumi, sumj, sumk, suml;

		sumi = 1; i = 0;
		while(1)
		{
			sumj = 1; j = 0;
			while(1)
			{
				sumk = 1; k = 0;
				while(1)
				{
					suml = 1; l = 0;
					while(1)
					{
						sum = (sumi*sumj*sumk*suml);
						if( sum > max_mixed_radices_to_test ) break;

						sizes.push_back(sum);
						l++;
						suml *= 2;
					}

					if(l == 0) break;
					k++;
					sumk *= 3;
				}

				if( (k == 0) && (l == 0) ) break;
				j++;
				sumj *= 5;
			}

			if( (j == 0) && (k == 0) && (l == 0) ) break;
			i++;
			sumi *= 7;
		}
	}
} supported_sizes_callback;

INSTANTIATE_TEST_CASE_P(
	mixed_radices_precallback,
	mixed_radix_precallback,
	::testing::ValuesIn( supported_sizes_callback.sizes )
);

INSTANTIATE_TEST_CASE_P(
	mixed_radices_postcallback,
	mixed_radix_postcallback,
	::testing::ValuesIn( supported_sizes_callback.sizes )
);

namespace callback_mixed
{

/**********************************************************************************************
**************************************Complex To Complex***************************************
**********************************************************************************************/
#pragma region Complex_To_Complex

template< typename T, typename cl_T, typename fftw_T >
void mixed_radix_complex_to_complex_precallback( size_t problem_size )
{
	try
	{
		if(verbose) std::cout << "Now testing problem size " << problem_size << std::endl;

		std::vector<size_t> lengths;
		lengths.push_back( problem_size );
		size_t batch = 1;

		std::vector<size_t> input_strides;
		std::vector<size_t> output_strides;

		size_t input_distance = 0;
		size_t output_distance = 0;

		layout::buffer_layout_t in_layout = layout::complex_planar;
		layout::buffer_layout_t out_layout = layout::complex_planar;

		placeness::placeness_t placeness = placeness::in_place;

		direction::direction_t direction = direction::forward;

		data_pattern pattern = sawtooth;
		precallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
	}
	catch( const std::exception& err ) {
		handle_exception(err);
	}
}

TEST_P( mixed_radix_precallback, single_precision_complex_to_complex_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_complex_to_complex_precallback<float, cl_float, fftwf_complex>(problem_size);
}

TEST_P( mixed_radix_precallback, double_precision_complex_to_complex_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_complex_to_complex_precallback<double, cl_double, fftw_complex>(problem_size);
}

template< typename T, typename cl_T, typename fftw_T >
void mixed_radix_complex_to_complex_postcallback( size_t problem_size )
{
	try
	{
		if(verbose) std::cout << "Now testing problem size " << problem_size << std::endl;

		std::vector<size_t> lengths;
		lengths.push_back( problem_size );
		size_t batch = 1;

		std::vector<size_t> input_strides;
		std::vector<size_t> output_strides;

		size_t input_distance = 0;
		size_t output_distance = 0;

		layout::buffer_layout_t in_layout = layout::complex_planar;
		layout::buffer_layout_t out_layout = layout::complex_planar;

		placeness::placeness_t placeness = placeness::in_place;

		direction::direction_t direction = direction::forward;

		data_pattern pattern = sawtooth;
		postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
	}
	catch( const std::exception& err ) {
		handle_exception(err);
	}
}

TEST_P( mixed_radix_postcallback, single_precision_complex_to_complex_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_complex_to_complex_postcallback<float, cl_float, fftwf_complex>(problem_size);
}

TEST_P( mixed_radix_postcallback, double_precision_complex_to_complex_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_complex_to_complex_postcallback<double, cl_double, fftw_complex>(problem_size);
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void precall_normal_1D_forward_in_place_complex_to_complex_userdatatype()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = impulse;
	precallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 1.0f, true );
}

TEST_F(accuracy_test_callback_single, precall_normal_1D_forward_in_place_complex_to_complex_userdatatype)
{
	try { precall_normal_1D_forward_in_place_complex_to_complex_userdatatype< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

//Precallback with LDS
template< class T, class cl_T, class fftw_T >
void precall_lds_1D_forward_64_in_place_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 64 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = impulse;
	precallback_complex_to_complex_lds<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_callback_single, precall_lds_1D_forward_64_in_place_complex_to_complex)
{
	try { precall_lds_1D_forward_64_in_place_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

//Postcallback with LDS
template< class T, class cl_T, class fftw_T >
void postcall_lds_1D_forward_64_in_place_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 64 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = impulse;
	postcallback_complex_to_complex_lds<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_callback_single, postcall_lds_1D_forward_64_in_place_complex_to_complex)
{
	try { postcall_lds_1D_forward_64_in_place_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pre_and_post_callback_lds_1D_forward_64_in_place_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 64 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = impulse;
	pre_and_post_callback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 1.0f, true );
}

TEST_F(accuracy_test_callback_single, pre_and_post_callback_lds_1D_forward_64_in_place_complex_to_complex)
{
	try { pre_and_post_callback_lds_1D_forward_64_in_place_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

#pragma endregion

/**********************************************************************************************
**************************************Complex To Real***************************************
**********************************************************************************************/
#pragma region Complex_To_Real

template< typename T, typename cl_T, typename fftw_T >
void mixed_radix_hermitian_to_real_precallback( size_t problem_size )
{
	try
	{
		if(verbose) std::cout << "Now testing problem size " << problem_size << std::endl;

		std::vector<size_t> lengths;
		lengths.push_back( problem_size );
		size_t batch = 1;

		std::vector<size_t> input_strides;
		std::vector<size_t> output_strides;

		size_t input_distance = 0;
		size_t output_distance = 0;

		layout::buffer_layout_t layout = layout::hermitian_interleaved;

		placeness::placeness_t placeness = placeness::in_place;

		data_pattern pattern = sawtooth;
		precallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
	}
	catch( const std::exception& err ) {
		handle_exception(err);
	}
}

TEST_P( mixed_radix_precallback, single_precision_hermitian_to_real_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_hermitian_to_real_precallback<float, cl_float, fftwf_complex>(problem_size);
}

TEST_P( mixed_radix_precallback, double_precision_hermitian_to_real_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_hermitian_to_real_precallback<double, cl_double, fftw_complex>(problem_size);
}

template< typename T, typename cl_T, typename fftw_T >
void mixed_radix_hermitian_to_real_postcallback( size_t problem_size )
{
	try
	{
		if(verbose) std::cout << "Now testing problem size " << problem_size << std::endl;

		std::vector<size_t> lengths;
		lengths.push_back( problem_size );
		size_t batch = 1;

		std::vector<size_t> input_strides;
		std::vector<size_t> output_strides;

		size_t input_distance = 0;
		size_t output_distance = 0;

		layout::buffer_layout_t layout = layout::hermitian_interleaved;

		placeness::placeness_t placeness = placeness::in_place;

		data_pattern pattern = sawtooth;
		postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
	}
	catch( const std::exception& err ) {
		handle_exception(err);
	}
}

TEST_P( mixed_radix_postcallback, single_precision_hermitian_to_real_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_hermitian_to_real_postcallback<float, cl_float, fftwf_complex>(problem_size);
}

TEST_P( mixed_radix_postcallback, double_precision_hermitian_to_real_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_hermitian_to_real_postcallback<double, cl_double, fftw_complex>(problem_size);
}

#pragma endregion

/**********************************************************************************************
**************************************Real To Complex***************************************
**********************************************************************************************/
#pragma region Real_To_Complex

template< typename T, typename cl_T, typename fftw_T >
void mixed_radix_real_to_hermitian_precallback( size_t problem_size )
{
	try
	{
		if(verbose) std::cout << "Now testing problem size " << problem_size << std::endl;

		std::vector<size_t> lengths;
		lengths.push_back( problem_size );
		size_t batch = 1;

		std::vector<size_t> input_strides;
		std::vector<size_t> output_strides;

		size_t input_distance = 0;
		size_t output_distance = 0;

		layout::buffer_layout_t layout = layout::hermitian_interleaved;

		placeness::placeness_t placeness = placeness::in_place;

		data_pattern pattern = sawtooth;
		precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
	}
	catch( const std::exception& err ) {
		handle_exception(err);
	}
}

TEST_P( mixed_radix_precallback, single_precision_real_to_hermitian_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_real_to_hermitian_precallback<float, cl_float, fftwf_complex>(problem_size);
}

TEST_P( mixed_radix_precallback, double_precision_real_to_hermitian_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_real_to_hermitian_precallback<double, cl_double, fftw_complex>(problem_size);
}

template< typename T, typename cl_T, typename fftw_T >
void mixed_radix_real_to_hermitian_postcallback( size_t problem_size )
{
	try
	{
		if(verbose) std::cout << "Now testing problem size " << problem_size << std::endl;

		std::vector<size_t> lengths;
		lengths.push_back( problem_size );
		size_t batch = 1;

		std::vector<size_t> input_strides;
		std::vector<size_t> output_strides;

		size_t input_distance = 0;
		size_t output_distance = 0;

		layout::buffer_layout_t layout = layout::hermitian_interleaved;

		placeness::placeness_t placeness = placeness::in_place;

		data_pattern pattern = sawtooth;
		postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
	}
	catch( const std::exception& err ) {
		handle_exception(err);
	}
}

TEST_P( mixed_radix_postcallback, single_precision_real_to_hermitian_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_real_to_hermitian_postcallback<float, cl_float, fftwf_complex>(problem_size);
}

TEST_P( mixed_radix_postcallback, double_precision_real_to_hermitian_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_real_to_hermitian_postcallback<double, cl_double, fftw_complex>(problem_size);
}

#pragma endregion

}
