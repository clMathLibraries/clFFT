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


#include <gtest/gtest.h>
#include<math.h>

#include "test_constants.h"
#include "fftw_transform.h"
#include "cl_transform.h"
#include "accuracy_test_common.h"
#include <stdexcept>
#include <vector>

class mixed_radix : public ::testing::TestWithParam<size_t> {
	protected:
		mixed_radix(){}
		virtual ~mixed_radix(){}
		virtual void SetUp(){}
		virtual void TearDown(){}
};

template< typename T, typename cl_T, typename fftw_T >
void mixed_radix_complex_to_complex( size_t problem_size )
{
	try
	{
		if(verbose) std::cout << "Now testing problem size " << problem_size << std::endl;

		std::vector<size_t> lengths;
		lengths.push_back( problem_size );
		size_t batch = 500;

		std::vector<size_t> input_strides;
		std::vector<size_t> output_strides;

		size_t input_distance = 0;
		size_t output_distance = 0;

		layout::buffer_layout_t in_layout = layout::complex_planar;
		layout::buffer_layout_t out_layout = layout::complex_planar;

		placeness::placeness_t placeness = placeness::in_place;

		direction::direction_t direction = direction::forward;

		data_pattern pattern = sawtooth;
		complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
	}
	catch( const std::exception& err ) {
		handle_exception(err);
	}
}

TEST_P( mixed_radix, single_precision_complex_to_complex_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_complex_to_complex<float, cl_float, fftwf_complex>(problem_size);
}

TEST_P( mixed_radix, double_precision_complex_to_complex_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_complex_to_complex<double, cl_double, fftw_complex>(problem_size);
}

template< typename T, typename cl_T, typename fftw_T >
void mixed_radix_real_to_hermitian( size_t problem_size )
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
		real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
	}
	catch( const std::exception& err ) {
		handle_exception(err);
	}
}

TEST_P( mixed_radix, single_precision_real_to_hermitian_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_real_to_hermitian<float, cl_float, fftwf_complex>(problem_size);
}

TEST_P( mixed_radix, double_precision_real_to_hermitian_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_real_to_hermitian<double, cl_double, fftw_complex>(problem_size);
}

template< typename T, typename cl_T, typename fftw_T >
void mixed_radix_hermitian_to_real( size_t problem_size )
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
		complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
	}
	catch( const std::exception& err ) {
		handle_exception(err);
	}
}

TEST_P( mixed_radix, single_precision_hermitian_to_real_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_hermitian_to_real<float, cl_float, fftwf_complex>(problem_size);
}

TEST_P( mixed_radix, double_precision_hermitian_to_real_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_hermitian_to_real<double, cl_double, fftw_complex>(problem_size);
}

class Supported_Fft_Sizes
{
public:
	std::vector<size_t> sizes;
	const size_t max_mixed_radices_to_test;

	Supported_Fft_Sizes()
	: max_mixed_radices_to_test( 4096 )
	{
		size_t i=0, j=0, k=0, l=0, m=0, n=0;
		size_t sum, sumi, sumj, sumk, suml, summ, sumn;

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
						summ = 1; m = 0;
						while (1)
						{
							sumn = 1; n = 0;
							while (1)
							{
								sum = (sumi*sumj*sumk*suml*summ*sumn);
								if (sum > max_mixed_radices_to_test) break;

								sizes.push_back(sum);
								n++;
								sumn *= 2;
							}

							if(n == 0) break;
							m++;
							summ *= 3;
						}

						if( (m == 0) && (n == 0) ) break;
						l++;
						suml *= 5;
					}

					if( (l == 0) && (m == 0) && (n == 0) ) break;
					k++;
					sumk *= 7;
				}

				if( (k == 0) && (l == 0) && (m == 0) && (n == 0) ) break;
				j++;
				sumj *= 11;
			}

			if( (j == 0) &&  (k == 0) && (l == 0) && (m == 0) && (n == 0) ) break;
			i++;
			sumi *= 13;
		}
	}
} supported_sizes;

INSTANTIATE_TEST_CASE_P(
	mixed_radices,
	mixed_radix,
	::testing::ValuesIn( supported_sizes.sizes )
);

 // ============================================== //
 // the following is a place to stick static tests //
 // with mixed radices. the tests will most likely //
 // be created in response to failed random tests. //
 // ============================================== //

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_mixed_single : public ::testing::Test {
protected:
	accuracy_test_mixed_single(){}
	virtual ~accuracy_test_mixed_single(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_mixed_double : public ::testing::Test {
protected:
	accuracy_test_mixed_double(){}
	virtual ~accuracy_test_mixed_double(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void hermitian_to_real_transforms_with_non_unit_output_strides_should_pass()
{
	std::vector<size_t> lengths;
	lengths.push_back( 10 );
	size_t batch = 1;

	std::vector<size_t> input_strides;
	size_t input_distance = 0;

	std::vector<size_t> output_strides;
	output_strides.push_back( 2 );
	size_t output_distance = 0;

	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_mixed_single, hermitian_to_real_transforms_with_non_unit_output_strides_should_pass)
{
	try { hermitian_to_real_transforms_with_non_unit_output_strides_should_pass< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_mixed_double, hermitian_to_real_transforms_with_non_unit_output_strides_should_pass)
{
	try { hermitian_to_real_transforms_with_non_unit_output_strides_should_pass< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void hermitian_to_real_transforms_with_non_unit_input_strides_should_pass()
{
	std::vector<size_t> lengths;
	lengths.push_back( 6 );
	lengths.push_back( 67500 );
	size_t batch = 1;

	std::vector<size_t> input_strides;
	input_strides.push_back( 2 );
	input_strides.push_back( 12 );
	size_t input_distance = 810074;

	std::vector<size_t> output_strides;
	size_t output_distance = 0;

	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_mixed_single, hermitian_to_real_transforms_with_non_unit_input_strides_should_pass)
{
	try { hermitian_to_real_transforms_with_non_unit_input_strides_should_pass< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_mixed_double, hermitian_to_real_transforms_with_non_unit_input_strides_should_pass)
{
	try { hermitian_to_real_transforms_with_non_unit_input_strides_should_pass< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_targeted_real_to_hermitian_transform()
{
	std::vector<size_t> lengths;
	lengths.push_back( 15 );
	lengths.push_back( 2 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	input_strides.push_back( 1 );
	input_strides.push_back( 16 );
	size_t input_distance = 32;

	std::vector<size_t> output_strides;
	output_strides.push_back( 1 );
	output_strides.push_back( 8 );
	size_t output_distance = 16;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_mixed_single, small_targeted_real_to_hermitian_transform)
{
	try { small_targeted_real_to_hermitian_transform< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_mixed_double, small_targeted_real_to_hermitian_transform)
{
	try { small_targeted_real_to_hermitian_transform< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void larger_targeted_real_to_hermitian_transform()
{
	std::vector<size_t> lengths;
	lengths.push_back( 15 );
	lengths.push_back( 4500 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	input_strides.push_back( 1 );
	input_strides.push_back( 16 );
	size_t input_distance = 72000;

	std::vector<size_t> output_strides;
	output_strides.push_back( 1 );
	output_strides.push_back( 8 );
	size_t output_distance = 36000;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_mixed_single, larger_targeted_real_to_hermitian_transform)
{
	try { larger_targeted_real_to_hermitian_transform< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_mixed_double, larger_targeted_real_to_hermitian_transform)
{
	try { larger_targeted_real_to_hermitian_transform< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void another_targeted_real_to_hermitian_transform()
{
	std::vector<size_t> lengths;
	lengths.push_back( 30 );
	lengths.push_back( 10125 );
	size_t batch = 1;

	std::vector<size_t> input_strides;
	input_strides.push_back( 1 );
	input_strides.push_back( 32 );
	size_t input_distance = 324000;

	std::vector<size_t> output_strides;
	output_strides.push_back( 1 );
	output_strides.push_back( 16 );
	size_t output_distance = 162000;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_mixed_single, another_targeted_real_to_hermitian_transform)
{
	try { another_targeted_real_to_hermitian_transform< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_mixed_double, another_targeted_real_to_hermitian_transform)
{
	try { another_targeted_real_to_hermitian_transform< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void possible_driver_bug_1D_length_375_fails()
{
	std::vector<size_t> lengths;
	lengths.push_back( 375 );
	size_t batch = 1;

	std::vector<size_t> input_strides;
	size_t input_distance = 0;

	std::vector<size_t> output_strides;
	size_t output_distance = 0;

	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_mixed_single, possible_driver_bug_1D_length_375_fails)
{
	try { possible_driver_bug_1D_length_375_fails< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_mixed_double, possible_driver_bug_1D_length_375_fails)
{
	try { possible_driver_bug_1D_length_375_fails< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}