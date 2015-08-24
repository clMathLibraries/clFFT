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
class accuracy_test_precallback_single : public ::testing::Test {
protected:
	accuracy_test_precallback_single(){}
	virtual ~accuracy_test_precallback_single(){}
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

class Supported_Fft_Sizes_precallback
{
public:
	std::vector<size_t> sizes;
	const size_t max_mixed_radices_to_test;

	Supported_Fft_Sizes_precallback()
	: max_mixed_radices_to_test( 4096 )
	{
		size_t i=0, j=0, k=0;
		size_t sum, sumi, sumj, sumk;

		sumi = 1; i = 0;
		while(1)
		{
			sumj = 1; j = 0;
			while(1)
			{
				sumk = 1; k = 0;
				while(1)
				{
					sum = (sumi*sumj*sumk);
					if( sum > max_mixed_radices_to_test ) break;

					sizes.push_back(sum);
					k++;
					sumk *= 2;
				}

				if(k == 0) break;
				j++;
				sumj *= 3;
			}

			if( (j == 0) && (k == 0) ) break;
			i++;
			sumi *= 5;
		}
	}
} supported_sizes;

INSTANTIATE_TEST_CASE_P(
	mixed_radices_precallback,
	mixed_radix_precallback,
	::testing::ValuesIn( supported_sizes.sizes )
);

namespace precallback
{

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
		precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
	}
	catch( const std::exception& err ) {
		handle_exception(err);
	}
}

TEST_P( mixed_radix_precallback, single_precision_real_to_hermitian_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_real_to_hermitian<float, cl_float, fftwf_complex>(problem_size);
}

TEST_P( mixed_radix_precallback, double_precision_real_to_hermitian_auto_generated ) {
	size_t problem_size = GetParam();
	RecordProperty("problem_size", (int)problem_size);
	mixed_radix_real_to_hermitian<double, cl_double, fftw_complex>(problem_size);
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_in_place_hermitian_interleaved_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
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

TEST_F(accuracy_test_precallback_single, pow2_large_1D_in_place_hermitian_interleaved_to_real)
{
	try { pow2_large_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_large_1D_in_place_hermitian_interleaved_to_real)
{
	try { pow2_large_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_out_of_place_hermitian_planar_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_large_1D_out_of_place_hermitian_planar_to_real)
{
	try { pow2_large_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_large_1D_out_of_place_hermitian_planar_to_real)
{
	try { pow2_large_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
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
	precallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_forward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = impulse;
	precallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_normal_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow2_normal_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow2_normal_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
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
	precallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
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

	data_pattern pattern = sawtooth;
	precallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_forward_in_place_complex_to_complex_userdatatype()
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
	precallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, true );
}

TEST_F(accuracy_test_precallback_single, pow2_normal_1D_forward_in_place_complex_to_complex_userdatatype)
{
	try { pow2_normal_1D_forward_in_place_complex_to_complex_userdatatype< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_forward_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t out_layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, out_layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_normal_1D_forward_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_1D_forward_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_1D_forward_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_1D_forward_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_small_1D_forward_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t out_layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, out_layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_small_1D_forward_in_place_real_to_hermitian_interleaved)
{
	try { pow2_small_1D_forward_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_small_1D_forward_in_place_real_to_hermitian_interleaved)
{
	try { pow2_small_1D_forward_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_forward_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t out_layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, out_layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_large_1D_forward_in_place_real_to_hermitian_interleaved)
{
	try { pow2_large_1D_forward_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_large_1D_forward_in_place_real_to_hermitian_interleaved)
{
	try { pow2_large_1D_forward_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_4M_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4194304 );
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

TEST_F(accuracy_test_precallback_single, pow2_large_1D_4M_in_place_real_to_hermitian_interleaved)
{
	try { pow2_large_1D_4M_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_large_1D_4M_in_place_real_to_hermitian_interleaved)
{
	try { pow2_large_1D_4M_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_array_real_to_hermitian()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 8;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_normal_1D_array_real_to_hermitian)
{
	try { pow2_normal_1D_array_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_1D_array_real_to_hermitian)
{
	try { pow2_normal_1D_array_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_array_real_to_hermitian_with_odd_batch_size()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 5;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_normal_1D_array_real_to_hermitian_with_odd_batch_size)
{
	try { pow2_normal_1D_array_real_to_hermitian_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_1D_array_real_to_hermitian_with_odd_batch_size)
{
	try { pow2_normal_1D_array_real_to_hermitian_with_odd_batch_size< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_out_of_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_normal_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_1D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_normal_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow2_normal_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow2_normal_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_normal_1D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_precallback_single, pow3_normal_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow3_normal_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow3_normal_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow3_normal_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_small_1D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_precallback_single, pow3_small_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow3_small_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow3_small_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow3_small_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_large_1D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_precallback_single, pow3_large_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow3_large_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow3_large_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow3_large_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_normal_1D_out_of_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal3 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow3_normal_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow3_normal_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow3_normal_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow3_normal_1D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_normal_1D_array_real_to_hermitian_with_odd_batch_size()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal3 );
	size_t batch = 5;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow3_normal_1D_array_real_to_hermitian_with_odd_batch_size)
{
	try { pow3_normal_1D_array_real_to_hermitian_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow3_normal_1D_array_real_to_hermitian_with_odd_batch_size)
{
	try { pow3_normal_1D_array_real_to_hermitian_with_odd_batch_size< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 9 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	input_strides.push_back( 16 );

	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 128;

	std::vector<size_t> output_strides;
	output_strides.push_back( 2 );

	size_t output_distance = lengths[lengths.size()-1] * output_strides[output_strides.size()-1] + 2;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = impulse;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow3_very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { pow3_very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow3_very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { pow3_very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_very_small_1D_non_unit_stride_and_distance_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 9 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	input_strides.push_back( 2 );

	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 2;

	std::vector<size_t> output_strides( input_strides );
	size_t output_distance = input_distance;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = impulse;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow3_very_small_1D_non_unit_stride_and_distance_real_to_complex)
{
	try { pow3_very_small_1D_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow3_very_small_1D_non_unit_stride_and_distance_real_to_complex)
{
	try { pow3_very_small_1D_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 9 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	input_strides.push_back( 16 );

	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 128;

	std::vector<size_t> output_strides;
	output_strides.push_back( 2 );

	size_t output_distance = lengths[lengths.size()-1] * output_strides[output_strides.size()-1] + 2;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = impulse;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow3_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { pow3_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow3_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { pow3_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_normal_1D_user_defined_scale_real_to_hermitian()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal3 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = impulse;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness, 42.0f );
}

TEST_F(accuracy_test_precallback_single, pow3_normal_1D_user_defined_scale_real_to_hermitian)
{
	try { pow3_normal_1D_user_defined_scale_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow3_normal_1D_user_defined_scale_real_to_hermitian)
{
	try { pow3_normal_1D_user_defined_scale_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_normal_1D_round_trip_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal3 );
	size_t batch = 1;

	data_pattern pattern = impulse;
	precallback_real_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch );
}

TEST_F(accuracy_test_precallback_single, pow3_normal_1D_round_trip_real_to_complex)
{
	try { pow3_normal_1D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow3_normal_1D_round_trip_real_to_complex)
{
	try { pow3_normal_1D_round_trip_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_large_1D_round_trip_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( large3 );
	size_t batch = 1;

	data_pattern pattern = impulse;
	precallback_real_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch );
}

TEST_F(accuracy_test_precallback_single, pow3_large_1D_round_trip_real_to_complex)
{
	try { pow3_large_1D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow3_large_1D_round_trip_real_to_complex)
{
	try { pow3_large_1D_round_trip_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_2D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( normal2 );
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

TEST_F(accuracy_test_precallback_single, pow2_normal_2D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_2D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_2D_out_of_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_normal_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_2D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_2D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(2) );
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_large_2D_out_of_place_real_to_hermitian_planar)
{
	try { pow2_large_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_large_2D_out_of_place_real_to_hermitian_planar)
{
	try { pow2_large_2D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_small_2D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
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

TEST_F(accuracy_test_precallback_single, pow2_small_2D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_small_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_small_2D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_small_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_small_2D_non_unit_stride_and_distance_real_to_hermitian()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4 );
	lengths.push_back( 4 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	input_strides.push_back( 5 );
	input_strides.push_back( lengths[0] * input_strides[0] + 1 );

	std::vector<size_t> output_strides;
	output_strides.push_back( 2 );
	output_strides.push_back( lengths[0] * output_strides[0] + 2 );

	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 30;
	size_t output_distance = lengths[lengths.size()-1] * output_strides[output_strides.size()-1] + 42;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_small_2D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { pow2_small_2D_non_unit_stride_and_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_small_2D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { pow2_small_2D_non_unit_stride_and_distance_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void mixed_another_targeted_real_to_hermitian_transform()
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
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, mixed_another_targeted_real_to_hermitian_transform)
{
	try { mixed_another_targeted_real_to_hermitian_transform< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, mixed_another_targeted_real_to_hermitian_transform)
{
	try { mixed_another_targeted_real_to_hermitian_transform< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void mixed_larger_targeted_real_to_hermitian_transform()
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
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, mixed_larger_targeted_real_to_hermitian_transform)
{
	try { mixed_larger_targeted_real_to_hermitian_transform< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, mixed_larger_targeted_real_to_hermitian_transform)
{
	try { mixed_larger_targeted_real_to_hermitian_transform< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void mixed_1D_length_375_real_to_hermitian()
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
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, mixed_1D_length_375_real_to_hermitian)
{
	try { mixed_1D_length_375_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, mixed_1D_length_375_real_to_hermitian)
{
	try { mixed_1D_length_375_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_normal_1D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal5 );
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

TEST_F(accuracy_test_precallback_single, pow5_normal_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow5_normal_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow5_normal_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow5_normal_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_small_1D_out_of_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small5 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow5_small_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow5_small_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow5_small_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow5_small_1D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_large_1D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large5 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow5_large_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow5_large_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow5_large_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow5_large_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_normal_2D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal5 );
	lengths.push_back( normal5 );
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

TEST_F(accuracy_test_precallback_single, pow5_normal_2D_in_place_real_to_hermitian_interleaved)
{
	try { pow5_normal_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow5_normal_2D_in_place_real_to_hermitian_interleaved)
{
	try { pow5_normal_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_large_2D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(5) );
	lengths.push_back( normal5 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow5_large_2D_out_of_place_real_to_hermitian_planar)
{
	try { pow5_large_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow5_large_2D_out_of_place_real_to_hermitian_planar)
{
	try { pow5_large_2D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 25 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	input_strides.push_back( 16 );

	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 128;

	std::vector<size_t> output_strides;
	output_strides.push_back( 2 );

	size_t output_distance = lengths[lengths.size()-1] * output_strides[output_strides.size()-1] + 2;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = impulse;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow5_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { pow5_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow5_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { pow5_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_3D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( normal2 );
	lengths.push_back( small2 );
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

TEST_F(accuracy_test_precallback_single, pow2_normal_3D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_3D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_3D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_3D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_normal_3D_out_of_place_real_to_hermitian_planar)
{
	try { pow2_normal_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_normal_3D_out_of_place_real_to_hermitian_planar)
{
	try { pow2_normal_3D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_small_3D_out_of_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	precallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_precallback_single, pow2_small_3D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow2_small_3D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_small_3D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow2_small_3D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_3D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( 2 );
	lengths.push_back( 2 );
	lengths.push_back( large2 );
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

TEST_F(accuracy_test_precallback_single, pow2_large_3D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_large_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_precallback_double, pow2_large_3D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_large_3D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void lds_1D_forward_64_in_place_complex_interleaved_to_complex_interleaved()
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

TEST_F(accuracy_test_precallback_single, lds_1D_forward_64_in_place_complex_interleaved_to_complex_interleaved)
{
	try { lds_1D_forward_64_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

}