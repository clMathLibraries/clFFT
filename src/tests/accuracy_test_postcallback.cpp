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
#include "typedefs.h"
#include "accuracy_test_common.h"
#include <stdexcept>
#include <vector>

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_postcallback_single : public ::testing::Test {
protected:
	accuracy_test_postcallback_single(){}
	virtual ~accuracy_test_postcallback_single(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_postcallback_double : public ::testing::Test {
protected:
	accuracy_test_postcallback_double(){}
	virtual ~accuracy_test_postcallback_double(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

namespace postcallback
{

#pragma region Complex_To_Complex

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

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_forward_out_of_place_complex_planar_to_complex_planar()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { pow2_normal_1D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { pow2_normal_1D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
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

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow2_normal_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow2_normal_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_verysmall_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( 2 );
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
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_verysmall_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_1D_verysmall_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_verysmall_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_1D_verysmall_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_non_unit_stride_and_distance_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 2;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	input_strides.push_back( 42 );
	output_strides.push_back( 42 );
	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 14;
	size_t output_distance = lengths[lengths.size()-1] * output_strides[output_strides.size()-1] + 14;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_non_unit_stride_and_distance_complex_to_complex)
{
	try { pow2_normal_1D_non_unit_stride_and_distance_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_non_unit_stride_and_distance_complex_to_complex)
{
	try { pow2_normal_1D_non_unit_stride_and_distance_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void mixed_normal_1D_val_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( 100 );
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
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, mixed_normal_1D_val_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { mixed_normal_1D_val_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, mixed_normal_1D_val_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { mixed_normal_1D_val_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void pow3_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal3 );
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
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow3_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow3_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_small_1D_forward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_postcallback_single, pow3_small_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow3_small_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_small_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow3_small_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void pow5_normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal5 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { pow5_normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { pow5_normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
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

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_large_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_large_1D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_withEndTranpose_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( 131072 );
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
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_withEndTranpose_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_large_1D_withEndTranpose_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_withEndTranpose_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_large_1D_withEndTranpose_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_forward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
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

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow2_large_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow2_large_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_withEndTranpose_forward_out_of_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( 131072 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_withEndTranpose_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { pow2_large_1D_withEndTranpose_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_withEndTranpose_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { pow2_large_1D_withEndTranpose_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_16M_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( 16777216 );
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
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_16M_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_large_1D_16M_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_16M_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_large_1D_16M_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_1M_forward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( 1048576 );
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

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_1M_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow2_large_1D_1M_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_1M_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow2_large_1D_1M_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large3 );
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
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow3_normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow3_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void pow5_small_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small5 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_small_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow5_small_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_small_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow5_small_1D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large5 );
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
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow5_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow5_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_large_1D_forward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large5 );
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

TEST_F(accuracy_test_postcallback_single, pow5_large_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow5_large_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_large_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow5_large_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow7_normal_1D_backward_out_of_place_complex_planar_to_complex_planar()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t in_layout = layout::complex_planar;
    layout::buffer_layout_t out_layout = layout::complex_planar;
    placeness::placeness_t placeness = placeness::out_of_place;
    direction::direction_t direction = direction::backward;

    data_pattern pattern = sawtooth;
    postcallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_postcallback_single, pow7_normal_1D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { pow7_normal_1D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow7_normal_1D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { pow7_normal_1D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow7_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back(large7);
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
    postcallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_postcallback_single, pow7_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow7_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow7_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow7_large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_2D_array_complex_to_complex_with_odd_batch_size()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( small2 );
	size_t batch = 5;
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

TEST_F(accuracy_test_postcallback_single, pow2_normal_2D_array_complex_to_complex_with_odd_batch_size)
{
	try { pow2_normal_2D_array_complex_to_complex_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_2D_array_complex_to_complex_with_odd_batch_size)
{
	try { pow2_normal_2D_array_complex_to_complex_with_odd_batch_size< double, cl_double, fftw_complex >(); }
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
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_small_2D_forward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( 8 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_small_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { pow2_small_2D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_small_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { pow2_small_2D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_2D_backward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(2) );
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_2D_backward_in_place_complex_planar_to_complex_planar)
{
	try { pow2_large_2D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_2D_backward_in_place_complex_planar_to_complex_planar)
{
	try { pow2_large_2D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow3_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow3_normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_large_2D_backward_out_of_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(5) );
	lengths.push_back( normal5 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_large_2D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { pow5_large_2D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_large_2D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { pow5_large_2D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow7_small_2D_backward_out_of_place_complex_interleaved_to_complex_planar()
{
    std::vector<size_t> lengths;
    lengths.push_back(small7);
    lengths.push_back(small7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t in_layout = layout::complex_interleaved;
    layout::buffer_layout_t out_layout = layout::complex_planar;
    placeness::placeness_t placeness = placeness::out_of_place;
    direction::direction_t direction = direction::backward;

    data_pattern pattern = sawtooth;
    postcallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_postcallback_single, pow7_small_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { pow7_small_2D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow7_small_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { pow7_small_2D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved()
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
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { pow2_normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_normal_3D_forward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_postcallback_single, pow3_normal_3D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow3_normal_3D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_normal_3D_forward_in_place_complex_planar_to_complex_planar)
{
	try { pow3_normal_3D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pre_and_post_callback_normal_complex_to_complex()
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

	data_pattern pattern = sawtooth;
	pre_and_post_callback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pre_and_post_callback_normal_complex_to_complex)
{
	try { pre_and_post_callback_normal_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pre_and_post_callback_normal_complex_to_complex)
{
	try { pre_and_post_callback_normal_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pre_and_post_callback_singlepass_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 2 );
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
	pre_and_post_callback_complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pre_and_post_callback_singlepass_complex_to_complex)
{
	try { pre_and_post_callback_singlepass_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pre_and_post_callback_singlepass_complex_to_complex)
{
	try { pre_and_post_callback_singlepass_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

#pragma endregion

#pragma region Real_To_Complex

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
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

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
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
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow2_normal_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow2_normal_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_small_1D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
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

TEST_F(accuracy_test_postcallback_single, pow2_small_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_small_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_small_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_small_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
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
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_array_real_to_hermitian)
{
	try { pow2_normal_1D_array_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_array_real_to_hermitian)
{
	try { pow2_normal_1D_array_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4 );
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
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { pow2_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { pow2_very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
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
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_normal_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow3_normal_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_normal_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow3_normal_1D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_normal_1D_array_real_to_hermitian_with_odd_batch_size()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal5 );
	size_t batch = 5;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_normal_1D_array_real_to_hermitian_with_odd_batch_size)
{
	try { pow5_normal_1D_array_real_to_hermitian_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_normal_1D_array_real_to_hermitian_with_odd_batch_size)
{
	try { pow5_normal_1D_array_real_to_hermitian_with_odd_batch_size< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_normal_1D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal5 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_normal_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow5_normal_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_normal_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow5_normal_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow7_small_1D_in_place_real_to_hermitian_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back(small7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::in_place;

    data_pattern pattern = sawtooth;
    postcallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_postcallback_single, pow7_small_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow7_small_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow7_small_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow7_small_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_in_place_real_to_hermitian_interleaved()
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
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_large_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_large_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
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
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_4M_in_place_real_to_hermitian_interleaved)
{
	try { pow2_large_1D_4M_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_4M_in_place_real_to_hermitian_interleaved)
{
	try { pow2_large_1D_4M_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_4M_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4194304 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_4M_out_of_place_real_to_hermitian_planar)
{
	try { pow2_large_1D_4M_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_4M_out_of_place_real_to_hermitian_planar)
{
	try { pow2_large_1D_4M_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_large_1D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large3 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_large_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow3_large_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_large_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow3_large_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
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
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_large_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow3_large_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_large_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow3_large_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_large_1D_out_of_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large3 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_large_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow3_large_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_large_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow3_large_1D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
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
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_large_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow5_large_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_large_1D_out_of_place_real_to_hermitian_planar)
{
	try { pow5_large_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow7_large_1D_in_place_real_to_hermitian_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back(large7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::in_place;

    data_pattern pattern = sawtooth;
    postcallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_postcallback_single, pow7_large_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow7_large_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow7_large_1D_in_place_real_to_hermitian_interleaved)
{
	try { pow7_large_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
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
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_2D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_2D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_2D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_2D_out_of_place_real_to_hermitian_planar)
{
	try { pow2_normal_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_2D_out_of_place_real_to_hermitian_planar)
{
	try { pow2_normal_2D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_small_2D_array_real_to_hermitian()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 8;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_small_2D_array_real_to_hermitian)
{
	try { pow2_small_2D_array_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_small_2D_array_real_to_hermitian)
{
	try { pow2_small_2D_array_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_2D_out_of_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(2) );
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow2_large_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow2_large_2D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
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
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_small_2D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { pow2_small_2D_non_unit_stride_and_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_small_2D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { pow2_small_2D_non_unit_stride_and_distance_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_normal_2D_in_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_postcallback_single, pow3_normal_2D_in_place_real_to_hermitian_interleaved)
{
	try { pow3_normal_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_normal_2D_in_place_real_to_hermitian_interleaved)
{
	try { pow3_normal_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_large_2D_out_of_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(5) );
	lengths.push_back( normal5 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_large_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow5_large_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_large_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { pow5_large_2D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow7_small_2D_non_unit_stride_and_distance_real_to_hermitian()
{
    std::vector<size_t> lengths;
    lengths.push_back(small7);
    lengths.push_back(small7);
    size_t batch = 2;

    std::vector<size_t> input_strides;
    input_strides.push_back(5);
    input_strides.push_back(lengths[0] * input_strides[0] + 1);

    std::vector<size_t> output_strides;
    output_strides.push_back(2);
    output_strides.push_back(lengths[0] * output_strides[0] + 2);

    size_t input_distance = lengths[lengths.size() - 1] * input_strides[input_strides.size() - 1] + 30;
    size_t output_distance = lengths[lengths.size() - 1] * output_strides[output_strides.size() - 1] + 42;

    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    postcallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_postcallback_single, pow7_small_2D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { pow7_small_2D_non_unit_stride_and_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow7_small_2D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { pow7_small_2D_non_unit_stride_and_distance_real_to_hermitian< double, cl_double, fftw_complex >(); }
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
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_3D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_3D_in_place_real_to_hermitian_interleaved)
{
	try { pow2_normal_3D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_small_3D_non_unit_stride_and_distance_real_to_hermitian()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4 );
	lengths.push_back( 4 );
	lengths.push_back( 4 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	input_strides.push_back( 5 );
	input_strides.push_back( lengths[0] * input_strides[0] + 1 );
	input_strides.push_back( lengths[1] * input_strides[1] + 1 );

	std::vector<size_t> output_strides;
	output_strides.push_back( 2 );
	output_strides.push_back( lengths[0] * output_strides[0] + 2 );
	output_strides.push_back( lengths[1] * output_strides[1] + 2 );

	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 30;
	size_t output_distance = lengths[lengths.size()-1] * output_strides[output_strides.size()-1] + 42;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_small_3D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { pow2_small_3D_non_unit_stride_and_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_small_3D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { pow2_small_3D_non_unit_stride_and_distance_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_large_3D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_large_3D_out_of_place_real_to_hermitian_planar)
{
	try { pow3_large_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_large_3D_out_of_place_real_to_hermitian_planar)
{
	try { pow3_large_3D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

#pragma endregion

#pragma region Complex_To_Real

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_in_place_hermitian_interleaved_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
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

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_in_place_hermitian_interleaved_to_real)
{
	try { pow2_normal_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_in_place_hermitian_interleaved_to_real)
{
	try { pow2_normal_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_out_of_place_hermitian_interleaved_to_real()
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
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow2_normal_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow2_normal_1D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_out_of_place_hermitian_planar_to_real()
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
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_out_of_place_hermitian_planar_to_real)
{
	try { pow2_normal_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_out_of_place_hermitian_planar_to_real)
{
	try { pow2_normal_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_1D_user_defined_scale_hermitian_to_real()
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
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness, 42.0f );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_1D_user_defined_scale_hermitian_to_real)
{
	try { pow2_normal_1D_user_defined_scale_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_1D_user_defined_scale_hermitian_to_real)
{
	try { pow2_normal_1D_user_defined_scale_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_small_1D_non_unit_stride_hermitian_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	input_strides.push_back( 3 );
	output_strides.push_back( 3 );
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_small_1D_non_unit_stride_hermitian_to_real)
{
	try { pow2_small_1D_non_unit_stride_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_small_1D_non_unit_stride_hermitian_to_real)
{
	try { pow2_small_1D_non_unit_stride_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_normal_1D_out_of_place_hermitian_interleaved_to_real()
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
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_normal_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow3_normal_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_normal_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow3_normal_1D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_small_1D_out_of_place_hermitian_planar_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( small5 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_small_1D_out_of_place_hermitian_planar_to_real)
{
	try { pow5_small_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_small_1D_out_of_place_hermitian_planar_to_real)
{
	try { pow5_small_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow7_normal_1D_in_place_hermitian_interleaved_to_real()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::in_place;

    data_pattern pattern = sawtooth;
    postcallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_postcallback_single, pow7_normal_1D_in_place_hermitian_interleaved_to_real)
{
	try { pow7_normal_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow7_normal_1D_in_place_hermitian_interleaved_to_real)
{
	try { pow7_normal_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
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
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_in_place_hermitian_interleaved_to_real)
{
	try { pow2_large_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_in_place_hermitian_interleaved_to_real)
{
	try { pow2_large_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_large_1D_out_of_place_hermitian_interleaved_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow2_large_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow2_large_1D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
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
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_large_1D_out_of_place_hermitian_planar_to_real)
{
	try { pow2_large_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_large_1D_out_of_place_hermitian_planar_to_real)
{
	try { pow2_large_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_large_1D_in_place_hermitian_interleaved_to_real()
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
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_large_1D_in_place_hermitian_interleaved_to_real)
{
	try { pow3_large_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_large_1D_in_place_hermitian_interleaved_to_real)
{
	try { pow3_large_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_large_1D_out_of_place_hermitian_planar_to_real()
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
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_large_1D_out_of_place_hermitian_planar_to_real)
{
	try { pow5_large_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_large_1D_out_of_place_hermitian_planar_to_real)
{
	try { pow5_large_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow7_large_1D_out_of_place_hermitian_interleaved_to_real()
{
    std::vector<size_t> lengths;
    lengths.push_back(large7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    postcallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_postcallback_single, pow7_large_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow7_large_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow7_large_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow7_large_1D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_2D_in_place_hermitian_interleaved_to_real()
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
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_2D_in_place_hermitian_interleaved_to_real)
{
	try { pow2_normal_2D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_2D_in_place_hermitian_interleaved_to_real)
{
	try { pow2_normal_2D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_small_2D_non_unit_stride_and_distance_hermitian_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4 );
	lengths.push_back( 4 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	input_strides.push_back( 12 );
	input_strides.push_back( lengths[0] * input_strides[0] + 9 );

	std::vector<size_t> output_strides;
	output_strides.push_back( 7 );
	output_strides.push_back( lengths[0] * output_strides[0] + 32 );

	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 50;
	size_t output_distance = lengths[lengths.size()-1] * output_strides[output_strides.size()-1] + 60;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_small_2D_non_unit_stride_and_distance_hermitian_to_real)
{
	try { pow2_small_2D_non_unit_stride_and_distance_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_small_2D_non_unit_stride_and_distance_hermitian_to_real)
{
	try { pow2_small_2D_non_unit_stride_and_distance_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_normal_2D_out_of_place_hermitian_planar_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_normal_2D_out_of_place_hermitian_planar_to_real)
{
	try { pow3_normal_2D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_normal_2D_out_of_place_hermitian_planar_to_real)
{
	try { pow3_normal_2D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_large_2D_out_of_place_hermitian_interleaved_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(5) );
	lengths.push_back( normal5 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_large_2D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow5_large_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_large_2D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow5_large_2D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow7_small_2D_in_place_hermitian_interleaved_to_real()
{
    std::vector<size_t> lengths;
    lengths.push_back(small7);
    lengths.push_back(small7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::in_place;

    data_pattern pattern = sawtooth;
    postcallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_postcallback_single, pow7_small_2D_in_place_hermitian_interleaved_to_real)
{
	try { pow7_small_2D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow7_small_2D_in_place_hermitian_interleaved_to_real)
{
	try { pow7_small_2D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_normal_3D_in_place_hermitian_interleaved_to_real()
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
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_normal_3D_in_place_hermitian_interleaved_to_real)
{
	try { pow2_normal_3D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_normal_3D_in_place_hermitian_interleaved_to_real)
{
	try { pow2_normal_3D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow2_small_3D_non_unit_stride_and_distance_hermitian_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4 );
	lengths.push_back( 4 );
	lengths.push_back( 4 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	input_strides.push_back( 5 );
	input_strides.push_back( lengths[0] * input_strides[0] + 1 );
	input_strides.push_back( lengths[1] * input_strides[1] + 1 );

	std::vector<size_t> output_strides;
	output_strides.push_back( 2 );
	output_strides.push_back( lengths[0] * output_strides[0] + 2 );
	output_strides.push_back( lengths[1] * output_strides[1] + 2 );

	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 30;
	size_t output_distance = lengths[lengths.size()-1] * output_strides[output_strides.size()-1] + 42;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow2_small_3D_non_unit_stride_and_distance_hermitian_to_real)
{
	try { pow2_small_3D_non_unit_stride_and_distance_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow2_small_3D_non_unit_stride_and_distance_hermitian_to_real)
{
	try { pow2_small_3D_non_unit_stride_and_distance_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow3_normal_3D_out_of_place_hermitian_planar_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( small3 );
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow3_normal_3D_out_of_place_hermitian_planar_to_real)
{
	try { pow3_normal_3D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow3_normal_3D_out_of_place_hermitian_planar_to_real)
{
	try { pow3_normal_3D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

template< class T, class cl_T, class fftw_T >
void pow5_large_3D_out_of_place_hermitian_interleaved_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( large5 );
	lengths.push_back( 5 );
	lengths.push_back( 5 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_postcallback_single, pow5_large_3D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow5_large_3D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_postcallback_double, pow5_large_3D_out_of_place_hermitian_interleaved_to_real)
{
	try { pow5_large_3D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ huge 1D ^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************

template< class T, class cl_T, class fftw_T>
void huge_1D_forward_in_place_complex_to_complex(size_t lenSize, size_t batchSize, layout::buffer_layout_t layoutType, direction::direction_t direction_type)
{
	std::vector<size_t> lengths;
	lengths.push_back(lenSize);
	size_t batch = batchSize;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layoutType;
	layout::buffer_layout_t out_layout = layoutType;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction_type;

	data_pattern pattern = sawtooth;
	postcallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}
//TESTS disabled by default since they take a long time to execute
//TO enable this tests
//1. make sure ENV CLFFT_REQUEST_LIB_NOMEMALLOC=1
//2. pass --gtest_also_run_disabled_tests to TEST.exe

//177147 = 243 * 243 * 3, forward (no need for backward), planar and interleaved, single and double, batch size 1 and 3
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_177147_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_177147_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_177147_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_177147_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
//interleaved
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_177147_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_177147_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_177147_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_177147_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}

//1594323 = 729 * 729 * 3 foward (no need for backward), planar and interleaved, single and double, batch size 1 and 3
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1594323_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1594323_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1594323_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1594323_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}

//interleaved
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1594323_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1594323_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1594323_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1594323_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}

//14348907 = 2187 * 2187 * 3 backward, planar and interleaved, single and double, batch size 1 and 3
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_14348907_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_14348907_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}

//interleaved
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_14348907_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_14348907_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}

//78125 = 125 * 125 * 5, forward planar and interleaved, single and double, batch size 1 and 3
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_78125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(78125, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_78125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(78125, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_78125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(78125, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_78125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(78125, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}

//interleaved
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_78125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(78125, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_78125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(78125, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_78125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(78125, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_78125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(78125, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}

//1953125 = 625 * 625 * 5 forward, planar and interleaved, single and double, batch size 1 and 3
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1953125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1953125, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1953125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1953125, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1953125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1953125, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1953125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1953125, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}

//interleaved
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1953125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1953125, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1953125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1953125, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1953125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1953125, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1953125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1953125, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}

//48828125 = 3125 * 3125 * 5 forward, planar and interleaved, single and double, batch size 1 and 3
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_48828125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(48828125, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_48828125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(48828125, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_48828125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(48828125, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_48828125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(48828125, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}

//interleaved
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_48828125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(48828125, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_48828125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(48828125, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_48828125_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(48828125, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_postcallback_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_48828125_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(48828125, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}

#pragma endregion
} //namespace
