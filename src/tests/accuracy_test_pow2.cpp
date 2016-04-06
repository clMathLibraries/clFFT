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
class accuracy_test_pow2_single : public ::testing::Test {
protected:
	accuracy_test_pow2_single(){}
	virtual ~accuracy_test_pow2_single(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_pow2_double : public ::testing::Test {
protected:
	accuracy_test_pow2_double(){}
	virtual ~accuracy_test_pow2_double(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

namespace power2
{
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ normal 1D ^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_forward_in_place_complex_planar_to_complex_planar()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { normal_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { normal_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_backward_in_place_complex_planar_to_complex_planar()
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_backward_in_place_complex_planar_to_complex_planar)
{
	try { normal_1D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_backward_in_place_complex_planar_to_complex_planar)
{
	try { normal_1D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void len65536_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( 65536 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, len65536_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, len65536_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}
// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_forward_out_of_place_complex_planar_to_complex_planar()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_1D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_1D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_backward_out_of_place_complex_planar_to_complex_planar()
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_1D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_1D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_in_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_in_place_real_to_hermitian_interleaved)
{
	try { normal_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_in_place_real_to_hermitian_interleaved)
{
	try { normal_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_in_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_in_place_hermitian_interleaved_to_real)
{
	try { normal_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_in_place_hermitian_interleaved_to_real)
{
	try { normal_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_out_of_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { normal_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { normal_1D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_out_of_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { normal_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { normal_1D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_out_of_place_real_to_hermitian_planar()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_out_of_place_real_to_hermitian_planar)
{
	try { normal_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_out_of_place_real_to_hermitian_planar)
{
	try { normal_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_out_of_place_hermitian_planar_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_out_of_place_hermitian_planar_to_real)
{
	try { normal_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_out_of_place_hermitian_planar_to_real)
{
	try { normal_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ small 1D ^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_forward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { small_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { small_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_backward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_backward_in_place_complex_planar_to_complex_planar)
{
	try { small_1D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_backward_in_place_complex_planar_to_complex_planar)
{
	try { small_1D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_forward_out_of_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_1D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_1D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_backward_out_of_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_1D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_1D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_forward_out_of_place_complex_planar_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_1D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_1D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_backward_out_of_place_complex_planar_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_1D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_1D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_forward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_1D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_1D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_backward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_1D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_1D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_in_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_in_place_real_to_hermitian_interleaved)
{
	try { small_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_in_place_real_to_hermitian_interleaved)
{
	try { small_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_in_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_in_place_hermitian_interleaved_to_real)
{
	try { small_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_in_place_hermitian_interleaved_to_real)
{
	try { small_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_out_of_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { small_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { small_1D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_out_of_place_hermitian_interleaved_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { small_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { small_1D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_out_of_place_real_to_hermitian_planar)
{
	try { small_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_out_of_place_real_to_hermitian_planar)
{
	try { small_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_out_of_place_hermitian_planar_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_out_of_place_hermitian_planar_to_real)
{
	try { small_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_out_of_place_hermitian_planar_to_real)
{
	try { small_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ large 1D ^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_forward_in_place_complex_planar_to_complex_planar()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { large_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { large_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ huge 1D ^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
//TESTS disabled by default since they take a long time to execute
//TO enable this tests
//1. make sure ENV CLFFT_REQUEST_LIB_NOMEMALLOC=1
//2. pass --gtest_also_run_disabled_tests to TEST.exe

#define CLFFT_TEST_HUGE
#ifdef CLFFT_TEST_HUGE

#define HUGE_TEST_MAKE(test_name, len, bat) \
template< class T, class cl_T, class fftw_T > \
void test_name() \
{ \
	std::vector<size_t> lengths; \
	lengths.push_back( len ); \
	size_t batch = bat; \
\
	std::vector<size_t> input_strides; \
	std::vector<size_t> output_strides; \
	size_t input_distance = 0; \
	size_t output_distance = 0; \
	layout::buffer_layout_t in_layout = layout::complex_planar; \
	layout::buffer_layout_t out_layout = layout::complex_planar; \
	placeness::placeness_t placeness = placeness::in_place; \
	direction::direction_t direction = direction::forward; \
\
	data_pattern pattern = sawtooth; \
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness ); \
}

#define SP_HUGE_TEST(test_name, len, bat) \
\
	HUGE_TEST_MAKE(test_name, len, bat) \
\
	TEST_F(accuracy_test_pow2_single, test_name) \
	{ \
		try { test_name< float, cl_float, fftwf_complex >(); } \
		catch( const std::exception& err ) { handle_exception(err);	} \
	}

#define DP_HUGE_TEST(test_name, len, bat) \
\
	HUGE_TEST_MAKE(test_name, len, bat) \
\
	TEST_F(accuracy_test_pow2_double, test_name) \
	{ \
		try { test_name< double, cl_double, fftw_complex >(); } \
		catch( const std::exception& err ) { handle_exception(err);	} \
	}

SP_HUGE_TEST( DISABLED_huge_sp_test_1, 1048576,    11 )
SP_HUGE_TEST( DISABLED_huge_sp_test_2, 1048576*2,  7  )
SP_HUGE_TEST( DISABLED_huge_sp_test_3, 1048576*4,  3  )
SP_HUGE_TEST( DISABLED_huge_sp_test_4, 1048576*8,  5  )
SP_HUGE_TEST( DISABLED_huge_sp_test_5, 1048576*16, 3  )
SP_HUGE_TEST( DISABLED_huge_sp_test_6, 1048576*32, 2  )
SP_HUGE_TEST( DISABLED_huge_sp_test_7, 1048576*64, 1  )

DP_HUGE_TEST( DISABLED_huge_dp_test_1, 524288,    11 )
DP_HUGE_TEST( DISABLED_huge_dp_test_2, 524288*2,  7  )
DP_HUGE_TEST( DISABLED_huge_dp_test_3, 524288*4,  3  )
DP_HUGE_TEST( DISABLED_huge_dp_test_4, 524288*8,  5  )
DP_HUGE_TEST( DISABLED_huge_dp_test_5, 524288*16, 3  )
DP_HUGE_TEST( DISABLED_huge_dp_test_6, 524288*32, 2  )
DP_HUGE_TEST( DISABLED_huge_dp_test_7, 524288*64, 1  )

SP_HUGE_TEST( DISABLED_large_sp_test_1, 8192,    11 )
SP_HUGE_TEST( DISABLED_large_sp_test_2, 8192*2,  7  )
SP_HUGE_TEST( DISABLED_large_sp_test_3, 8192*4,  3  )
SP_HUGE_TEST( DISABLED_large_sp_test_4, 8192*8,  5  )
SP_HUGE_TEST( DISABLED_large_sp_test_5, 8192*16, 3  )
SP_HUGE_TEST( DISABLED_large_sp_test_6, 8192*32, 21  )
SP_HUGE_TEST( DISABLED_large_sp_test_7, 8192*64, 17  )

DP_HUGE_TEST( DISABLED_large_dp_test_1, 4096,    11 )
DP_HUGE_TEST( DISABLED_large_dp_test_2, 4096*2,  7  )
DP_HUGE_TEST( DISABLED_large_dp_test_3, 4096*4,  3  )
DP_HUGE_TEST( DISABLED_large_dp_test_4, 4096*8,  5  )
DP_HUGE_TEST( DISABLED_large_dp_test_5, 4096*16, 3  )
DP_HUGE_TEST( DISABLED_large_dp_test_6, 4096*32, 21  )
DP_HUGE_TEST( DISABLED_large_dp_test_7, 4096*64, 17  )

#endif

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_backward_in_place_complex_planar_to_complex_planar()
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_backward_in_place_complex_planar_to_complex_planar)
{
	try { large_1D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_backward_in_place_complex_planar_to_complex_planar)
{
	try { large_1D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_forward_out_of_place_complex_planar_to_complex_planar()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_1D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_1D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_backward_out_of_place_complex_planar_to_complex_planar()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_1D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_1D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_forward_out_of_place_complex_planar_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_1D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_1D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_backward_out_of_place_complex_planar_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_1D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_1D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_forward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_1D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_1D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_backward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_1D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_1D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_in_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_in_place_real_to_hermitian_interleaved)
{
	try { large_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_in_place_real_to_hermitian_interleaved)
{
	try { large_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_4M_in_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_4M_in_place_real_to_hermitian_interleaved)
{
	try { large_1D_4M_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_4M_in_place_real_to_hermitian_interleaved)
{
	try { large_1D_4M_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}
// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_in_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_in_place_hermitian_interleaved_to_real)
{
	try { large_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_in_place_hermitian_interleaved_to_real)
{
	try { large_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_out_of_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { large_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { large_1D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_out_of_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { large_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { large_1D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_out_of_place_real_to_hermitian_planar()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_out_of_place_real_to_hermitian_planar)
{
	try { large_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_out_of_place_real_to_hermitian_planar)
{
	try { large_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_out_of_place_hermitian_planar_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_out_of_place_hermitian_planar_to_real)
{
	try { large_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_out_of_place_hermitian_planar_to_real)
{
	try { large_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ normal 2D ^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_forward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_forward_in_place_complex_planar_to_complex_planar)
{
	try { normal_2D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_forward_in_place_complex_planar_to_complex_planar)
{
	try { normal_2D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_backward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_backward_in_place_complex_planar_to_complex_planar)
{
	try { normal_2D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_backward_in_place_complex_planar_to_complex_planar)
{
	try { normal_2D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_forward_out_of_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_2D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_2D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_backward_out_of_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( normal2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_2D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_2D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar()
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
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar()
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
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_in_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_in_place_real_to_hermitian_interleaved)
{
	try { normal_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_in_place_real_to_hermitian_interleaved)
{
	try { normal_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_in_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_in_place_hermitian_interleaved_to_real)
{
	try { normal_2D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_in_place_hermitian_interleaved_to_real)
{
	try { normal_2D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_out_of_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { normal_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { normal_2D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_out_of_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_out_of_place_hermitian_interleaved_to_real)
{
	try { normal_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_out_of_place_hermitian_interleaved_to_real)
{
	try { normal_2D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_out_of_place_real_to_hermitian_planar()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_out_of_place_real_to_hermitian_planar)
{
	try { normal_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_out_of_place_real_to_hermitian_planar)
{
	try { normal_2D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_out_of_place_hermitian_planar_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_out_of_place_hermitian_planar_to_real)
{
	try { normal_2D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_out_of_place_hermitian_planar_to_real)
{
	try { normal_2D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ small 2D ^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_forward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_forward_in_place_complex_planar_to_complex_planar)
{
	try { small_2D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_forward_in_place_complex_planar_to_complex_planar)
{
	try { small_2D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_backward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_backward_in_place_complex_planar_to_complex_planar)
{
	try { small_2D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_backward_in_place_complex_planar_to_complex_planar)
{
	try { small_2D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_backward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_forward_out_of_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_2D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_2D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_backward_out_of_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_2D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_2D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_forward_out_of_place_complex_planar_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_2D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_2D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_backward_out_of_place_complex_planar_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_2D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_2D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_forward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_2D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_2D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_backward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_2D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_2D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_in_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_in_place_real_to_hermitian_interleaved)
{
	try { small_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_in_place_real_to_hermitian_interleaved)
{
	try { small_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_in_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_in_place_hermitian_interleaved_to_real)
{
	try { small_2D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_in_place_hermitian_interleaved_to_real)
{
	try { small_2D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_out_of_place_real_to_hermitian_interleaved()
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
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { small_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { small_2D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_out_of_place_hermitian_interleaved_to_real()
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
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_out_of_place_hermitian_interleaved_to_real)
{
	try { small_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_out_of_place_hermitian_interleaved_to_real)
{
	try { small_2D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_out_of_place_real_to_hermitian_planar)
{
	try { small_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_out_of_place_real_to_hermitian_planar)
{
	try { small_2D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_out_of_place_hermitian_planar_to_real()
{
	std::vector<size_t> lengths;
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_out_of_place_hermitian_planar_to_real)
{
	try { small_2D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_out_of_place_hermitian_planar_to_real)
{
	try { small_2D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ large 2D ^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_forward_in_place_complex_planar_to_complex_planar()
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
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_forward_in_place_complex_planar_to_complex_planar)
{
	try { large_2D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_forward_in_place_complex_planar_to_complex_planar)
{
	try { large_2D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_backward_in_place_complex_planar_to_complex_planar()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_backward_in_place_complex_planar_to_complex_planar)
{
	try { large_2D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_backward_in_place_complex_planar_to_complex_planar)
{
	try { large_2D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(2) );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_backward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(2) );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_forward_out_of_place_complex_planar_to_complex_planar()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_2D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_2D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_backward_out_of_place_complex_planar_to_complex_planar()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_2D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_2D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(2) );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(2) );
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_forward_out_of_place_complex_planar_to_complex_interleaved()
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
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_2D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_2D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_backward_out_of_place_complex_planar_to_complex_interleaved()
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
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_2D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_2D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_forward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(2) );
	lengths.push_back( normal2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_2D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_2D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_backward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( MaxLength2D<T>(2) );
	lengths.push_back( normal2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_2D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_2D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_in_place_real_to_hermitian_interleaved()
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
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_in_place_real_to_hermitian_interleaved)
{
	try { large_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_in_place_real_to_hermitian_interleaved)
{
	try { large_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_in_place_hermitian_interleaved_to_real()
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
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_in_place_hermitian_interleaved_to_real)
{
	try { large_2D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_in_place_hermitian_interleaved_to_real)
{
	try { large_2D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_out_of_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { large_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { large_2D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_out_of_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_out_of_place_hermitian_interleaved_to_real)
{
	try { large_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_out_of_place_hermitian_interleaved_to_real)
{
	try { large_2D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_out_of_place_real_to_hermitian_planar()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_out_of_place_real_to_hermitian_planar)
{
	try { large_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_out_of_place_real_to_hermitian_planar)
{
	try { large_2D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_out_of_place_hermitian_planar_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_out_of_place_hermitian_planar_to_real)
{
	try { large_2D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_out_of_place_hermitian_planar_to_real)
{
	try { large_2D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ normal 3D ^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_forward_in_place_complex_planar_to_complex_planar()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_forward_in_place_complex_planar_to_complex_planar)
{
	try { normal_3D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_forward_in_place_complex_planar_to_complex_planar)
{
	try { normal_3D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_backward_in_place_complex_planar_to_complex_planar()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_backward_in_place_complex_planar_to_complex_planar)
{
	try { normal_3D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_backward_in_place_complex_planar_to_complex_planar)
{
	try { normal_3D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_forward_out_of_place_complex_planar_to_complex_planar()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_3D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_3D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_backward_out_of_place_complex_planar_to_complex_planar()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_3D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_3D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar()
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
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar()
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
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_in_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_in_place_real_to_hermitian_interleaved)
{
	try { normal_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_in_place_real_to_hermitian_interleaved)
{
	try { normal_3D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_in_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_in_place_hermitian_interleaved_to_real)
{
	try { normal_3D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_in_place_hermitian_interleaved_to_real)
{
	try { normal_3D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_out_of_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_out_of_place_real_to_hermitian_interleaved)
{
	try { normal_3D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_out_of_place_real_to_hermitian_interleaved)
{
	try { normal_3D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_out_of_place_hermitian_interleaved_to_real()
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
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_out_of_place_hermitian_interleaved_to_real)
{
	try { normal_3D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_out_of_place_hermitian_interleaved_to_real)
{
	try { normal_3D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_out_of_place_real_to_hermitian_planar()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_out_of_place_real_to_hermitian_planar)
{
	try { normal_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_out_of_place_real_to_hermitian_planar)
{
	try { normal_3D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_3D_out_of_place_hermitian_planar_to_real()
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
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_3D_out_of_place_hermitian_planar_to_real)
{
	try { normal_3D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_3D_out_of_place_hermitian_planar_to_real)
{
	try { normal_3D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ small 3D ^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_forward_in_place_complex_planar_to_complex_planar()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_forward_in_place_complex_planar_to_complex_planar)
{
	try { small_3D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_forward_in_place_complex_planar_to_complex_planar)
{
	try { small_3D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_backward_in_place_complex_planar_to_complex_planar()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_backward_in_place_complex_planar_to_complex_planar)
{
	try { small_3D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_backward_in_place_complex_planar_to_complex_planar)
{
	try { small_3D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_forward_in_place_complex_interleaved_to_complex_interleaved()
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
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_forward_out_of_place_complex_planar_to_complex_planar()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_3D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_3D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_backward_out_of_place_complex_planar_to_complex_planar()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_3D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_3D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_forward_out_of_place_complex_planar_to_complex_interleaved()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_3D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_3D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_backward_out_of_place_complex_planar_to_complex_interleaved()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_3D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_3D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_forward_out_of_place_complex_interleaved_to_complex_planar()
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
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_3D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_3D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_backward_out_of_place_complex_interleaved_to_complex_planar()
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
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_3D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_3D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_in_place_real_to_hermitian_interleaved()
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
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, _small_3D_in_place_real_to_hermitian_interleaved)
{
	try { small_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, _small_3D_in_place_real_to_hermitian_interleaved)
{
	try { small_3D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_in_place_hermitian_interleaved_to_real()
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
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, _small_3D_in_place_hermitian_interleaved_to_real)
{
	try { small_3D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, _small_3D_in_place_hermitian_interleaved_to_real)
{
	try { small_3D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_out_of_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, _small_3D_out_of_place_real_to_hermitian_interleaved)
{
	try { small_3D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, _small_3D_out_of_place_real_to_hermitian_interleaved)
{
	try { small_3D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_out_of_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, _small_3D_out_of_place_hermitian_interleaved_to_real)
{
	try { small_3D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, _small_3D_out_of_place_hermitian_interleaved_to_real)
{
	try { small_3D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_out_of_place_real_to_hermitian_planar()
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
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, _small_3D_out_of_place_real_to_hermitian_planar)
{
	try { small_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, _small_3D_out_of_place_real_to_hermitian_planar)
{
	try { small_3D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_out_of_place_hermitian_planar_to_real()
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
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, _small_3D_out_of_place_hermitian_planar_to_real)
{
	try { small_3D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, _small_3D_out_of_place_hermitian_planar_to_real)
{
	try { small_3D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ large 3D ^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_forward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_forward_in_place_complex_planar_to_complex_planar)
{
	try { large_3D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_forward_in_place_complex_planar_to_complex_planar)
{
	try { large_3D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_backward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_backward_in_place_complex_planar_to_complex_planar)
{
	try { large_3D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_backward_in_place_complex_planar_to_complex_planar)
{
	try { large_3D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_backward_in_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_forward_out_of_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_3D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_3D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_backward_out_of_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_3D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_3D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_forward_out_of_place_complex_planar_to_complex_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_3D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_3D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_backward_out_of_place_complex_planar_to_complex_interleaved()
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
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_3D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_3D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_forward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( 2 );
	lengths.push_back( large2 );
	lengths.push_back( 2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_3D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_3D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_backward_out_of_place_complex_interleaved_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_3D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_3D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_in_place_real_to_hermitian_interleaved()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_in_place_real_to_hermitian_interleaved)
{
	try { large_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_in_place_real_to_hermitian_interleaved)
{
	try { large_3D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_in_place_hermitian_interleaved_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
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

TEST_F(accuracy_test_pow2_single, large_3D_in_place_hermitian_interleaved_to_real)
{
	try { large_3D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_in_place_hermitian_interleaved_to_real)
{
	try { large_3D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_out_of_place_real_to_hermitian_interleaved()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_out_of_place_real_to_hermitian_interleaved)
{
	try { large_3D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_out_of_place_real_to_hermitian_interleaved)
{
	try { large_3D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_out_of_place_hermitian_interleaved_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_out_of_place_hermitian_interleaved_to_real)
{
	try { large_3D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_out_of_place_hermitian_interleaved_to_real)
{
	try { large_3D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_out_of_place_real_to_hermitian_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	lengths.push_back( 2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_out_of_place_real_to_hermitian_planar)
{
	try { large_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_out_of_place_real_to_hermitian_planar)
{
	try { large_3D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_3D_out_of_place_hermitian_planar_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( 2 );
	lengths.push_back( large2 );
	lengths.push_back( 2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_3D_out_of_place_hermitian_planar_to_real)
{
	try { large_3D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_3D_out_of_place_hermitian_planar_to_real)
{
	try { large_3D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ special ^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_array_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 8;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_array_complex_to_complex)
{
	try { normal_1D_array_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_array_complex_to_complex)
{
	try { normal_1D_array_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_array_complex_to_complex_with_odd_batch_size()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 5;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_array_complex_to_complex_with_odd_batch_size)
{
	try { normal_1D_array_complex_to_complex_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_array_complex_to_complex_with_odd_batch_size)
{
	try { normal_1D_array_complex_to_complex_with_odd_batch_size< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_array_real_to_hermitian()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_array_real_to_hermitian)
{
	try { normal_1D_array_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_array_real_to_hermitian)
{
	try { normal_1D_array_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_array_real_to_hermitian_with_odd_batch_size()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_array_real_to_hermitian_with_odd_batch_size)
{
	try { normal_1D_array_real_to_hermitian_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_array_real_to_hermitian_with_odd_batch_size)
{
	try { normal_1D_array_real_to_hermitian_with_odd_batch_size< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_array_hermitian_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_array_hermitian_to_real)
{
	try { normal_1D_array_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_array_hermitian_to_real)
{
	try { normal_1D_array_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_array_hermitian_to_real_with_odd_batch_size()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 5;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_array_hermitian_to_real_with_odd_batch_size)
{
	try { normal_1D_array_hermitian_to_real_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_array_hermitian_to_real_with_odd_batch_size)
{
	try { normal_1D_array_hermitian_to_real_with_odd_batch_size< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_array_real_to_hermitian()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_array_real_to_hermitian)
{
	try { small_2D_array_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_array_real_to_hermitian)
{
	try { small_2D_array_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_array_real_to_hermitian_with_odd_batch_size()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 5;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_array_real_to_hermitian_with_odd_batch_size)
{
	try { small_2D_array_real_to_hermitian_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_array_real_to_hermitian_with_odd_batch_size)
{
	try { small_2D_array_real_to_hermitian_with_odd_batch_size< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_array_hermitian_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_array_hermitian_to_real)
{
	try { small_2D_array_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_array_hermitian_to_real)
{
	try { small_2D_array_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_array_hermitian_to_real_with_odd_batch_size()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 5;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_planar;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_array_hermitian_to_real_with_odd_batch_size)
{
	try { small_2D_array_hermitian_to_real_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_array_hermitian_to_real_with_odd_batch_size)
{
	try { small_2D_array_hermitian_to_real_with_odd_batch_size< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_array_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	size_t batch = 2;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_array_complex_to_complex)
{
	try { large_1D_array_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_array_complex_to_complex)
{
	try { large_1D_array_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void astoundingly_large_1D_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 1024 );
	size_t batch = 65536;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, DISABLED_astoundingly_large_1D_complex_to_complex)
{
	try { astoundingly_large_1D_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, DISABLED_astoundingly_large_1D_complex_to_complex)
{
	try { astoundingly_large_1D_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void very_small_1D_non_unit_stride_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	input_strides.push_back( 3 );
	output_strides.push_back( 3 );
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, very_small_1D_non_unit_stride_complex_to_complex)
{
	try { very_small_1D_non_unit_stride_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, very_small_1D_non_unit_stride_complex_to_complex)
{
	try { very_small_1D_non_unit_stride_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_non_unit_stride_real_to_hermitian()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_non_unit_stride_real_to_hermitian)
{
	try {small_1D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_non_unit_stride_real_to_hermitian)
{
	try { small_1D_non_unit_stride_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_1D_non_unit_stride_hermitian_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_non_unit_stride_hermitian_to_real)
{
	try { small_1D_non_unit_stride_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_1D_non_unit_stride_hermitian_to_real)
{
	try { small_1D_non_unit_stride_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void very_small_1D_non_unit_stride_real_to_hermitian()
{
	std::vector<size_t> lengths;
	lengths.push_back( 8 );
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, very_small_1D_non_unit_stride_real_to_hermitian)
{
	try {very_small_1D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, very_small_1D_non_unit_stride_real_to_hermitian)
{
	try { very_small_1D_non_unit_stride_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void very_small_1D_non_unit_stride_hermitian_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( 8 );
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, very_small_1D_non_unit_stride_hermitian_to_real)
{
	try { very_small_1D_non_unit_stride_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, very_small_1D_non_unit_stride_hermitian_to_real)
{
	try { very_small_1D_non_unit_stride_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void very_very_small_1D_non_unit_stride_real_to_hermitian()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	input_strides.push_back( 3 );
	output_strides.push_back( 3 );
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = impulse;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, very_very_small_1D_non_unit_stride_real_to_hermitian)
{
	try {very_very_small_1D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, very_very_small_1D_non_unit_stride_real_to_hermitian)
{
	try { very_very_small_1D_non_unit_stride_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void very_very_small_1D_non_unit_stride_hermitian_to_real()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	input_strides.push_back( 3 );
	output_strides.push_back( 3 );
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = impulse;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, very_very_small_1D_non_unit_stride_hermitian_to_real)
{
	try { very_very_small_1D_non_unit_stride_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, very_very_small_1D_non_unit_stride_hermitian_to_real)
{
	try { very_very_small_1D_non_unit_stride_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_non_unit_stride_and_distance_complex_to_complex()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_non_unit_stride_and_distance_complex_to_complex)
{
	try { normal_1D_non_unit_stride_and_distance_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_non_unit_stride_and_distance_complex_to_complex)
{
	try { normal_1D_non_unit_stride_and_distance_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void very_small_1D_non_unit_stride_and_distance_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	input_strides.push_back( 2 );

	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 2;

	std::vector<size_t> output_strides( input_strides );
	size_t output_distance = input_distance;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = impulse;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, very_small_1D_non_unit_stride_and_distance_real_to_complex)
{
	try { very_small_1D_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, very_small_1D_non_unit_stride_and_distance_real_to_complex)
{
	try { very_small_1D_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex()
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
	placeness::placeness_t placeness = placeness::in_place;

	data_pattern pattern = impulse;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_forward_user_defined_scale_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 42.0f );
}

TEST_F(accuracy_test_pow2_single, normal_1D_forward_user_defined_scale_complex_to_complex)
{
	try { normal_1D_forward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_forward_user_defined_scale_complex_to_complex)
{
	try { normal_1D_forward_user_defined_scale_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_backward_user_defined_scale_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 42.5f );
}

TEST_F(accuracy_test_pow2_single, normal_1D_backward_user_defined_scale_complex_to_complex)
{
	try { normal_1D_backward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_backward_user_defined_scale_complex_to_complex)
{
	try { normal_1D_backward_user_defined_scale_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_non_unit_stride_and_distance_real_to_hermitian()
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
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { normal_1D_non_unit_stride_and_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { normal_1D_non_unit_stride_and_distance_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_user_defined_scale_real_to_hermitian()
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

	data_pattern pattern = impulse;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness, 42.0f );
}

TEST_F(accuracy_test_pow2_single, normal_1D_user_defined_scale_real_to_hermitian)
{
	try { normal_1D_user_defined_scale_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_user_defined_scale_real_to_hermitian)
{
	try { normal_1D_user_defined_scale_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_non_unit_stride_and_distance_hermitian_to_real()
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
	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_non_unit_stride_and_distance_hermitian_to_real)
{
	try { normal_1D_non_unit_stride_and_distance_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_non_unit_stride_and_distance_hermitian_to_real)
{
	try { normal_1D_non_unit_stride_and_distance_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_non_unit_stride_real_to_hermitian()
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

	size_t input_distance = 0;
	size_t output_distance = 0;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_non_unit_stride_real_to_hermitian)
{
	try { small_2D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_non_unit_stride_real_to_hermitian)
{
	try { small_2D_non_unit_stride_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_non_unit_distance_real_to_hermitian()
{
	std::vector<size_t> lengths;
	lengths.push_back( 4 );
	lengths.push_back( 4 );
	size_t batch = 2;

	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;

	size_t input_distance = lengths[0] * lengths[1] + 4;
	size_t output_distance = lengths[0] * lengths[1] + 5;

	layout::buffer_layout_t layout = layout::hermitian_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;

	data_pattern pattern = sawtooth;
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_non_unit_distance_real_to_hermitian)
{
	try { small_2D_non_unit_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_non_unit_distance_real_to_hermitian)
{
	try { small_2D_non_unit_distance_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_non_unit_stride_and_distance_real_to_hermitian()
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { small_2D_non_unit_stride_and_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { small_2D_non_unit_stride_and_distance_real_to_hermitian< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_non_unit_stride_and_distance_hermitian_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_non_unit_stride_and_distance_hermitian_to_real)
{
	try { small_2D_non_unit_stride_and_distance_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_non_unit_stride_and_distance_hermitian_to_real)
{
	try { small_2D_non_unit_stride_and_distance_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_user_defined_scale_hermitian_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness, 42.0f );
}

TEST_F(accuracy_test_pow2_single, normal_1D_user_defined_scale_hermitian_to_real)
{
	try { normal_1D_user_defined_scale_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_user_defined_scale_hermitian_to_real)
{
	try { normal_1D_user_defined_scale_hermitian_to_real< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void single_point_1D_forward_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 1 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = impulse;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f );
}

TEST_F(accuracy_test_pow2_single, single_point_1D_forward_complex_to_complex)
{
	try { single_point_1D_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, single_point_1D_forward_complex_to_complex)
{
	try { single_point_1D_forward_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void single_point_1D_backward_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 1 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = impulse;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f );
}

TEST_F(accuracy_test_pow2_single, single_point_1D_backward_complex_to_complex)
{
	try { single_point_1D_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, single_point_1D_backward_complex_to_complex)
{
	try { single_point_1D_backward_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_non_unit_stride_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	input_strides.push_back( 3 );
	output_strides.push_back( 3 );
	input_strides.push_back( lengths[0] * input_strides[0] + 20 );
	output_strides.push_back( lengths[0] * output_strides[0] + 20 );
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_non_unit_stride_complex_to_complex)
{
	try { small_2D_non_unit_stride_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_non_unit_stride_complex_to_complex)
{
	try { small_2D_non_unit_stride_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_2D_non_unit_stride_and_distance_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 2;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	input_strides.push_back( 42 );
	output_strides.push_back( 42 );
	input_strides.push_back( lengths[0] * input_strides[0] + 19 );
	output_strides.push_back( lengths[0] * output_strides[0] + 19 );
	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 14;
	size_t output_distance = lengths[lengths.size()-1] * output_strides[output_strides.size()-1] + 14;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_2D_non_unit_stride_and_distance_complex_to_complex)
{
	try { small_2D_non_unit_stride_and_distance_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_2D_non_unit_stride_and_distance_complex_to_complex)
{
	try { small_2D_non_unit_stride_and_distance_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_forward_user_defined_scale_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 42.0f );
}

TEST_F(accuracy_test_pow2_single, normal_2D_forward_user_defined_scale_complex_to_complex)
{
	try { normal_2D_forward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_forward_user_defined_scale_complex_to_complex)
{
	try { normal_2D_forward_user_defined_scale_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_backward_user_defined_scale_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( normal2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 42.5f );
}

TEST_F(accuracy_test_pow2_single, normal_2D_backward_user_defined_scale_complex_to_complex)
{
	try { normal_2D_backward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_backward_user_defined_scale_complex_to_complex)
{
	try { normal_2D_backward_user_defined_scale_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void rectangular_2D_array_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( normal2 );
	size_t batch = 8;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, rectangular_2D_array_complex_to_complex)
{
	try { rectangular_2D_array_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, rectangular_2D_array_complex_to_complex)
{
	try { rectangular_2D_array_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_array_complex_to_complex_with_odd_batch_size()
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_2D_array_complex_to_complex_with_odd_batch_size)
{
	try { normal_2D_array_complex_to_complex_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_array_complex_to_complex_with_odd_batch_size)
{
	try { normal_2D_array_complex_to_complex_with_odd_batch_size< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_array_forward_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( large2 );
	size_t batch = 8;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_array_forward_complex_to_complex)
{
	try { large_2D_array_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_array_forward_complex_to_complex)
{
	try { large_2D_array_forward_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_2D_array_backward_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( large2 );
	size_t batch = 8;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::in_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, large_2D_array_backward_complex_to_complex)
{
	try { large_2D_array_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_2D_array_backward_complex_to_complex)
{
	try { large_2D_array_backward_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void single_point_2D_forward_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 1 );
	lengths.push_back( 1 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = impulse;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f );
}

TEST_F(accuracy_test_pow2_single, single_point_2D_forward_complex_to_complex)
{
	try { single_point_2D_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, single_point_2D_forward_complex_to_complex)
{
	try { single_point_2D_forward_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void single_point_2D_backward_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 1 );
	lengths.push_back( 1 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = impulse;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f );
}

TEST_F(accuracy_test_pow2_single, single_point_2D_backward_complex_to_complex)
{
	try { single_point_2D_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, single_point_2D_backward_complex_to_complex)
{
	try { single_point_2D_backward_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void single_point_3D_forward_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 1 );
	lengths.push_back( 1 );
	lengths.push_back( 1 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = impulse;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f );
}

TEST_F(accuracy_test_pow2_single, single_point_3D_forward_complex_to_complex)
{
	try { single_point_3D_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, single_point_3D_forward_complex_to_complex)
{
	try { single_point_3D_forward_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void single_point_3D_backward_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( 1 );
	lengths.push_back( 1 );
	lengths.push_back( 1 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	std::vector<size_t> output_strides;
	size_t input_distance = 0;
	size_t output_distance = 0;
	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = impulse;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f );
}

TEST_F(accuracy_test_pow2_single, single_point_3D_backward_complex_to_complex)
{
	try { single_point_3D_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, single_point_3D_backward_complex_to_complex)
{
	try { single_point_3D_backward_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_non_unit_stride_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 1;
	std::vector<size_t> input_strides;
	input_strides.push_back( 2 );
	input_strides.push_back( lengths[0] * input_strides[0] + 20 );
	input_strides.push_back( lengths[1] * input_strides[1] + 17 );

	std::vector<size_t> output_strides( input_strides );

	size_t input_distance = 0;
	size_t output_distance = input_distance;

	layout::buffer_layout_t in_layout = layout::complex_planar;
	layout::buffer_layout_t out_layout = layout::complex_interleaved;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_non_unit_stride_complex_to_complex)
{
	try { small_3D_non_unit_stride_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_non_unit_stride_complex_to_complex)
{
	try { small_3D_non_unit_stride_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_non_unit_stride_and_distance_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 2;
	std::vector<size_t> input_strides;
	input_strides.push_back( 2 );
	input_strides.push_back( lengths[0] * input_strides[0] + 19 );
	input_strides.push_back( lengths[1] * input_strides[1] + 3 );

	size_t input_distance = lengths[lengths.size()-1] * input_strides[input_strides.size()-1] + 14;

	std::vector<size_t> output_strides( input_strides );
	size_t output_distance = input_distance;

	layout::buffer_layout_t in_layout = layout::complex_interleaved;
	layout::buffer_layout_t out_layout = layout::complex_planar;
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow2_single, small_3D_non_unit_stride_and_distance_complex_to_complex)
{
	try { small_3D_non_unit_stride_and_distance_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_non_unit_stride_and_distance_complex_to_complex)
{
	try { small_3D_non_unit_stride_and_distance_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

 // *****************************************************
 // *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_round_trip_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 1;
	layout::buffer_layout_t layout = layout::complex_interleaved;

	data_pattern pattern = sawtooth;
	complex_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch, layout );
}

TEST_F(accuracy_test_pow2_single, normal_1D_round_trip_complex_to_complex)
{
	try { normal_1D_round_trip_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_round_trip_complex_to_complex)
{
	try { normal_1D_round_trip_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

 // *****************************************************
 // *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_round_trip_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( normal2 );
	size_t batch = 1;
	layout::buffer_layout_t layout = layout::complex_planar;

	data_pattern pattern = sawtooth;
	complex_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch, layout );
}

template< class T, class cl_T, class fftw_T >
void testcase_2D_round_trip_complex_to_complex(size_t l0, size_t l1)
{
	std::vector<size_t> lengths;
	lengths.push_back( l0 );
	lengths.push_back( l1 );
	size_t batch = 1;
	layout::buffer_layout_t layout = layout::complex_planar;

	data_pattern pattern = sawtooth;
	complex_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch, layout );
}

// added this regression test to catch failures seen in transposes
TEST_F(accuracy_test_pow2_single, testcase1_2D_round_trip_complex_to_complex)
{
	try { testcase_2D_round_trip_complex_to_complex< float, cl_float, fftwf_complex >(1024, 16); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_single, normal_2D_round_trip_complex_to_complex)
{
	try { normal_2D_round_trip_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_round_trip_complex_to_complex)
{
	try { normal_2D_round_trip_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

 // *****************************************************
 // *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_round_trip_complex_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 1;
	layout::buffer_layout_t layout = layout::complex_planar;

	data_pattern pattern = sawtooth;
	complex_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch, layout );
}

TEST_F(accuracy_test_pow2_single, small_3D_round_trip_complex_to_complex)
{
	try { small_3D_round_trip_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_round_trip_complex_to_complex)
{
	try { small_3D_round_trip_complex_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

 // *****************************************************
 // *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_round_trip_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	size_t batch = 1;

	data_pattern pattern = impulse;
	real_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch );
}

TEST_F(accuracy_test_pow2_single, normal_1D_round_trip_real_to_complex)
{
	try { normal_1D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_1D_round_trip_real_to_complex)
{
	try { normal_1D_round_trip_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

 // *****************************************************
 // *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_round_trip_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( large2 );
	size_t batch = 1;

	data_pattern pattern = impulse;
	real_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch );
}

TEST_F(accuracy_test_pow2_single, large_1D_round_trip_real_to_complex)
{
	try { large_1D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, large_1D_round_trip_real_to_complex)
{
	try { large_1D_round_trip_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

 // *****************************************************
 // *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_round_trip_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( normal2 );
	lengths.push_back( normal2 );
	size_t batch = 1;

	data_pattern pattern = impulse;
	real_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch );
}

TEST_F(accuracy_test_pow2_single, normal_2D_round_trip_real_to_complex)
{
	try { normal_2D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, normal_2D_round_trip_real_to_complex)
{
	try { normal_2D_round_trip_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

 // *****************************************************
 // *****************************************************
template< class T, class cl_T, class fftw_T >
void small_3D_round_trip_real_to_complex()
{
	std::vector<size_t> lengths;
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	lengths.push_back( small2 );
	size_t batch = 1;

	data_pattern pattern = impulse;
	real_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch );
}

TEST_F(accuracy_test_pow2_single, small_3D_round_trip_real_to_complex)
{
	try { small_3D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow2_double, small_3D_round_trip_real_to_complex)
{
	try { small_3D_round_trip_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

} //namespace
