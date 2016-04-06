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
class accuracy_test_pow3_single : public ::testing::Test {
protected:
	accuracy_test_pow3_single(){}
	virtual ~accuracy_test_pow3_single(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_pow3_double : public ::testing::Test {
protected:
	accuracy_test_pow3_double(){}
	virtual ~accuracy_test_pow3_double(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

namespace power3
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { normal_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_forward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_backward_in_place_complex_planar_to_complex_planar)
{
	try { normal_1D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_backward_in_place_complex_planar_to_complex_planar)
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_1D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_forward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_1D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_backward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_in_place_real_to_hermitian_interleaved)
{
	try { normal_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_in_place_real_to_hermitian_interleaved)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_in_place_hermitian_interleaved_to_real)
{
	try { normal_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_in_place_hermitian_interleaved_to_real)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { normal_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_out_of_place_real_to_hermitian_interleaved)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { normal_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_out_of_place_hermitian_interleaved_to_real)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_out_of_place_real_to_hermitian_planar)
{
	try { normal_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_out_of_place_real_to_hermitian_planar)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_out_of_place_hermitian_planar_to_real)
{
	try { normal_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_out_of_place_hermitian_planar_to_real)
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, small_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { small_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_forward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_backward_in_place_complex_planar_to_complex_planar)
{
	try { small_1D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_backward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_1D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_forward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_1D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_backward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_1D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_1D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_1D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_1D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_in_place_real_to_hermitian_interleaved)
{
	try { small_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_in_place_real_to_hermitian_interleaved)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_in_place_hermitian_interleaved_to_real)
{
	try { small_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_in_place_hermitian_interleaved_to_real)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { small_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_out_of_place_real_to_hermitian_interleaved)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { small_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_out_of_place_hermitian_interleaved_to_real)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_out_of_place_real_to_hermitian_planar)
{
	try { small_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_out_of_place_real_to_hermitian_planar)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_out_of_place_hermitian_planar_to_real)
{
	try { small_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_out_of_place_hermitian_planar_to_real)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { large_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_forward_in_place_complex_planar_to_complex_planar)
{
	try { large_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_backward_in_place_complex_planar_to_complex_planar()
{
	std::vector<size_t> lengths;
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_backward_in_place_complex_planar_to_complex_planar)
{
	try { large_1D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_backward_in_place_complex_planar_to_complex_planar)
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_1D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_forward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_1D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_backward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_1D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_1D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_1D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_1D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_in_place_real_to_hermitian_interleaved)
{
	try { large_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_in_place_real_to_hermitian_interleaved)
{
	try { large_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_in_place_hermitian_interleaved_to_real()
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow3_single, large_1D_in_place_hermitian_interleaved_to_real)
{
	try { large_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_in_place_hermitian_interleaved_to_real)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_out_of_place_real_to_hermitian_interleaved)
{
	try { large_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_out_of_place_real_to_hermitian_interleaved)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_out_of_place_hermitian_interleaved_to_real)
{
	try { large_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_out_of_place_hermitian_interleaved_to_real)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_out_of_place_real_to_hermitian_planar)
{
	try { large_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_out_of_place_real_to_hermitian_planar)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_out_of_place_hermitian_planar_to_real)
{
	try { large_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_out_of_place_hermitian_planar_to_real)
{
	try { large_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
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
	complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}
//TESTS disabled by default since they take a long time to execute
//TO enable this tests
//1. make sure ENV CLFFT_REQUEST_LIB_NOMEMALLOC=1
//2. pass --gtest_also_run_disabled_tests to TEST.exe
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_5292_10)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(5292, 10, layout::complex_planar, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}

//177147 = 243 * 243 * 3, backward and forward, planar and interleaved, single and double, batch size 1 and 3
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_177147_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_177147_1)
{
    try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 1, layout::complex_planar, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_177147_3)
{
    try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 3, layout::complex_planar, direction::forward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_177147_3)
{
    try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 3, layout::complex_planar, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_177147_1)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 1, layout::complex_planar, direction::forward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_177147_1)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 1, layout::complex_planar, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_177147_3)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 3, layout::complex_planar, direction::forward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_177147_3)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 3, layout::complex_planar, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}
//interleaved
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_177147_1)
{
    try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 1, layout::complex_interleaved, direction::forward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_177147_1)
{
    try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 1, layout::complex_interleaved, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_177147_3)
{
    try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 3, layout::complex_interleaved, direction::forward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_177147_3)
{
    try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(177147, 3, layout::complex_interleaved, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_177147_1)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 1, layout::complex_interleaved, direction::forward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_177147_1)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 1, layout::complex_interleaved, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_177147_3)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 3, layout::complex_interleaved, direction::forward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_177147_3)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(177147, 3, layout::complex_interleaved, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}

//1594323 = 729 * 729 * 3 backward and forward, planar and interleaved, single and double, batch size 1 and 3
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1594323_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_1594323_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 1, layout::complex_planar, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1594323_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_1594323_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 3, layout::complex_planar, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1594323_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_1594323_1)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 1, layout::complex_planar, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_1594323_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 3, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_1594323_3)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 3, layout::complex_planar, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}

//interleaved
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1594323_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_1594323_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 1, layout::complex_interleaved, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1594323_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 3, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_1594323_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(1594323, 3, layout::complex_interleaved, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1594323_1)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 1, layout::complex_interleaved, direction::forward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_1594323_1)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 1, layout::complex_interleaved, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_1594323_3)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 3, layout::complex_interleaved, direction::forward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_1594323_3)
{
    try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(1594323, 3, layout::complex_interleaved, direction::backward); }
    catch (const std::exception& err) { handle_exception(err); }
}

//14348907 = 2187 * 2187 * 3 backward and forward, planar and interleaved, single and double, batch size 1 and 3

TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_14348907_1)
{
    try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 1, layout::complex_planar, direction::forward); }
    catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_14348907_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 1, layout::complex_planar, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 1, layout::complex_planar, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_14348907_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_14348907_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 1, layout::complex_planar, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_planar_to_complex_planar_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 1, layout::complex_planar, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_planar_to_complex_planar_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 1, layout::complex_planar, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}
//interleaved
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_14348907_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_14348907_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 1, layout::complex_interleaved, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_single, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< float, cl_float, fftwf_complex >(14348907, 1, layout::complex_interleaved, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_14348907_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_14348907_1)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 1, layout::complex_interleaved, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_forward_in_place_complex_interleaved_to_complex_interleaved_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 1, layout::complex_interleaved, direction::forward); }
	catch (const std::exception& err) { handle_exception(err); }
}
TEST_F(accuracy_test_pow3_double, DISABLED_huge_1D_backward_in_place_complex_interleaved_to_complex_interleaved_14348907_3)
{
	try { huge_1D_forward_in_place_complex_to_complex< double, cl_double, fftw_complex >(14348907, 1, layout::complex_interleaved, direction::backward); }
	catch (const std::exception& err) { handle_exception(err); }
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_forward_in_place_complex_planar_to_complex_planar)
{
	try { normal_2D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_forward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_backward_in_place_complex_planar_to_complex_planar)
{
	try { normal_2D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_backward_in_place_complex_planar_to_complex_planar)
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_2D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_forward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_2D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_backward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_2D_in_place_real_to_hermitian_interleaved)
{
	try { normal_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_in_place_real_to_hermitian_interleaved)
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_2D_in_place_hermitian_interleaved_to_real)
{
	try { normal_2D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_in_place_hermitian_interleaved_to_real)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { normal_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_out_of_place_real_to_hermitian_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_out_of_place_hermitian_interleaved_to_real)
{
	try { normal_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_out_of_place_hermitian_interleaved_to_real)
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_2D_out_of_place_real_to_hermitian_planar)
{
	try { normal_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_out_of_place_real_to_hermitian_planar)
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_2D_out_of_place_hermitian_planar_to_real)
{
	try { normal_2D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_out_of_place_hermitian_planar_to_real)
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, small_2D_forward_in_place_complex_planar_to_complex_planar)
{
	try { small_2D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_forward_in_place_complex_planar_to_complex_planar)
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, small_2D_backward_in_place_complex_planar_to_complex_planar)
{
	try { small_2D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_backward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_2D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_forward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_2D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_backward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_2D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_2D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_2D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_2D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_in_place_real_to_hermitian_interleaved)
{
	try { small_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_in_place_real_to_hermitian_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_in_place_hermitian_interleaved_to_real)
{
	try { small_2D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_in_place_hermitian_interleaved_to_real)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { small_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_out_of_place_real_to_hermitian_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_out_of_place_hermitian_interleaved_to_real)
{
	try { small_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_out_of_place_hermitian_interleaved_to_real)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_out_of_place_real_to_hermitian_planar)
{
	try { small_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_out_of_place_real_to_hermitian_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_out_of_place_hermitian_planar_to_real)
{
	try { small_2D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_out_of_place_hermitian_planar_to_real)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_forward_in_place_complex_planar_to_complex_planar)
{
	try { large_2D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_forward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_backward_in_place_complex_planar_to_complex_planar)
{
	try { large_2D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_backward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( MaxLength2D<T>(3) );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, large_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_2D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_forward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_2D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_backward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_2D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_2D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_2D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_2D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_in_place_real_to_hermitian_interleaved)
{
	try { large_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_in_place_real_to_hermitian_interleaved)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_in_place_hermitian_interleaved_to_real)
{
	try { large_2D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_in_place_hermitian_interleaved_to_real)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_out_of_place_real_to_hermitian_interleaved)
{
	try { large_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_out_of_place_real_to_hermitian_interleaved)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_out_of_place_hermitian_interleaved_to_real)
{
	try { large_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_out_of_place_hermitian_interleaved_to_real)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_out_of_place_real_to_hermitian_planar)
{
	try { large_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_out_of_place_real_to_hermitian_planar)
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
	lengths.push_back( MaxLength2D<T>(3) );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_out_of_place_hermitian_planar_to_real)
{
	try { large_2D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_out_of_place_hermitian_planar_to_real)
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_3D_forward_in_place_complex_planar_to_complex_planar)
{
	try { normal_3D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_forward_in_place_complex_planar_to_complex_planar)
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_3D_backward_in_place_complex_planar_to_complex_planar)
{
	try { normal_3D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_backward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::forward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_3D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_3D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_forward_out_of_place_complex_planar_to_complex_planar)
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
	placeness::placeness_t placeness = placeness::out_of_place;
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_3D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { normal_3D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_backward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_in_place_real_to_hermitian_interleaved)
{
	try { normal_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_in_place_real_to_hermitian_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_in_place_hermitian_interleaved_to_real)
{
	try { normal_3D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_in_place_hermitian_interleaved_to_real)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_out_of_place_real_to_hermitian_interleaved)
{
	try { normal_3D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_out_of_place_real_to_hermitian_interleaved)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_out_of_place_hermitian_interleaved_to_real)
{
	try { normal_3D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_out_of_place_hermitian_interleaved_to_real)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_3D_out_of_place_real_to_hermitian_planar)
{
	try { normal_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_out_of_place_real_to_hermitian_planar)
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
	complex_to_real<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow3_single, normal_3D_out_of_place_hermitian_planar_to_real)
{
	try { normal_3D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_3D_out_of_place_hermitian_planar_to_real)
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
	lengths.push_back( small3 );
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
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, small_3D_forward_in_place_complex_planar_to_complex_planar)
{
	try { small_3D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_forward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( small3 );
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
	direction::direction_t direction = direction::backward;

	data_pattern pattern = sawtooth;
	complex_to_complex<T, cl_T, fftw_T>( pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness );
}

TEST_F(accuracy_test_pow3_single, small_3D_backward_in_place_complex_planar_to_complex_planar)
{
	try { small_3D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_backward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_3D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_forward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { small_3D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_backward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_3D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { small_3D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_3D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { small_3D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, _small_3D_in_place_real_to_hermitian_interleaved)
{
	try { small_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, _small_3D_in_place_real_to_hermitian_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, _small_3D_in_place_hermitian_interleaved_to_real)
{
	try { small_3D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, _small_3D_in_place_hermitian_interleaved_to_real)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, _small_3D_out_of_place_real_to_hermitian_interleaved)
{
	try { small_3D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, _small_3D_out_of_place_real_to_hermitian_interleaved)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, _small_3D_out_of_place_hermitian_interleaved_to_real)
{
	try { small_3D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, _small_3D_out_of_place_hermitian_interleaved_to_real)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, _small_3D_out_of_place_real_to_hermitian_planar)
{
	try { small_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, _small_3D_out_of_place_real_to_hermitian_planar)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, _small_3D_out_of_place_hermitian_planar_to_real)
{
	try { small_3D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, _small_3D_out_of_place_hermitian_planar_to_real)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_forward_in_place_complex_planar_to_complex_planar)
{
	try { large_3D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_forward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_backward_in_place_complex_planar_to_complex_planar)
{
	try { large_3D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_backward_in_place_complex_planar_to_complex_planar)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_forward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_3D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_forward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_backward_out_of_place_complex_planar_to_complex_planar)
{
	try { large_3D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_backward_out_of_place_complex_planar_to_complex_planar)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
{
	try { large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_3D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( 3 );
	lengths.push_back( 3 );
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
{
	try { large_3D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
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
	lengths.push_back( 3 );
	lengths.push_back( large3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_3D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
{
	try { large_3D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
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
	lengths.push_back( 3 );
	lengths.push_back( 3 );
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_in_place_real_to_hermitian_interleaved)
{
	try { large_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_in_place_real_to_hermitian_interleaved)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_in_place_hermitian_interleaved_to_real)
{
	try { large_3D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_in_place_hermitian_interleaved_to_real)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_out_of_place_real_to_hermitian_interleaved)
{
	try { large_3D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_out_of_place_real_to_hermitian_interleaved)
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
	lengths.push_back( large3 );
	lengths.push_back( 3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_out_of_place_hermitian_interleaved_to_real)
{
	try { large_3D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_out_of_place_hermitian_interleaved_to_real)
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow3_single, large_3D_out_of_place_real_to_hermitian_planar)
{
	try { large_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_out_of_place_real_to_hermitian_planar)
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
	lengths.push_back( 3 );
	lengths.push_back( large3 );
	lengths.push_back( 3 );
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

TEST_F(accuracy_test_pow3_single, large_3D_out_of_place_hermitian_planar_to_real)
{
	try { large_3D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_3D_out_of_place_hermitian_planar_to_real)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_array_complex_to_complex)
{
	try { normal_1D_array_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_array_complex_to_complex)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_array_complex_to_complex_with_odd_batch_size)
{
	try { normal_1D_array_complex_to_complex_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_array_complex_to_complex_with_odd_batch_size)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_array_real_to_hermitian)
{
	try { normal_1D_array_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_array_real_to_hermitian)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_array_real_to_hermitian_with_odd_batch_size)
{
	try { normal_1D_array_real_to_hermitian_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_array_real_to_hermitian_with_odd_batch_size)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_array_hermitian_to_real)
{
	try { normal_1D_array_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_array_hermitian_to_real)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_array_hermitian_to_real_with_odd_batch_size)
{
	try { normal_1D_array_hermitian_to_real_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_array_hermitian_to_real_with_odd_batch_size)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_array_real_to_hermitian)
{
	try { small_2D_array_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_array_real_to_hermitian)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_array_real_to_hermitian_with_odd_batch_size)
{
	try { small_2D_array_real_to_hermitian_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_array_real_to_hermitian_with_odd_batch_size)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_array_hermitian_to_real)
{
	try { small_2D_array_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_array_hermitian_to_real)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_array_hermitian_to_real_with_odd_batch_size)
{
	try { small_2D_array_hermitian_to_real_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_array_hermitian_to_real_with_odd_batch_size)
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
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_1D_array_complex_to_complex)
{
	try { large_1D_array_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_array_complex_to_complex)
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
	lengths.push_back( 2187 );
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

TEST_F(accuracy_test_pow3_single, DISABLED_astoundingly_large_1D_complex_to_complex)
{
	try { astoundingly_large_1D_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, DISABLED_astoundingly_large_1D_complex_to_complex)
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
	lengths.push_back( 9 );
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

TEST_F(accuracy_test_pow3_single, very_small_1D_non_unit_stride_complex_to_complex)
{
	try { very_small_1D_non_unit_stride_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, very_small_1D_non_unit_stride_complex_to_complex)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_non_unit_stride_real_to_hermitian)
{
	try {small_1D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_non_unit_stride_real_to_hermitian)
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
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_1D_non_unit_stride_hermitian_to_real)
{
	try { small_1D_non_unit_stride_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_1D_non_unit_stride_hermitian_to_real)
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
	lengths.push_back( 27 );
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

TEST_F(accuracy_test_pow3_single, very_small_1D_non_unit_stride_real_to_hermitian)
{
	try {very_small_1D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, very_small_1D_non_unit_stride_real_to_hermitian)
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
	lengths.push_back( 27 );
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

TEST_F(accuracy_test_pow3_single, very_small_1D_non_unit_stride_hermitian_to_real)
{
	try { very_small_1D_non_unit_stride_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, very_small_1D_non_unit_stride_hermitian_to_real)
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
	lengths.push_back( 9 );
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

TEST_F(accuracy_test_pow3_single, very_very_small_1D_non_unit_stride_real_to_hermitian)
{
	try {very_very_small_1D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, very_very_small_1D_non_unit_stride_real_to_hermitian)
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
	lengths.push_back( 9 );
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

TEST_F(accuracy_test_pow3_single, very_very_small_1D_non_unit_stride_hermitian_to_real)
{
	try { very_very_small_1D_non_unit_stride_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, very_very_small_1D_non_unit_stride_hermitian_to_real)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_non_unit_stride_and_distance_complex_to_complex)
{
	try { normal_1D_non_unit_stride_and_distance_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_non_unit_stride_and_distance_complex_to_complex)
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow3_single, very_small_1D_non_unit_stride_and_distance_real_to_complex)
{
	try { very_small_1D_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, very_small_1D_non_unit_stride_and_distance_real_to_complex)
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow3_single, very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
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
	real_to_complex<T, cl_T, fftw_T>( pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness );
}

TEST_F(accuracy_test_pow3_single, very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
{
	try { very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_forward_user_defined_scale_complex_to_complex)
{
	try { normal_1D_forward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_forward_user_defined_scale_complex_to_complex)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_backward_user_defined_scale_complex_to_complex)
{
	try { normal_1D_backward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_backward_user_defined_scale_complex_to_complex)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { normal_1D_non_unit_stride_and_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_non_unit_stride_and_distance_real_to_hermitian)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_user_defined_scale_real_to_hermitian)
{
	try { normal_1D_user_defined_scale_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_user_defined_scale_real_to_hermitian)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_non_unit_stride_and_distance_hermitian_to_real)
{
	try { normal_1D_non_unit_stride_and_distance_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_non_unit_stride_and_distance_hermitian_to_real)
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
	lengths.push_back( 9 );
	lengths.push_back( 9 );
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

TEST_F(accuracy_test_pow3_single, small_2D_non_unit_stride_real_to_hermitian)
{
	try { small_2D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_non_unit_stride_real_to_hermitian)
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
	lengths.push_back( 9 );
	lengths.push_back( 9 );
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

TEST_F(accuracy_test_pow3_single, small_2D_non_unit_distance_real_to_hermitian)
{
	try { small_2D_non_unit_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_non_unit_distance_real_to_hermitian)
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
	lengths.push_back( 9 );
	lengths.push_back( 9 );
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

TEST_F(accuracy_test_pow3_single, small_2D_non_unit_stride_and_distance_real_to_hermitian)
{
	try { small_2D_non_unit_stride_and_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_non_unit_stride_and_distance_real_to_hermitian)
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
	lengths.push_back( 9 );
	lengths.push_back( 9 );
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

TEST_F(accuracy_test_pow3_single, small_2D_non_unit_stride_and_distance_hermitian_to_real)
{
	try { small_2D_non_unit_stride_and_distance_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_non_unit_stride_and_distance_hermitian_to_real)
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
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_1D_user_defined_scale_hermitian_to_real)
{
	try { normal_1D_user_defined_scale_hermitian_to_real< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_user_defined_scale_hermitian_to_real)
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

TEST_F(accuracy_test_pow3_single, single_point_1D_forward_complex_to_complex)
{
	try { single_point_1D_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, single_point_1D_forward_complex_to_complex)
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

TEST_F(accuracy_test_pow3_single, single_point_1D_backward_complex_to_complex)
{
	try { single_point_1D_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, single_point_1D_backward_complex_to_complex)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_non_unit_stride_complex_to_complex)
{
	try { small_2D_non_unit_stride_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_non_unit_stride_complex_to_complex)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_2D_non_unit_stride_and_distance_complex_to_complex)
{
	try { small_2D_non_unit_stride_and_distance_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_2D_non_unit_stride_and_distance_complex_to_complex)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_forward_user_defined_scale_complex_to_complex)
{
	try { normal_2D_forward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_forward_user_defined_scale_complex_to_complex)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_backward_user_defined_scale_complex_to_complex)
{
	try { normal_2D_backward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_backward_user_defined_scale_complex_to_complex)
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
	lengths.push_back( small3 );
	lengths.push_back( normal3 );
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

TEST_F(accuracy_test_pow3_single, rectangular_2D_array_complex_to_complex)
{
	try { rectangular_2D_array_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, rectangular_2D_array_complex_to_complex)
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
	lengths.push_back( normal3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, normal_2D_array_complex_to_complex_with_odd_batch_size)
{
	try { normal_2D_array_complex_to_complex_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_array_complex_to_complex_with_odd_batch_size)
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
	lengths.push_back( small3 );
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_array_forward_complex_to_complex)
{
	try { large_2D_array_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_array_forward_complex_to_complex)
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
	lengths.push_back( small3 );
	lengths.push_back( large3 );
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

TEST_F(accuracy_test_pow3_single, large_2D_array_backward_complex_to_complex)
{
	try { large_2D_array_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_2D_array_backward_complex_to_complex)
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

TEST_F(accuracy_test_pow3_single, single_point_2D_forward_complex_to_complex)
{
	try { single_point_2D_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, single_point_2D_forward_complex_to_complex)
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

TEST_F(accuracy_test_pow3_single, single_point_2D_backward_complex_to_complex)
{
	try { single_point_2D_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, single_point_2D_backward_complex_to_complex)
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

TEST_F(accuracy_test_pow3_single, single_point_3D_forward_complex_to_complex)
{
	try { single_point_3D_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, single_point_3D_forward_complex_to_complex)
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

TEST_F(accuracy_test_pow3_single, single_point_3D_backward_complex_to_complex)
{
	try { single_point_3D_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, single_point_3D_backward_complex_to_complex)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_non_unit_stride_complex_to_complex)
{
	try { small_3D_non_unit_stride_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_non_unit_stride_complex_to_complex)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
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

TEST_F(accuracy_test_pow3_single, small_3D_non_unit_stride_and_distance_complex_to_complex)
{
	try { small_3D_non_unit_stride_and_distance_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_non_unit_stride_and_distance_complex_to_complex)
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
	lengths.push_back( normal3 );
	size_t batch = 1;
	layout::buffer_layout_t layout = layout::complex_interleaved;

	data_pattern pattern = sawtooth;
	complex_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch, layout );
}

TEST_F(accuracy_test_pow3_single, normal_1D_round_trip_complex_to_complex)
{
	try { normal_1D_round_trip_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_round_trip_complex_to_complex)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
	size_t batch = 1;
	layout::buffer_layout_t layout = layout::complex_planar;

	data_pattern pattern = sawtooth;
	complex_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch, layout );
}

TEST_F(accuracy_test_pow3_single, normal_2D_round_trip_complex_to_complex)
{
	try { normal_2D_round_trip_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_round_trip_complex_to_complex)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	size_t batch = 1;
	layout::buffer_layout_t layout = layout::complex_planar;

	data_pattern pattern = sawtooth;
	complex_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch, layout );
}

TEST_F(accuracy_test_pow3_single, small_3D_round_trip_complex_to_complex)
{
	try { small_3D_round_trip_complex_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_round_trip_complex_to_complex)
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
	lengths.push_back( normal3 );
	size_t batch = 1;

	data_pattern pattern = impulse;
	real_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch );
}

TEST_F(accuracy_test_pow3_single, normal_1D_round_trip_real_to_complex)
{
	try { normal_1D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_1D_round_trip_real_to_complex)
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
	lengths.push_back( large3 );
	size_t batch = 1;

	data_pattern pattern = impulse;
	real_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch );
}

TEST_F(accuracy_test_pow3_single, large_1D_round_trip_real_to_complex)
{
	try { large_1D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, large_1D_round_trip_real_to_complex)
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
	lengths.push_back( normal3 );
	lengths.push_back( normal3 );
	size_t batch = 1;

	data_pattern pattern = impulse;
	real_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch );
}

TEST_F(accuracy_test_pow3_single, normal_2D_round_trip_real_to_complex)
{
	try { normal_2D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, normal_2D_round_trip_real_to_complex)
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
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	lengths.push_back( small3 );
	size_t batch = 1;

	data_pattern pattern = impulse;
	real_to_complex_round_trip<T, cl_T, fftw_T>( pattern, lengths, batch );
}

TEST_F(accuracy_test_pow3_single, small_3D_round_trip_real_to_complex)
{
	try { small_3D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

TEST_F(accuracy_test_pow3_double, small_3D_round_trip_real_to_complex)
{
	try { small_3D_round_trip_real_to_complex< double, cl_double, fftw_complex >(); }
	catch( const std::exception& err ) { handle_exception(err);	}
}

} //namespace
