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

namespace precallback
{
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

}