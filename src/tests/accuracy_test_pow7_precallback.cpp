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
class accuracy_test_pow7_precallback_single : public ::testing::Test {
protected:
	accuracy_test_pow7_precallback_single(){}
	virtual ~accuracy_test_pow7_precallback_single(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_pow7_precallback_double : public ::testing::Test {
protected:
	accuracy_test_pow7_precallback_double(){}
	virtual ~accuracy_test_pow7_precallback_double(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

namespace precallback_power7
{
/**********************************************************************************************
**************************************Complex To Complex***************************************
**********************************************************************************************/
#pragma region Complex_To_Complex
// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_forward_in_place_complex_planar_to_complex_planar()
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
    placeness::placeness_t placeness = placeness::in_place;
    direction::direction_t direction = direction::forward;

    data_pattern pattern = sawtooth;
    precallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_forward_in_place_complex_planar_to_complex_planar)
{
    try { normal_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_forward_in_place_complex_planar_to_complex_planar)
{
    try { normal_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

	// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
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
    precallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_1D_non_unit_stride_and_distance_complex_to_complex()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    size_t batch = 2;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    input_strides.push_back(42);
    output_strides.push_back(42);
    size_t input_distance = lengths[lengths.size() - 1] * input_strides[input_strides.size() - 1] + 14;
    size_t output_distance = lengths[lengths.size() - 1] * output_strides[output_strides.size() - 1] + 14;
    layout::buffer_layout_t in_layout = layout::complex_planar;
    layout::buffer_layout_t out_layout = layout::complex_interleaved;
    placeness::placeness_t placeness = placeness::out_of_place;
    direction::direction_t direction = direction::forward;

    data_pattern pattern = sawtooth;
    precallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_non_unit_stride_and_distance_complex_to_complex)
{
    try { normal_1D_non_unit_stride_and_distance_complex_to_complex< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_non_unit_stride_and_distance_complex_to_complex)
{
    try { normal_1D_non_unit_stride_and_distance_complex_to_complex< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void small_1D_backward_in_place_complex_planar_to_complex_planar()
{
    std::vector<size_t> lengths;
    lengths.push_back(small7);
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
    precallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, small_1D_backward_in_place_complex_planar_to_complex_planar)
{
    try { small_1D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, small_1D_backward_in_place_complex_planar_to_complex_planar)
{
    try { small_1D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_1D_round_trip_complex_to_complex()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    size_t batch = 1;
    layout::buffer_layout_t layout = layout::complex_interleaved;

    data_pattern pattern = sawtooth;
    precallback_complex_to_complex_round_trip<T, cl_T, fftw_T>(pattern, lengths, batch, layout);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_round_trip_complex_to_complex)
{
    try { normal_1D_round_trip_complex_to_complex< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_round_trip_complex_to_complex)
{
    try { normal_1D_round_trip_complex_to_complex< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void large_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
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
    precallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_2D_backward_in_place_complex_planar_to_complex_planar()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    lengths.push_back(normal7);
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
    precallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_2D_backward_in_place_complex_planar_to_complex_planar)
{
    try { normal_2D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_2D_backward_in_place_complex_planar_to_complex_planar)
{
    try { normal_2D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    lengths.push_back(normal7);
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
    precallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
    try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
{
    try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void large_2D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back(MaxLength2D<T>(7));
    lengths.push_back(normal7);
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
    precallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, large_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { large_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, large_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { large_2D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void small_2D_forward_in_place_complex_planar_to_complex_planar()
{
    std::vector<size_t> lengths;
    lengths.push_back(small7);
    lengths.push_back(small7);
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
    precallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, small_2D_forward_in_place_complex_planar_to_complex_planar)
{
    try { small_2D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, small_2D_forward_in_place_complex_planar_to_complex_planar)
{
    try { small_2D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    lengths.push_back(small7);
    lengths.push_back(small7);
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
    precallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void large_3D_forward_out_of_place_complex_planar_to_complex_planar()
{
    std::vector<size_t> lengths;
    lengths.push_back(large7);
    lengths.push_back(3);
    lengths.push_back(3);
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
    precallback_complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, large_3D_forward_out_of_place_complex_planar_to_complex_planar)
{
    try { large_3D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, large_3D_forward_out_of_place_complex_planar_to_complex_planar)
{
    try { large_3D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

#pragma endregion

/**********************************************************************************************
**************************************Complex To Real***************************************
**********************************************************************************************/
#pragma region Complex_To_Real

// *****************************************************
// *****************************************************
template< class T, class cl_T, class fftw_T >
void normal_1D_in_place_hermitian_interleaved_to_real()
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
    precallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_in_place_hermitian_interleaved_to_real)
{
    try { normal_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_in_place_hermitian_interleaved_to_real)
{
    try { normal_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_1D_out_of_place_hermitian_interleaved_to_real()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    precallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_out_of_place_hermitian_interleaved_to_real)
{
    try { normal_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_out_of_place_hermitian_interleaved_to_real)
{
    try { normal_1D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_1D_out_of_place_hermitian_planar_to_real()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_planar;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    precallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_out_of_place_hermitian_planar_to_real)
{
    try { normal_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_out_of_place_hermitian_planar_to_real)
{
    try { normal_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void large_1D_in_place_hermitian_interleaved_to_real()
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
    precallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, large_1D_in_place_hermitian_interleaved_to_real)
{
    try { large_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, large_1D_in_place_hermitian_interleaved_to_real)
{
    try { large_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_1D_non_unit_stride_and_distance_hermitian_to_real()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    size_t batch = 2;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    input_strides.push_back(42);
    output_strides.push_back(42);
    size_t input_distance = lengths[lengths.size() - 1] * input_strides[input_strides.size() - 1] + 14;
    size_t output_distance = lengths[lengths.size() - 1] * output_strides[output_strides.size() - 1] + 14;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    precallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_non_unit_stride_and_distance_hermitian_to_real)
{
    try { normal_1D_non_unit_stride_and_distance_hermitian_to_real< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_non_unit_stride_and_distance_hermitian_to_real)
{
    try { normal_1D_non_unit_stride_and_distance_hermitian_to_real< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_2D_out_of_place_hermitian_interleaved_to_real()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    lengths.push_back(normal7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    precallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_2D_out_of_place_hermitian_interleaved_to_real)
{
    try { normal_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_2D_out_of_place_hermitian_interleaved_to_real)
{
    try { normal_2D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void small_2D_out_of_place_hermitian_planar_to_real()
{
    std::vector<size_t> lengths;
    lengths.push_back(small7);
    lengths.push_back(small7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_planar;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    precallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, small_2D_out_of_place_hermitian_planar_to_real)
{
    try { small_2D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, small_2D_out_of_place_hermitian_planar_to_real)
{
    try { small_2D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void large_2D_out_of_place_hermitian_interleaved_to_real()
{
    std::vector<size_t> lengths;
    lengths.push_back(MaxLength2D<T>(7));
    lengths.push_back(normal7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    precallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, large_2D_out_of_place_hermitian_interleaved_to_real)
{
    try { large_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, large_2D_out_of_place_hermitian_interleaved_to_real)
{
    try { large_2D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_3D_in_place_hermitian_interleaved_to_real()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
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
    precallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_3D_in_place_hermitian_interleaved_to_real)
{
    try { normal_3D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_3D_in_place_hermitian_interleaved_to_real)
{
    try { normal_3D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void large_3D_out_of_place_hermitian_planar_to_real()
{
    std::vector<size_t> lengths;
    lengths.push_back(3);
    lengths.push_back(large7);
    lengths.push_back(3);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_planar;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    precallback_complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, large_3D_out_of_place_hermitian_planar_to_real)
{
    try { large_3D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, large_3D_out_of_place_hermitian_planar_to_real)
{
    try { large_3D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

#pragma endregion

/**********************************************************************************************
**************************************Real To Complex***************************************
**********************************************************************************************/
#pragma region Real_To_Complex

template< class T, class cl_T, class fftw_T >
void normal_1D_in_place_real_to_hermitian_interleaved()
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
    precallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_in_place_real_to_hermitian_interleaved)
{
    try { normal_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_in_place_real_to_hermitian_interleaved)
{
    try { normal_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void small_1D_out_of_place_real_to_hermitian_planar()
{
    std::vector<size_t> lengths;
    lengths.push_back(small7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_planar;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    precallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, small_1D_out_of_place_real_to_hermitian_planar)
{
    try { small_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, small_1D_out_of_place_real_to_hermitian_planar)
{
    try { small_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void large_1D_in_place_real_to_hermitian_interleaved()
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
    precallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, large_1D_in_place_real_to_hermitian_interleaved)
{
    try { large_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, large_1D_in_place_real_to_hermitian_interleaved)
{
    try { large_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void small_2D_in_place_real_to_hermitian_interleaved()
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
    precallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, small_2D_in_place_real_to_hermitian_interleaved)
{
    try { small_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, small_2D_in_place_real_to_hermitian_interleaved)
{
    try { small_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void large_2D_out_of_place_real_to_hermitian_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back(MaxLength2D<T>(7));
    lengths.push_back(normal7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    precallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, large_2D_out_of_place_real_to_hermitian_interleaved)
{
    try { large_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, large_2D_out_of_place_real_to_hermitian_interleaved)
{
    try { large_2D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_3D_in_place_real_to_hermitian_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back(small7);
    lengths.push_back(normal7);
    lengths.push_back(small7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::in_place;

    data_pattern pattern = sawtooth;
    precallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_3D_in_place_real_to_hermitian_interleaved)
{
    try { normal_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_3D_in_place_real_to_hermitian_interleaved)
{
    try { normal_3D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void large_3D_out_of_place_real_to_hermitian_planar()
{
    std::vector<size_t> lengths;
    lengths.push_back(large7);
    lengths.push_back(3);
    lengths.push_back(3);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_planar;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = sawtooth;
    precallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, large_3D_out_of_place_real_to_hermitian_planar)
{
    try { large_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, large_3D_out_of_place_real_to_hermitian_planar)
{
    try { large_3D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_1D_array_real_to_hermitian()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    size_t batch = 8;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::in_place;

    data_pattern pattern = sawtooth;
    precallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_array_real_to_hermitian)
{
    try { normal_1D_array_real_to_hermitian< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_array_real_to_hermitian)
{
    try { normal_1D_array_real_to_hermitian< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_1D_array_real_to_hermitian_with_odd_batch_size()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    size_t batch = 5;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::in_place;

    data_pattern pattern = sawtooth;
    precallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_array_real_to_hermitian_with_odd_batch_size)
{
    try { normal_1D_array_real_to_hermitian_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_array_real_to_hermitian_with_odd_batch_size)
{
    try { normal_1D_array_real_to_hermitian_with_odd_batch_size< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void very_small_1D_non_unit_stride_and_distance_real_to_complex()
{
    std::vector<size_t> lengths;
    lengths.push_back(9);
    size_t batch = 2;

    std::vector<size_t> input_strides;
    input_strides.push_back(2);

    size_t input_distance = lengths[lengths.size() - 1] * input_strides[input_strides.size() - 1] + 2;

    std::vector<size_t> output_strides(input_strides);
    size_t output_distance = input_distance;

    layout::buffer_layout_t layout = layout::hermitian_interleaved;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = impulse;
    precallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
}

TEST_F(accuracy_test_pow7_precallback_single, very_small_1D_non_unit_stride_and_distance_real_to_complex)
{
    try { very_small_1D_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, very_small_1D_non_unit_stride_and_distance_real_to_complex)
{
    try { very_small_1D_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

template< class T, class cl_T, class fftw_T >
void normal_1D_user_defined_scale_real_to_hermitian()
{
    std::vector<size_t> lengths;
    lengths.push_back(normal7);
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    layout::buffer_layout_t layout = layout::hermitian_planar;
    placeness::placeness_t placeness = placeness::out_of_place;

    data_pattern pattern = impulse;
    precallback_real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness, 42.0f);
}

TEST_F(accuracy_test_pow7_precallback_single, normal_1D_user_defined_scale_real_to_hermitian)
{
    try { normal_1D_user_defined_scale_real_to_hermitian< float, cl_float, fftwf_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

TEST_F(accuracy_test_pow7_precallback_double, normal_1D_user_defined_scale_real_to_hermitian)
{
    try { normal_1D_user_defined_scale_real_to_hermitian< double, cl_double, fftw_complex >(); }
    catch (const std::exception& err) { handle_exception(err); }
}

#pragma endregion
}