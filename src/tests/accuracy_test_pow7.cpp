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
class accuracy_test_pow7_single : public ::testing::Test {
protected:
    accuracy_test_pow7_single() {}
    virtual ~accuracy_test_pow7_single() {}
    virtual void SetUp() {}
    virtual void TearDown() {
    }
};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_pow7_double : public ::testing::Test {
protected:
    accuracy_test_pow7_double() {}
    virtual ~accuracy_test_pow7_double() {}
    virtual void SetUp() {}
    virtual void TearDown() {
    }
};

namespace power7
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_backward_in_place_complex_planar_to_complex_planar()
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
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_1D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_1D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_forward_out_of_place_complex_planar_to_complex_planar()
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
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_1D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_1D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_backward_out_of_place_complex_planar_to_complex_planar()
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_1D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_1D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_1D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_1D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar()
    {
        std::vector<size_t> lengths;
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_1D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_1D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_in_place_real_to_hermitian_interleaved)
    {
        try { normal_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_in_place_real_to_hermitian_interleaved)
    {
        try { normal_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_in_place_hermitian_interleaved_to_real)
    {
        try { normal_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_in_place_hermitian_interleaved_to_real)
    {
        try { normal_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_out_of_place_real_to_hermitian_interleaved()
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_out_of_place_real_to_hermitian_interleaved)
    {
        try { normal_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_out_of_place_real_to_hermitian_interleaved)
    {
        try { normal_1D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_out_of_place_hermitian_interleaved_to_real)
    {
        try { normal_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_out_of_place_hermitian_interleaved_to_real)
    {
        try { normal_1D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_out_of_place_real_to_hermitian_planar()
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_out_of_place_real_to_hermitian_planar)
    {
        try { normal_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_out_of_place_real_to_hermitian_planar)
    {
        try { normal_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_out_of_place_hermitian_planar_to_real)
    {
        try { normal_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_out_of_place_hermitian_planar_to_real)
    {
        try { normal_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { small_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { small_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { small_1D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { small_1D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_1D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_forward_out_of_place_complex_planar_to_complex_planar()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_1D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_1D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_backward_out_of_place_complex_planar_to_complex_planar()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_1D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_1D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_forward_out_of_place_complex_planar_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_1D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_1D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_backward_out_of_place_complex_planar_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_1D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_1D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_forward_out_of_place_complex_interleaved_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_1D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_1D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_backward_out_of_place_complex_interleaved_to_complex_planar()
    {
        std::vector<size_t> lengths;
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_1D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_1D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_in_place_real_to_hermitian_interleaved()
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_in_place_real_to_hermitian_interleaved)
    {
        try { small_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_in_place_real_to_hermitian_interleaved)
    {
        try { small_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_in_place_hermitian_interleaved_to_real()
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_in_place_hermitian_interleaved_to_real)
    {
        try { small_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_in_place_hermitian_interleaved_to_real)
    {
        try { small_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_out_of_place_real_to_hermitian_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_out_of_place_real_to_hermitian_interleaved)
    {
        try { small_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_out_of_place_real_to_hermitian_interleaved)
    {
        try { small_1D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_out_of_place_hermitian_interleaved_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_out_of_place_hermitian_interleaved_to_real)
    {
        try { small_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_out_of_place_hermitian_interleaved_to_real)
    {
        try { small_1D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_out_of_place_real_to_hermitian_planar)
    {
        try { small_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_out_of_place_real_to_hermitian_planar)
    {
        try { small_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_out_of_place_hermitian_planar_to_real()
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_out_of_place_hermitian_planar_to_real)
    {
        try { small_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_out_of_place_hermitian_planar_to_real)
    {
        try { small_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
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
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { large_1D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { large_1D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_backward_in_place_complex_planar_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { large_1D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { large_1D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_1D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_forward_out_of_place_complex_planar_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_1D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_1D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_backward_out_of_place_complex_planar_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_1D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_1D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_1D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_1D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_forward_out_of_place_complex_planar_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_1D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_1D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_backward_out_of_place_complex_planar_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_1D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_1D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_forward_out_of_place_complex_interleaved_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_1D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_1D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_backward_out_of_place_complex_interleaved_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_1D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_1D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_in_place_real_to_hermitian_interleaved)
    {
        try { large_1D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_in_place_real_to_hermitian_interleaved)
    {
        try { large_1D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_in_place_hermitian_interleaved_to_real)
    {
        try { large_1D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_in_place_hermitian_interleaved_to_real)
    {
        try { large_1D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_out_of_place_real_to_hermitian_interleaved()
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_out_of_place_real_to_hermitian_interleaved)
    {
        try { large_1D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_out_of_place_real_to_hermitian_interleaved)
    {
        try { large_1D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_out_of_place_hermitian_interleaved_to_real()
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_out_of_place_hermitian_interleaved_to_real)
    {
        try { large_1D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_out_of_place_hermitian_interleaved_to_real)
    {
        try { large_1D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_out_of_place_real_to_hermitian_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(large7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_planar;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_out_of_place_real_to_hermitian_planar)
    {
        try { large_1D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_out_of_place_real_to_hermitian_planar)
    {
        try { large_1D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_out_of_place_hermitian_planar_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(large7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_planar;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_out_of_place_hermitian_planar_to_real)
    {
        try { large_1D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_out_of_place_hermitian_planar_to_real)
    {
        try { large_1D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
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
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_2D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_2D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_2D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_2D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::in_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_2D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::in_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_2D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_forward_out_of_place_complex_planar_to_complex_planar()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_2D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_2D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_backward_out_of_place_complex_planar_to_complex_planar()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_2D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_2D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_2D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_2D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_2D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar()
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
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_2D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_in_place_real_to_hermitian_interleaved()
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
        placeness::placeness_t placeness = placeness::in_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_in_place_real_to_hermitian_interleaved)
    {
        try { normal_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_in_place_real_to_hermitian_interleaved)
    {
        try { normal_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_in_place_hermitian_interleaved_to_real()
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
        placeness::placeness_t placeness = placeness::in_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_in_place_hermitian_interleaved_to_real)
    {
        try { normal_2D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_in_place_hermitian_interleaved_to_real)
    {
        try { normal_2D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_out_of_place_real_to_hermitian_interleaved()
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_out_of_place_real_to_hermitian_interleaved)
    {
        try { normal_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_out_of_place_real_to_hermitian_interleaved)
    {
        try { normal_2D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_out_of_place_hermitian_interleaved_to_real)
    {
        try { normal_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_out_of_place_hermitian_interleaved_to_real)
    {
        try { normal_2D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_out_of_place_real_to_hermitian_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
        lengths.push_back(normal7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_planar;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_out_of_place_real_to_hermitian_planar)
    {
        try { normal_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_out_of_place_real_to_hermitian_planar)
    {
        try { normal_2D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_out_of_place_hermitian_planar_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
        lengths.push_back(normal7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_planar;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_out_of_place_hermitian_planar_to_real)
    {
        try { normal_2D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_out_of_place_hermitian_planar_to_real)
    {
        try { normal_2D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { small_2D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { small_2D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_backward_in_place_complex_planar_to_complex_planar()
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
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { small_2D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { small_2D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_forward_in_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::in_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_2D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::in_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_2D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_2D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_forward_out_of_place_complex_planar_to_complex_planar()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_2D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_2D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_backward_out_of_place_complex_planar_to_complex_planar()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_2D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_2D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_forward_out_of_place_complex_planar_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_2D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_2D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_backward_out_of_place_complex_planar_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_2D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_2D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_forward_out_of_place_complex_interleaved_to_complex_planar()
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
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_2D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_2D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_backward_out_of_place_complex_interleaved_to_complex_planar()
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_2D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_2D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_in_place_real_to_hermitian_interleaved)
    {
        try { small_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_in_place_real_to_hermitian_interleaved)
    {
        try { small_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_in_place_hermitian_interleaved_to_real()
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_in_place_hermitian_interleaved_to_real)
    {
        try { small_2D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_in_place_hermitian_interleaved_to_real)
    {
        try { small_2D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_out_of_place_real_to_hermitian_interleaved()
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
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_out_of_place_real_to_hermitian_interleaved)
    {
        try { small_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_out_of_place_real_to_hermitian_interleaved)
    {
        try { small_2D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_out_of_place_hermitian_interleaved_to_real()
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
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_out_of_place_hermitian_interleaved_to_real)
    {
        try { small_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_out_of_place_hermitian_interleaved_to_real)
    {
        try { small_2D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_out_of_place_real_to_hermitian_planar()
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_out_of_place_real_to_hermitian_planar)
    {
        try { small_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_out_of_place_real_to_hermitian_planar)
    {
        try { small_2D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_out_of_place_hermitian_planar_to_real)
    {
        try { small_2D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_out_of_place_hermitian_planar_to_real)
    {
        try { small_2D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
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
        lengths.push_back(MaxLength2D<T>(7));
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { large_2D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { large_2D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_backward_in_place_complex_planar_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(MaxLength2D<T>(7));
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { large_2D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { large_2D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_2D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_2D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_2D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_2D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_forward_out_of_place_complex_planar_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(MaxLength2D<T>(7));
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_2D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_2D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_backward_out_of_place_complex_planar_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(MaxLength2D<T>(7));
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_2D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_2D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_2D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_2D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_forward_out_of_place_complex_planar_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(MaxLength2D<T>(7));
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_2D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_2D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_backward_out_of_place_complex_planar_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(MaxLength2D<T>(7));
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_2D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_2D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_forward_out_of_place_complex_interleaved_to_complex_planar()
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
        layout::buffer_layout_t out_layout = layout::complex_planar;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_2D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_2D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_backward_out_of_place_complex_interleaved_to_complex_planar()
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
        layout::buffer_layout_t out_layout = layout::complex_planar;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_2D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_2D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_in_place_real_to_hermitian_interleaved()
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
        placeness::placeness_t placeness = placeness::in_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_in_place_real_to_hermitian_interleaved)
    {
        try { large_2D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_in_place_real_to_hermitian_interleaved)
    {
        try { large_2D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_in_place_hermitian_interleaved_to_real()
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
        placeness::placeness_t placeness = placeness::in_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_in_place_hermitian_interleaved_to_real)
    {
        try { large_2D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_in_place_hermitian_interleaved_to_real)
    {
        try { large_2D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_out_of_place_real_to_hermitian_interleaved)
    {
        try { large_2D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_out_of_place_real_to_hermitian_interleaved)
    {
        try { large_2D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_out_of_place_hermitian_interleaved_to_real)
    {
        try { large_2D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_out_of_place_hermitian_interleaved_to_real)
    {
        try { large_2D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_out_of_place_real_to_hermitian_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(MaxLength2D<T>(7));
        lengths.push_back(normal7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_planar;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_out_of_place_real_to_hermitian_planar)
    {
        try { large_2D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_out_of_place_real_to_hermitian_planar)
    {
        try { large_2D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_out_of_place_hermitian_planar_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(MaxLength2D<T>(7));
        lengths.push_back(normal7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_planar;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_out_of_place_hermitian_planar_to_real)
    {
        try { large_2D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_out_of_place_hermitian_planar_to_real)
    {
        try { large_2D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
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
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_3D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_3D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_backward_in_place_complex_planar_to_complex_planar()
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
        layout::buffer_layout_t in_layout = layout::complex_planar;
        layout::buffer_layout_t out_layout = layout::complex_planar;
        placeness::placeness_t placeness = placeness::in_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_3D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { normal_3D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_3D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_3D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_forward_out_of_place_complex_planar_to_complex_planar()
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
        layout::buffer_layout_t in_layout = layout::complex_planar;
        layout::buffer_layout_t out_layout = layout::complex_planar;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_3D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_3D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_backward_out_of_place_complex_planar_to_complex_planar()
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
        layout::buffer_layout_t in_layout = layout::complex_planar;
        layout::buffer_layout_t out_layout = layout::complex_planar;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_3D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { normal_3D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { normal_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved()
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
        layout::buffer_layout_t in_layout = layout::complex_planar;
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_3D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved()
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
        layout::buffer_layout_t in_layout = layout::complex_planar;
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { normal_3D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar()
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
        layout::buffer_layout_t out_layout = layout::complex_planar;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_3D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar()
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
        layout::buffer_layout_t out_layout = layout::complex_planar;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { normal_3D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_in_place_real_to_hermitian_interleaved)
    {
        try { normal_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_in_place_real_to_hermitian_interleaved)
    {
        try { normal_3D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_in_place_hermitian_interleaved_to_real)
    {
        try { normal_3D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_in_place_hermitian_interleaved_to_real)
    {
        try { normal_3D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_out_of_place_real_to_hermitian_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(normal7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_out_of_place_real_to_hermitian_interleaved)
    {
        try { normal_3D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_out_of_place_real_to_hermitian_interleaved)
    {
        try { normal_3D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_out_of_place_hermitian_interleaved_to_real()
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
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_out_of_place_hermitian_interleaved_to_real)
    {
        try { normal_3D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_out_of_place_hermitian_interleaved_to_real)
    {
        try { normal_3D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_out_of_place_real_to_hermitian_planar()
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
        layout::buffer_layout_t layout = layout::hermitian_planar;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_out_of_place_real_to_hermitian_planar)
    {
        try { normal_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_out_of_place_real_to_hermitian_planar)
    {
        try { normal_3D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_3D_out_of_place_hermitian_planar_to_real()
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
        layout::buffer_layout_t layout = layout::hermitian_planar;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_3D_out_of_place_hermitian_planar_to_real)
    {
        try { normal_3D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_3D_out_of_place_hermitian_planar_to_real)
    {
        try { normal_3D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
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
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { small_3D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { small_3D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_backward_in_place_complex_planar_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { small_3D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { small_3D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_forward_in_place_complex_interleaved_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_3D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_3D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_backward_in_place_complex_interleaved_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_3D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_3D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_forward_out_of_place_complex_planar_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_3D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_3D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_backward_out_of_place_complex_planar_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_3D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { small_3D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { small_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_forward_out_of_place_complex_planar_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_3D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_3D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_backward_out_of_place_complex_planar_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_3D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { small_3D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_forward_out_of_place_complex_interleaved_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_3D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_3D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_backward_out_of_place_complex_interleaved_to_complex_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_3D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { small_3D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_in_place_real_to_hermitian_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, _small_3D_in_place_real_to_hermitian_interleaved)
    {
        try { small_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, _small_3D_in_place_real_to_hermitian_interleaved)
    {
        try { small_3D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_in_place_hermitian_interleaved_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, _small_3D_in_place_hermitian_interleaved_to_real)
    {
        try { small_3D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, _small_3D_in_place_hermitian_interleaved_to_real)
    {
        try { small_3D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_out_of_place_real_to_hermitian_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, _small_3D_out_of_place_real_to_hermitian_interleaved)
    {
        try { small_3D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, _small_3D_out_of_place_real_to_hermitian_interleaved)
    {
        try { small_3D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_out_of_place_hermitian_interleaved_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, _small_3D_out_of_place_hermitian_interleaved_to_real)
    {
        try { small_3D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, _small_3D_out_of_place_hermitian_interleaved_to_real)
    {
        try { small_3D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_out_of_place_real_to_hermitian_planar()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, _small_3D_out_of_place_real_to_hermitian_planar)
    {
        try { small_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, _small_3D_out_of_place_real_to_hermitian_planar)
    {
        try { small_3D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_out_of_place_hermitian_planar_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, _small_3D_out_of_place_hermitian_planar_to_real)
    {
        try { small_3D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, _small_3D_out_of_place_hermitian_planar_to_real)
    {
        try { small_3D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
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
        placeness::placeness_t placeness = placeness::in_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { large_3D_forward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_forward_in_place_complex_planar_to_complex_planar)
    {
        try { large_3D_forward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_backward_in_place_complex_planar_to_complex_planar()
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
        placeness::placeness_t placeness = placeness::in_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { large_3D_backward_in_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_backward_in_place_complex_planar_to_complex_planar)
    {
        try { large_3D_backward_in_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_forward_in_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t in_layout = layout::complex_interleaved;
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::in_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_3D_forward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_forward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_3D_forward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_backward_in_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t in_layout = layout::complex_interleaved;
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::in_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_3D_backward_in_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_backward_in_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_3D_backward_in_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_3D_forward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_forward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_3D_forward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_backward_out_of_place_complex_planar_to_complex_planar()
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
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_3D_backward_out_of_place_complex_planar_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_backward_out_of_place_complex_planar_to_complex_planar)
    {
        try { large_3D_backward_out_of_place_complex_planar_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t in_layout = layout::complex_interleaved;
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_3D_forward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved()
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
        layout::buffer_layout_t in_layout = layout::complex_interleaved;
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved)
    {
        try { large_3D_backward_out_of_place_complex_interleaved_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_forward_out_of_place_complex_planar_to_complex_interleaved()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_3D_forward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_forward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_3D_forward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_backward_out_of_place_complex_planar_to_complex_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(3);
        lengths.push_back(3);
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_3D_backward_out_of_place_complex_planar_to_complex_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_backward_out_of_place_complex_planar_to_complex_interleaved)
    {
        try { large_3D_backward_out_of_place_complex_planar_to_complex_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_forward_out_of_place_complex_interleaved_to_complex_planar()
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
        layout::buffer_layout_t in_layout = layout::complex_interleaved;
        layout::buffer_layout_t out_layout = layout::complex_planar;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_3D_forward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_forward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_3D_forward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_backward_out_of_place_complex_interleaved_to_complex_planar()
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
        layout::buffer_layout_t in_layout = layout::complex_interleaved;
        layout::buffer_layout_t out_layout = layout::complex_planar;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_3D_backward_out_of_place_complex_interleaved_to_complex_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_backward_out_of_place_complex_interleaved_to_complex_planar)
    {
        try { large_3D_backward_out_of_place_complex_interleaved_to_complex_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_in_place_real_to_hermitian_interleaved()
    {
        std::vector<size_t> lengths;
        lengths.push_back(3);
        lengths.push_back(3);
        lengths.push_back(large7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::in_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_in_place_real_to_hermitian_interleaved)
    {
        try { large_3D_in_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_in_place_real_to_hermitian_interleaved)
    {
        try { large_3D_in_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_in_place_hermitian_interleaved_to_real()
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
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::in_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_in_place_hermitian_interleaved_to_real)
    {
        try { large_3D_in_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_in_place_hermitian_interleaved_to_real)
    {
        try { large_3D_in_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_out_of_place_real_to_hermitian_interleaved()
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
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_out_of_place_real_to_hermitian_interleaved)
    {
        try { large_3D_out_of_place_real_to_hermitian_interleaved< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_out_of_place_real_to_hermitian_interleaved)
    {
        try { large_3D_out_of_place_real_to_hermitian_interleaved< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_3D_out_of_place_hermitian_interleaved_to_real()
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
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_out_of_place_hermitian_interleaved_to_real)
    {
        try { large_3D_out_of_place_hermitian_interleaved_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_out_of_place_hermitian_interleaved_to_real)
    {
        try { large_3D_out_of_place_hermitian_interleaved_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_out_of_place_real_to_hermitian_planar)
    {
        try { large_3D_out_of_place_real_to_hermitian_planar< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_out_of_place_real_to_hermitian_planar)
    {
        try { large_3D_out_of_place_real_to_hermitian_planar< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_3D_out_of_place_hermitian_planar_to_real)
    {
        try { large_3D_out_of_place_hermitian_planar_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_3D_out_of_place_hermitian_planar_to_real)
    {
        try { large_3D_out_of_place_hermitian_planar_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
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
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_array_complex_to_complex)
    {
        try { normal_1D_array_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_array_complex_to_complex)
    {
        try { normal_1D_array_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_array_complex_to_complex_with_odd_batch_size()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_array_complex_to_complex_with_odd_batch_size)
    {
        try { normal_1D_array_complex_to_complex_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_array_complex_to_complex_with_odd_batch_size)
    {
        try { normal_1D_array_complex_to_complex_with_odd_batch_size< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_array_real_to_hermitian)
    {
        try { normal_1D_array_real_to_hermitian< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_array_real_to_hermitian)
    {
        try { normal_1D_array_real_to_hermitian< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_array_real_to_hermitian_with_odd_batch_size)
    {
        try { normal_1D_array_real_to_hermitian_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_array_real_to_hermitian_with_odd_batch_size)
    {
        try { normal_1D_array_real_to_hermitian_with_odd_batch_size< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_array_hermitian_to_real()
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_array_hermitian_to_real)
    {
        try { normal_1D_array_hermitian_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_array_hermitian_to_real)
    {
        try { normal_1D_array_hermitian_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_array_hermitian_to_real_with_odd_batch_size()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
        size_t batch = 5;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_planar;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_array_hermitian_to_real_with_odd_batch_size)
    {
        try { normal_1D_array_hermitian_to_real_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_array_hermitian_to_real_with_odd_batch_size)
    {
        try { normal_1D_array_hermitian_to_real_with_odd_batch_size< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_array_real_to_hermitian()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 8;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::in_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_array_real_to_hermitian)
    {
        try { small_2D_array_real_to_hermitian< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_array_real_to_hermitian)
    {
        try { small_2D_array_real_to_hermitian< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_array_real_to_hermitian_with_odd_batch_size()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 5;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::in_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_array_real_to_hermitian_with_odd_batch_size)
    {
        try { small_2D_array_real_to_hermitian_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_array_real_to_hermitian_with_odd_batch_size)
    {
        try { small_2D_array_real_to_hermitian_with_odd_batch_size< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_array_hermitian_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 8;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::in_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_array_hermitian_to_real)
    {
        try { small_2D_array_hermitian_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_array_hermitian_to_real)
    {
        try { small_2D_array_hermitian_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_array_hermitian_to_real_with_odd_batch_size()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 5;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_planar;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_array_hermitian_to_real_with_odd_batch_size)
    {
        try { small_2D_array_hermitian_to_real_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_array_hermitian_to_real_with_odd_batch_size)
    {
        try { small_2D_array_hermitian_to_real_with_odd_batch_size< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_array_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_array_complex_to_complex)
    {
        try { large_1D_array_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_array_complex_to_complex)
    {
        try { large_1D_array_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void astoundingly_large_1D_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(2187);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, DISABLED_astoundingly_large_1D_complex_to_complex)
    {
        try { astoundingly_large_1D_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, DISABLED_astoundingly_large_1D_complex_to_complex)
    {
        try { astoundingly_large_1D_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void very_small_1D_non_unit_stride_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(9);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        input_strides.push_back(3);
        output_strides.push_back(3);
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t in_layout = layout::complex_planar;
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, very_small_1D_non_unit_stride_complex_to_complex)
    {
        try { very_small_1D_non_unit_stride_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, very_small_1D_non_unit_stride_complex_to_complex)
    {
        try { very_small_1D_non_unit_stride_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_non_unit_stride_real_to_hermitian()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        input_strides.push_back(3);
        output_strides.push_back(3);
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_non_unit_stride_real_to_hermitian)
    {
        try { small_1D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_non_unit_stride_real_to_hermitian)
    {
        try { small_1D_non_unit_stride_real_to_hermitian< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_1D_non_unit_stride_hermitian_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        input_strides.push_back(3);
        output_strides.push_back(3);
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_1D_non_unit_stride_hermitian_to_real)
    {
        try { small_1D_non_unit_stride_hermitian_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_1D_non_unit_stride_hermitian_to_real)
    {
        try { small_1D_non_unit_stride_hermitian_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void very_small_1D_non_unit_stride_real_to_hermitian()
    {
        std::vector<size_t> lengths;
        lengths.push_back(27);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        input_strides.push_back(3);
        output_strides.push_back(3);
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, very_small_1D_non_unit_stride_real_to_hermitian)
    {
        try { very_small_1D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, very_small_1D_non_unit_stride_real_to_hermitian)
    {
        try { very_small_1D_non_unit_stride_real_to_hermitian< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void very_small_1D_non_unit_stride_hermitian_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(27);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        input_strides.push_back(3);
        output_strides.push_back(3);
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, very_small_1D_non_unit_stride_hermitian_to_real)
    {
        try { very_small_1D_non_unit_stride_hermitian_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, very_small_1D_non_unit_stride_hermitian_to_real)
    {
        try { very_small_1D_non_unit_stride_hermitian_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void very_very_small_1D_non_unit_stride_real_to_hermitian()
    {
        std::vector<size_t> lengths;
        lengths.push_back(9);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        input_strides.push_back(3);
        output_strides.push_back(3);
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = impulse;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, very_very_small_1D_non_unit_stride_real_to_hermitian)
    {
        try { very_very_small_1D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, very_very_small_1D_non_unit_stride_real_to_hermitian)
    {
        try { very_very_small_1D_non_unit_stride_real_to_hermitian< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void very_very_small_1D_non_unit_stride_hermitian_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(9);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        input_strides.push_back(3);
        output_strides.push_back(3);
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = impulse;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, very_very_small_1D_non_unit_stride_hermitian_to_real)
    {
        try { very_very_small_1D_non_unit_stride_hermitian_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, very_very_small_1D_non_unit_stride_hermitian_to_real)
    {
        try { very_very_small_1D_non_unit_stride_hermitian_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_non_unit_stride_and_distance_complex_to_complex)
    {
        try { normal_1D_non_unit_stride_and_distance_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_non_unit_stride_and_distance_complex_to_complex)
    {
        try { normal_1D_non_unit_stride_and_distance_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, very_small_1D_non_unit_stride_and_distance_real_to_complex)
    {
        try { very_small_1D_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, very_small_1D_non_unit_stride_and_distance_real_to_complex)
    {
        try { very_small_1D_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(9);
        size_t batch = 2;

        std::vector<size_t> input_strides;
        input_strides.push_back(16);

        size_t input_distance = lengths[lengths.size() - 1] * input_strides[input_strides.size() - 1] + 128;

        std::vector<size_t> output_strides;
        output_strides.push_back(2);

        size_t output_distance = lengths[lengths.size() - 1] * output_strides[output_strides.size() - 1] + 2;

        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = impulse;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
    {
        try { very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
    {
        try { very_small_1D_out_of_place_different_input_output_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(9);
        size_t batch = 2;

        std::vector<size_t> input_strides;
        input_strides.push_back(16);

        size_t input_distance = lengths[lengths.size() - 1] * input_strides[input_strides.size() - 1] + 128;

        std::vector<size_t> output_strides;
        output_strides.push_back(2);

        size_t output_distance = lengths[lengths.size() - 1] * output_strides[output_strides.size() - 1] + 2;

        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::in_place;

        data_pattern pattern = impulse;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
    {
        try { very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex)
    {
        try { very_small_1D_in_place_different_input_output_non_unit_stride_and_distance_real_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_forward_user_defined_scale_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 42.0f);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_forward_user_defined_scale_complex_to_complex)
    {
        try { normal_1D_forward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_forward_user_defined_scale_complex_to_complex)
    {
        try { normal_1D_forward_user_defined_scale_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_backward_user_defined_scale_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 42.5f);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_backward_user_defined_scale_complex_to_complex)
    {
        try { normal_1D_backward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_backward_user_defined_scale_complex_to_complex)
    {
        try { normal_1D_backward_user_defined_scale_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_non_unit_stride_and_distance_real_to_hermitian()
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_non_unit_stride_and_distance_real_to_hermitian)
    {
        try { normal_1D_non_unit_stride_and_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_non_unit_stride_and_distance_real_to_hermitian)
    {
        try { normal_1D_non_unit_stride_and_distance_real_to_hermitian< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness, 42.0f);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_user_defined_scale_real_to_hermitian)
    {
        try { normal_1D_user_defined_scale_real_to_hermitian< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_user_defined_scale_real_to_hermitian)
    {
        try { normal_1D_user_defined_scale_real_to_hermitian< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_non_unit_stride_and_distance_hermitian_to_real)
    {
        try { normal_1D_non_unit_stride_and_distance_hermitian_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_non_unit_stride_and_distance_hermitian_to_real)
    {
        try { normal_1D_non_unit_stride_and_distance_hermitian_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_non_unit_stride_real_to_hermitian()
    {
        std::vector<size_t> lengths;
        lengths.push_back(9);
        lengths.push_back(9);
        size_t batch = 2;

        std::vector<size_t> input_strides;
        input_strides.push_back(5);
        input_strides.push_back(lengths[0] * input_strides[0] + 1);

        std::vector<size_t> output_strides;
        output_strides.push_back(2);
        output_strides.push_back(lengths[0] * output_strides[0] + 2);

        size_t input_distance = 0;
        size_t output_distance = 0;

        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_non_unit_stride_real_to_hermitian)
    {
        try { small_2D_non_unit_stride_real_to_hermitian< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_non_unit_stride_real_to_hermitian)
    {
        try { small_2D_non_unit_stride_real_to_hermitian< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_non_unit_distance_real_to_hermitian()
    {
        std::vector<size_t> lengths;
        lengths.push_back(9);
        lengths.push_back(9);
        size_t batch = 2;

        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;

        size_t input_distance = lengths[0] * lengths[1] + 4;
        size_t output_distance = lengths[0] * lengths[1] + 5;

        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_non_unit_distance_real_to_hermitian)
    {
        try { small_2D_non_unit_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_non_unit_distance_real_to_hermitian)
    {
        try { small_2D_non_unit_distance_real_to_hermitian< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_non_unit_stride_and_distance_real_to_hermitian()
    {
        std::vector<size_t> lengths;
        lengths.push_back(9);
        lengths.push_back(9);
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
        real_to_complex<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_non_unit_stride_and_distance_real_to_hermitian)
    {
        try { small_2D_non_unit_stride_and_distance_real_to_hermitian< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_non_unit_stride_and_distance_real_to_hermitian)
    {
        try { small_2D_non_unit_stride_and_distance_real_to_hermitian< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_non_unit_stride_and_distance_hermitian_to_real()
    {
        std::vector<size_t> lengths;
        lengths.push_back(9);
        lengths.push_back(9);
        size_t batch = 2;

        std::vector<size_t> input_strides;
        input_strides.push_back(12);
        input_strides.push_back(lengths[0] * input_strides[0] + 9);

        std::vector<size_t> output_strides;
        output_strides.push_back(7);
        output_strides.push_back(lengths[0] * output_strides[0] + 32);

        size_t input_distance = lengths[lengths.size() - 1] * input_strides[input_strides.size() - 1] + 50;
        size_t output_distance = lengths[lengths.size() - 1] * output_strides[output_strides.size() - 1] + 60;

        layout::buffer_layout_t layout = layout::hermitian_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;

        data_pattern pattern = sawtooth;
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_non_unit_stride_and_distance_hermitian_to_real)
    {
        try { small_2D_non_unit_stride_and_distance_hermitian_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_non_unit_stride_and_distance_hermitian_to_real)
    {
        try { small_2D_non_unit_stride_and_distance_hermitian_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_user_defined_scale_hermitian_to_real()
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
        complex_to_real<T, cl_T, fftw_T>(pattern, lengths, batch, input_strides, output_strides, input_distance, output_distance, layout, placeness, 42.0f);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_user_defined_scale_hermitian_to_real)
    {
        try { normal_1D_user_defined_scale_hermitian_to_real< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_user_defined_scale_hermitian_to_real)
    {
        try { normal_1D_user_defined_scale_hermitian_to_real< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void single_point_1D_forward_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(1);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f);
    }

    TEST_F(accuracy_test_pow7_single, single_point_1D_forward_complex_to_complex)
    {
        try { single_point_1D_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, single_point_1D_forward_complex_to_complex)
    {
        try { single_point_1D_forward_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void single_point_1D_backward_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(1);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f);
    }

    TEST_F(accuracy_test_pow7_single, single_point_1D_backward_complex_to_complex)
    {
        try { single_point_1D_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, single_point_1D_backward_complex_to_complex)
    {
        try { single_point_1D_backward_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_non_unit_stride_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        input_strides.push_back(3);
        output_strides.push_back(3);
        input_strides.push_back(lengths[0] * input_strides[0] + 20);
        output_strides.push_back(lengths[0] * output_strides[0] + 20);
        size_t input_distance = 0;
        size_t output_distance = 0;
        layout::buffer_layout_t in_layout = layout::complex_planar;
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_non_unit_stride_complex_to_complex)
    {
        try { small_2D_non_unit_stride_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_non_unit_stride_complex_to_complex)
    {
        try { small_2D_non_unit_stride_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_2D_non_unit_stride_and_distance_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 2;
        std::vector<size_t> input_strides;
        std::vector<size_t> output_strides;
        input_strides.push_back(42);
        output_strides.push_back(42);
        input_strides.push_back(lengths[0] * input_strides[0] + 19);
        output_strides.push_back(lengths[0] * output_strides[0] + 19);
        size_t input_distance = lengths[lengths.size() - 1] * input_strides[input_strides.size() - 1] + 14;
        size_t output_distance = lengths[lengths.size() - 1] * output_strides[output_strides.size() - 1] + 14;
        layout::buffer_layout_t in_layout = layout::complex_interleaved;
        layout::buffer_layout_t out_layout = layout::complex_planar;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_2D_non_unit_stride_and_distance_complex_to_complex)
    {
        try { small_2D_non_unit_stride_and_distance_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_2D_non_unit_stride_and_distance_complex_to_complex)
    {
        try { small_2D_non_unit_stride_and_distance_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_forward_user_defined_scale_complex_to_complex()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 42.0f);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_forward_user_defined_scale_complex_to_complex)
    {
        try { normal_2D_forward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_forward_user_defined_scale_complex_to_complex)
    {
        try { normal_2D_forward_user_defined_scale_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_backward_user_defined_scale_complex_to_complex()
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
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::backward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 42.5f);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_backward_user_defined_scale_complex_to_complex)
    {
        try { normal_2D_backward_user_defined_scale_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_backward_user_defined_scale_complex_to_complex)
    {
        try { normal_2D_backward_user_defined_scale_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void rectangular_2D_array_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(normal7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, rectangular_2D_array_complex_to_complex)
    {
        try { rectangular_2D_array_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, rectangular_2D_array_complex_to_complex)
    {
        try { rectangular_2D_array_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_array_complex_to_complex_with_odd_batch_size()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
        lengths.push_back(small7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_array_complex_to_complex_with_odd_batch_size)
    {
        try { normal_2D_array_complex_to_complex_with_odd_batch_size< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_array_complex_to_complex_with_odd_batch_size)
    {
        try { normal_2D_array_complex_to_complex_with_odd_batch_size< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_array_forward_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_array_forward_complex_to_complex)
    {
        try { large_2D_array_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_array_forward_complex_to_complex)
    {
        try { large_2D_array_forward_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_2D_array_backward_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(large7);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, large_2D_array_backward_complex_to_complex)
    {
        try { large_2D_array_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_2D_array_backward_complex_to_complex)
    {
        try { large_2D_array_backward_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void single_point_2D_forward_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(1);
        lengths.push_back(1);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f);
    }

    TEST_F(accuracy_test_pow7_single, single_point_2D_forward_complex_to_complex)
    {
        try { single_point_2D_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, single_point_2D_forward_complex_to_complex)
    {
        try { single_point_2D_forward_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void single_point_2D_backward_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(1);
        lengths.push_back(1);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f);
    }

    TEST_F(accuracy_test_pow7_single, single_point_2D_backward_complex_to_complex)
    {
        try { single_point_2D_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, single_point_2D_backward_complex_to_complex)
    {
        try { single_point_2D_backward_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void single_point_3D_forward_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(1);
        lengths.push_back(1);
        lengths.push_back(1);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f);
    }

    TEST_F(accuracy_test_pow7_single, single_point_3D_forward_complex_to_complex)
    {
        try { single_point_3D_forward_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, single_point_3D_forward_complex_to_complex)
    {
        try { single_point_3D_forward_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void single_point_3D_backward_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(1);
        lengths.push_back(1);
        lengths.push_back(1);
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
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness, 0.42f);
    }

    TEST_F(accuracy_test_pow7_single, single_point_3D_backward_complex_to_complex)
    {
        try { single_point_3D_backward_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, single_point_3D_backward_complex_to_complex)
    {
        try { single_point_3D_backward_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_non_unit_stride_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 1;
        std::vector<size_t> input_strides;
        input_strides.push_back(2);
        input_strides.push_back(lengths[0] * input_strides[0] + 20);
        input_strides.push_back(lengths[1] * input_strides[1] + 17);

        std::vector<size_t> output_strides(input_strides);

        size_t input_distance = 0;
        size_t output_distance = input_distance;

        layout::buffer_layout_t in_layout = layout::complex_planar;
        layout::buffer_layout_t out_layout = layout::complex_interleaved;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_non_unit_stride_complex_to_complex)
    {
        try { small_3D_non_unit_stride_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_non_unit_stride_complex_to_complex)
    {
        try { small_3D_non_unit_stride_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_non_unit_stride_and_distance_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 2;
        std::vector<size_t> input_strides;
        input_strides.push_back(2);
        input_strides.push_back(lengths[0] * input_strides[0] + 19);
        input_strides.push_back(lengths[1] * input_strides[1] + 3);

        size_t input_distance = lengths[lengths.size() - 1] * input_strides[input_strides.size() - 1] + 14;

        std::vector<size_t> output_strides(input_strides);
        size_t output_distance = input_distance;

        layout::buffer_layout_t in_layout = layout::complex_interleaved;
        layout::buffer_layout_t out_layout = layout::complex_planar;
        placeness::placeness_t placeness = placeness::out_of_place;
        direction::direction_t direction = direction::forward;

        data_pattern pattern = sawtooth;
        complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_non_unit_stride_and_distance_complex_to_complex)
    {
        try { small_3D_non_unit_stride_and_distance_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_non_unit_stride_and_distance_complex_to_complex)
    {
        try { small_3D_non_unit_stride_and_distance_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_round_trip_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
        size_t batch = 1;
        layout::buffer_layout_t layout = layout::complex_interleaved;

        data_pattern pattern = sawtooth;
        complex_to_complex_round_trip<T, cl_T, fftw_T>(pattern, lengths, batch, layout);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_round_trip_complex_to_complex)
    {
        try { normal_1D_round_trip_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_round_trip_complex_to_complex)
    {
        try { normal_1D_round_trip_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_round_trip_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
        lengths.push_back(normal7);
        size_t batch = 1;
        layout::buffer_layout_t layout = layout::complex_planar;

        data_pattern pattern = sawtooth;
        complex_to_complex_round_trip<T, cl_T, fftw_T>(pattern, lengths, batch, layout);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_round_trip_complex_to_complex)
    {
        try { normal_2D_round_trip_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_round_trip_complex_to_complex)
    {
        try { normal_2D_round_trip_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_round_trip_complex_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 1;
        layout::buffer_layout_t layout = layout::complex_planar;

        data_pattern pattern = sawtooth;
        complex_to_complex_round_trip<T, cl_T, fftw_T>(pattern, lengths, batch, layout);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_round_trip_complex_to_complex)
    {
        try { small_3D_round_trip_complex_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_round_trip_complex_to_complex)
    {
        try { small_3D_round_trip_complex_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_1D_round_trip_real_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
        size_t batch = 1;

        data_pattern pattern = impulse;
        real_to_complex_round_trip<T, cl_T, fftw_T>(pattern, lengths, batch);
    }

    TEST_F(accuracy_test_pow7_single, normal_1D_round_trip_real_to_complex)
    {
        try { normal_1D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_1D_round_trip_real_to_complex)
    {
        try { normal_1D_round_trip_real_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void large_1D_round_trip_real_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(large7);
        size_t batch = 1;

        data_pattern pattern = impulse;
        real_to_complex_round_trip<T, cl_T, fftw_T>(pattern, lengths, batch);
    }

    TEST_F(accuracy_test_pow7_single, large_1D_round_trip_real_to_complex)
    {
        try { large_1D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, large_1D_round_trip_real_to_complex)
    {
        try { large_1D_round_trip_real_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void normal_2D_round_trip_real_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(normal7);
        lengths.push_back(normal7);
        size_t batch = 1;

        data_pattern pattern = impulse;
        real_to_complex_round_trip<T, cl_T, fftw_T>(pattern, lengths, batch);
    }

    TEST_F(accuracy_test_pow7_single, normal_2D_round_trip_real_to_complex)
    {
        try { normal_2D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, normal_2D_round_trip_real_to_complex)
    {
        try { normal_2D_round_trip_real_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************
    template< class T, class cl_T, class fftw_T >
    void small_3D_round_trip_real_to_complex()
    {
        std::vector<size_t> lengths;
        lengths.push_back(small7);
        lengths.push_back(small7);
        lengths.push_back(small7);
        size_t batch = 1;

        data_pattern pattern = impulse;
        real_to_complex_round_trip<T, cl_T, fftw_T>(pattern, lengths, batch);
    }

    TEST_F(accuracy_test_pow7_single, small_3D_round_trip_real_to_complex)
    {
        try { small_3D_round_trip_real_to_complex< float, cl_float, fftwf_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    TEST_F(accuracy_test_pow7_double, small_3D_round_trip_real_to_complex)
    {
        try { small_3D_round_trip_real_to_complex< double, cl_double, fftw_complex >(); }
        catch (const std::exception& err) { handle_exception(err); }
    }

    // *****************************************************
    // *****************************************************

    struct InpSizeParameters {
        unsigned int x_dim;
        unsigned int y_dim;
        unsigned int z_dim;
        clfftPrecision precision;

        InpSizeParameters(unsigned int ip_x_dim, unsigned int ip_y_dim, unsigned int ip_z_dim, clfftPrecision ip_precision)
        {
            x_dim = ip_x_dim;
            y_dim = ip_y_dim;
            z_dim = ip_z_dim;
            precision = ip_precision;
        }
    };

    class TestParameterGenerator {
    private:
        std::vector<InpSizeParameters> data_sets;
    public:
        TestParameterGenerator()
        {
            generate();
        }

        std::vector<InpSizeParameters> & parameter_sets() { return data_sets; }

    private:
        void generate(void) {

            size_t SP_MAX_LEN = 1 << 24;
            size_t DP_MAX_LEN = 1 << 22;
            int x, y, z, is_1D_parameters_pushed = 0;
            size_t max_pow7 = 8; /*because 7 ^ 9 is greater than SP_MAX_LEN*/

            /*Generate test parameters*/
            for ( z = 0; z <= max_pow7; z++)
            {
                for ( y = is_1D_parameters_pushed; y <= max_pow7; y++)
                {
                    for ( x = 1; x <= max_pow7; x++)
                    {
                        is_1D_parameters_pushed = 1;
                        if (pow(7,(x + y + z)) <= (SP_MAX_LEN))
                        {
                            data_sets.push_back(InpSizeParameters((unsigned int)pow(7 , x), (unsigned int)pow(7 , y), (unsigned int)pow(7 , z), CLFFT_SINGLE));
                        }
                        else
                        {
                            break;
                        }
                        if (pow(7,(x + y + z)) <= (DP_MAX_LEN))
                        {
                            data_sets.push_back(InpSizeParameters((unsigned int)pow(7 , x), (unsigned int)pow(7 , y), (unsigned int)pow(7 , z), CLFFT_DOUBLE));
                        }          
                    }
                }
            }
        }
    }; //class TestParameterGenerator

}; //namespace

  // *****************************************************
  // *****************************************************

class accuracy_test_pow7_all_ip_size : public ::testing::TestWithParam<power7::InpSizeParameters> {
    protected:
        accuracy_test_pow7_all_ip_size() {}
        virtual ~accuracy_test_pow7_all_ip_size() {}
        virtual void SetUp() {}
        virtual void TearDown() {}
};

template< class T, class cl_T, class fftw_T >
void accuracy_test_pow7_all_ip_size_in_place(power7::InpSizeParameters params)
{
    std::vector<size_t> lengths;
    if (params.x_dim > 1) lengths.push_back(params.x_dim);
    if (params.y_dim > 1) lengths.push_back(params.y_dim);
    if (params.z_dim > 1) lengths.push_back(params.z_dim);

    size_t batch = (1 << 24) / (params.x_dim * params.y_dim * params.z_dim);
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;

    layout::buffer_layout_t in_layout = layout::complex_interleaved;
    layout::buffer_layout_t out_layout = layout::complex_interleaved;
    placeness::placeness_t placeness = placeness::in_place;
    direction::direction_t direction = direction::forward;

    data_pattern pattern = sawtooth;

    complex_to_complex<T, cl_T, fftw_T>(pattern, direction, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_layout, out_layout, placeness);
}

TEST_P(accuracy_test_pow7_all_ip_size, power7_all_input_size) {
    power7::InpSizeParameters params = GetParam();

    RecordProperty("x_dim_size", params.x_dim);
    RecordProperty("y_dim_size", params.y_dim);
    RecordProperty("z_dim_size", params.z_dim);
    RecordProperty("precision", params.precision);

    switch(params.precision )
    {
        case CLFFT_SINGLE:
            try { accuracy_test_pow7_all_ip_size_in_place< float, cl_float, fftwf_complex >(params); }
            catch (const std::exception& err) { handle_exception(err); }
            break;
        case CLFFT_DOUBLE:
            try { accuracy_test_pow7_all_ip_size_in_place< double, cl_double, fftw_complex >(params); }
            catch (const std::exception& err) { handle_exception(err); }
            break;
        default:
            FAIL() << "input parameter corruption in the test:accuracy_test_pow7_all_ip_size.";
    };

}

INSTANTIATE_TEST_CASE_P(
    clfft_pow7_AllInpSizeTest,
    accuracy_test_pow7_all_ip_size,
    ::testing::ValuesIn(power7::TestParameterGenerator
        ().parameter_sets())
    );
