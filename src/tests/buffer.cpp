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

#include "buffer.h"

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class test_harness_tests_buffer : public ::testing::Test {
protected:
	test_harness_tests_buffer() {}
	virtual ~test_harness_tests_buffer(){
	}
	virtual void SetUp()
	{
		suppress_output = true;
	}
	virtual void TearDown()
	{
		suppress_output = false;
	}
};

/*****************************************************/
/*****************************************************/
TEST_F(test_harness_tests_buffer, even_length_hermitian_buffers_are_halfish_size) {
	size_t dimensions = 3;
	size_t lengths[3] = {8, 2, 4};
	size_t* stride_null = NULL;
	size_t batch = 3;
	size_t distance = 0;

	buffer<float> interleaved_buffer( dimensions, lengths, stride_null, batch, distance, layout::hermitian_interleaved, CLFFT_OUTOFPLACE );
	EXPECT_EQ( 5, interleaved_buffer.length(dimx) );
	EXPECT_EQ( 2, interleaved_buffer.length(dimy) );
	EXPECT_EQ( 4, interleaved_buffer.length(dimz) );

	buffer<float> planar_buffer( dimensions, lengths, stride_null, batch, distance, layout::hermitian_planar, CLFFT_OUTOFPLACE );
	EXPECT_EQ( 5, planar_buffer.length(dimx) );
	EXPECT_EQ( 2, planar_buffer.length(dimy) );
	EXPECT_EQ( 4, planar_buffer.length(dimz) );
}

/*****************************************************/
/*****************************************************/
TEST_F(test_harness_tests_buffer, odd_length_hermitian_buffers_are_halfish_size) {
	size_t dimensions = 3;
	size_t lengths[3] = {9, 2, 4};
	size_t* stride_null = NULL;
	size_t batch = 3;
	size_t distance = 0;

	buffer<float> interleaved_buffer( dimensions, lengths, stride_null, batch, distance, layout::hermitian_interleaved, CLFFT_OUTOFPLACE );
	EXPECT_EQ( 5, interleaved_buffer.length(dimx) );
	EXPECT_EQ( 2, interleaved_buffer.length(dimy) );
	EXPECT_EQ( 4, interleaved_buffer.length(dimz) );

	buffer<float> planar_buffer( dimensions, lengths, stride_null, batch, distance, layout::hermitian_planar, CLFFT_OUTOFPLACE );
	EXPECT_EQ( 5, planar_buffer.length(dimx) );
	EXPECT_EQ( 2, planar_buffer.length(dimy) );
	EXPECT_EQ( 4, planar_buffer.length(dimz) );
}

/*****************************************************/
/*****************************************************/
TEST_F(test_harness_tests_buffer, even_sized_in_place_real_buffers_should_have_padding) {
	// TODO for now, they should all have padding
	// eventually, this should be just in-place buffers
	size_t dimensions = 1;
	size_t lengths[1] = {8};
	size_t* stride_null = NULL;
	size_t batch = 3;
	size_t distance = 0;

	buffer<float> buffer_1d_float( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	// length of x should not change
	EXPECT_EQ( 8, buffer_1d_float.length(dimx) );
	// just the memory size should change
	EXPECT_EQ( 10 * batch * sizeof(float), buffer_1d_float.size_in_bytes() );

	buffer<double> buffer_1d_double( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	EXPECT_EQ( 8, buffer_1d_double.length(dimx) );
	EXPECT_EQ( 10 * batch * sizeof(double), buffer_1d_double.size_in_bytes() );

	dimensions = 2;
	size_t lengths2d[2] = {4, 2};
	buffer<float> buffer_2d_float( dimensions, lengths2d, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	EXPECT_EQ( 4, buffer_2d_float.length(dimx) );
	EXPECT_EQ( 6 * lengths2d[dimy] * batch * sizeof(float), buffer_2d_float.size_in_bytes() );

	buffer<double> buffer_2d_double( dimensions, lengths2d, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	EXPECT_EQ( 4, buffer_2d_double.length(dimx) );
	EXPECT_EQ( 6 * lengths2d[dimy] * batch * sizeof(double), buffer_2d_double.size_in_bytes() );

	dimensions = 3;
	size_t lengths3d[3] = {16, 8, 2};
	buffer<float> buffer_3d_float( dimensions, lengths3d, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	EXPECT_EQ( 16, buffer_3d_float.length(dimx) );
	EXPECT_EQ( 18 * lengths3d[dimy] * lengths3d[dimz] * batch * sizeof(float), buffer_3d_float.size_in_bytes() );

	buffer<double> buffer_3d_double( dimensions, lengths3d, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	EXPECT_EQ( 16, buffer_3d_double.length(dimx) );
	EXPECT_EQ( 18 * lengths3d[dimy] * lengths3d[dimz] * batch * sizeof(double), buffer_3d_double.size_in_bytes() );
}

/*****************************************************/
/*****************************************************/
TEST_F(test_harness_tests_buffer, odd_sized_in_place_real_buffers_should_have_padding) {
	// TODO for now, they should all have padding
	// eventually, this should be just in-place buffers
	size_t dimensions = 1;
	size_t lengths[1] = {15};
	size_t* stride_null = NULL;
	size_t batch = 3;
	size_t distance = 0;

	buffer<float> buffer_1d_float( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	buffer_1d_float.set_all_to_linear_increase();
	// length of x should not change
	EXPECT_EQ( 15, buffer_1d_float.length(dimx) );
	// just the memory size should change
	EXPECT_EQ( 16 * batch * sizeof(float), buffer_1d_float.size_in_bytes() );

	buffer<double> buffer_1d_double( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	buffer_1d_double.set_all_to_linear_increase();
	EXPECT_EQ( 15, buffer_1d_double.length(dimx) );
	EXPECT_EQ( 16 * batch * sizeof(double), buffer_1d_double.size_in_bytes() );

	dimensions = 2;
	size_t lengths2d[2] = {7, 2};
	buffer<float> buffer_2d_float( dimensions, lengths2d, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	buffer_2d_float.set_all_to_linear_increase();
	EXPECT_EQ( 7, buffer_2d_float.length(dimx) );
	EXPECT_EQ( 8 * lengths2d[dimy] * batch * sizeof(float), buffer_2d_float.size_in_bytes() );

	buffer<double> buffer_2d_double( dimensions, lengths2d, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	buffer_2d_double.set_all_to_linear_increase();
	EXPECT_EQ( 7, buffer_2d_double.length(dimx) );
	EXPECT_EQ( 8 * lengths2d[dimy] * batch * sizeof(double), buffer_2d_double.size_in_bytes() );

	dimensions = 3;
	size_t lengths3d[3] = {7, 4, 2};
	buffer<float> buffer_3d_float( dimensions, lengths3d, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	buffer_3d_float.set_all_to_linear_increase();
	EXPECT_EQ( 7, buffer_3d_float.length(dimx) );
	EXPECT_EQ( 8 * lengths3d[dimy] * lengths3d[dimz] * batch * sizeof(float), buffer_3d_float.size_in_bytes() );

	buffer<double> buffer_3d_double( dimensions, lengths3d, stride_null, batch, distance, layout::real, CLFFT_INPLACE );
	buffer_3d_double.set_all_to_linear_increase();
	EXPECT_EQ( 7, buffer_3d_double.length(dimx) );
	EXPECT_EQ( 8 * lengths3d[dimy] * lengths3d[dimz] * batch * sizeof(double), buffer_3d_double.size_in_bytes() );
}

/*****************************************************/
/*****************************************************/
TEST_F(test_harness_tests_buffer, real_imag_and_complex_functions_return_correct_values) {
	try
	{
		size_t dimensions = 1;
		size_t length = 8;
		size_t* stride_null = NULL;
		size_t batch = 1;
		size_t distance = 0;

		buffer<float> interleaved_buffer( dimensions, &length, stride_null, batch, distance, layout::complex_interleaved, CLFFT_OUTOFPLACE );
		interleaved_buffer.set_all_to_linear_increase();

		EXPECT_FLOAT_EQ( 1.0f, interleaved_buffer.real( 0 ) );
		EXPECT_FLOAT_EQ( 1.5f, interleaved_buffer.imag( 0 ) );
		EXPECT_FLOAT_EQ( 1.0f, interleaved_buffer.complex( 0 ).real() );
		EXPECT_FLOAT_EQ( 1.5f, interleaved_buffer.complex( 0 ).imag() );
		EXPECT_FLOAT_EQ( 4.0f, interleaved_buffer.real( 3 ) );
		EXPECT_FLOAT_EQ( 4.5f, interleaved_buffer.imag( 3 ) );
		EXPECT_FLOAT_EQ( 4.0f, interleaved_buffer.complex( 3 ).real() );
		EXPECT_FLOAT_EQ( 4.5f, interleaved_buffer.complex( 3 ).imag() );
		EXPECT_FLOAT_EQ( 8.0f, interleaved_buffer.real( 7 ) );
		EXPECT_FLOAT_EQ( 8.5f, interleaved_buffer.imag( 7 ) );
		EXPECT_FLOAT_EQ( 8.0f, interleaved_buffer.complex( 7 ).real() );
		EXPECT_FLOAT_EQ( 8.5f, interleaved_buffer.complex( 7 ).imag() );

		dimensions = 3;
		size_t lengths[3] = { 2, 4, 8 };
		batch = 2;
		buffer<float> real_buffer( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_OUTOFPLACE );
		real_buffer.set_all_to_linear_increase();

		EXPECT_FLOAT_EQ( 1.0f, real_buffer.real( 0, 0, 0, 0 ) );
		EXPECT_FLOAT_EQ( 4.0f, real_buffer.real( 1, 1, 0, 0 ) );
		EXPECT_FLOAT_EQ( 68.0f, real_buffer.real( 1, 1, 0, 1 ) );
		EXPECT_FLOAT_EQ( 56.0f, real_buffer.real( 1, 3, 6, 0 ) );
		EXPECT_FLOAT_EQ( 120.0f, real_buffer.real( 1, 3, 6, 1 ) );

		dimensions = 2;
		lengths[0] = 4;
		lengths[1] = 2;
		size_t strides[2] = {2, 11};
		distance = 30;

		buffer<double> planar_buffer( dimensions, lengths, strides, batch, distance, layout::complex_planar, CLFFT_OUTOFPLACE );
		planar_buffer.set_all_to_linear_increase();

		EXPECT_DOUBLE_EQ( 1.0f, planar_buffer.real( 0, 0, 0, 0 ) );
		EXPECT_DOUBLE_EQ( 1.5f, planar_buffer.imag( 0, 0, 0, 0 ) );
		EXPECT_DOUBLE_EQ( 1.0f, planar_buffer.complex( 0, 0, 0, 0 ).real() );
		EXPECT_DOUBLE_EQ( 1.5f, planar_buffer.complex( 0, 0, 0, 0 ).imag() );
		EXPECT_DOUBLE_EQ( 4.0f, planar_buffer.real( 3, 0, 0, 0 ) );
		EXPECT_DOUBLE_EQ( 4.5f, planar_buffer.imag( 3, 0, 0, 0 ) );
		EXPECT_DOUBLE_EQ( 4.0f, planar_buffer.complex( 3, 0, 0, 0 ).real() );
		EXPECT_DOUBLE_EQ( 4.5f, planar_buffer.complex( 3, 0, 0, 0 ).imag() );
		EXPECT_DOUBLE_EQ( 15.0f, planar_buffer.real( 2, 1, 0, 1 ) );
		EXPECT_DOUBLE_EQ( 15.5f, planar_buffer.imag( 2, 1, 0, 1 ) );
		EXPECT_DOUBLE_EQ( 15.0f, planar_buffer.complex( 2, 1, 0, 1 ).real() );
		EXPECT_DOUBLE_EQ( 15.5f, planar_buffer.complex( 2, 1, 0, 1 ).imag() );
	}
	catch( const std::exception& err )
	{
		handle_exception(err);
	}
}

/*****************************************************/
/*****************************************************/
TEST_F(test_harness_tests_buffer, equivalence_operator_returns_correct_result_with_pointwise_compare) {
	try
	{
		bool comparison_type_restore = comparison_type;
		comparison_type = pointwise_compare;

		size_t dimensions = 3;
		size_t lengths[3] = { 16, 32, 64 };
		size_t* stride_null = NULL;
		size_t batch = 2;
		size_t distance = 0;

		// complex test
		buffer<float> thing_1( dimensions, lengths, stride_null, batch, distance, layout::complex_interleaved, CLFFT_OUTOFPLACE );
		buffer<float> thing_2( dimensions, lengths, stride_null, batch, distance, layout::complex_planar, CLFFT_OUTOFPLACE );

		thing_1.set_all_to_sawtooth(1.0f);
		thing_2.set_all_to_sawtooth(1.0f);
		EXPECT_EQ( true, thing_1 == thing_2 );

		thing_2.set_one_data_point( 42.0f, 0.0f, 0, 0, 0, 0 );
		EXPECT_EQ( false, thing_1 == thing_2 );

		thing_1.set_all_to_sawtooth(1.0f);
		thing_2.set_all_to_sawtooth(1.0f);
		thing_2.set_one_data_point( 16.0f, 0.0f, 15, 31, 63, 1 );
		EXPECT_EQ( false, thing_1 == thing_2 );

		thing_1.set_all_to_sawtooth(1.0f);
		thing_2.set_all_to_sawtooth(1.0f);
		thing_2.set_one_data_point( 96.0f, 7.8f, 7, 16, 12, 1 );
		EXPECT_EQ( false, thing_1 == thing_2 );

		thing_1.set_all_to_sawtooth( 42.0f );
		thing_2.set_all_to_sawtooth( 42.0f );
		EXPECT_EQ( true, thing_1 == thing_2 );

		//---------------------------------------------------//

		// real test
		buffer<double> thing_3( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_OUTOFPLACE );
		buffer<double> thing_4( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_OUTOFPLACE );

		thing_3.set_all_to_sawtooth(1.0f);
		thing_4.set_all_to_sawtooth(1.0f);
		EXPECT_EQ( true, thing_3 == thing_4 );

		thing_4.set_one_data_point( 42.0f, 0, 0, 0, 0 );
		EXPECT_EQ( false, thing_3 == thing_4 );

		thing_3.set_all_to_sawtooth( 42.0f );
		thing_4.set_all_to_sawtooth( 42.0f );
		EXPECT_EQ( true, thing_3 == thing_4 );

		comparison_type = comparison_type_restore;
	}
	catch( const std::exception& err )
	{
		handle_exception(err);
	}
}

/*****************************************************/
/*****************************************************/
TEST_F(test_harness_tests_buffer, inequivalence_operator_returns_correct_result_with_pointwise_compare) {
	try
	{
		bool comparison_type_restore = comparison_type;
		comparison_type = pointwise_compare;

		size_t dimensions = 3;
		size_t lengths[3] = { 16, 32, 64 };
		size_t* stride_null = NULL;
		size_t batch = 2;
		size_t distance = 0;

		buffer<float> thing_1( dimensions, lengths, stride_null, batch, distance, layout::complex_interleaved, CLFFT_OUTOFPLACE );

		buffer<float> thing_2( dimensions, lengths, stride_null, batch, distance, layout::complex_planar, CLFFT_OUTOFPLACE );

		thing_1.set_all_to_sawtooth(1.0f);
		thing_2.set_all_to_sawtooth(1.0f);
		EXPECT_EQ( false, thing_1 != thing_2 );

		thing_2.set_one_data_point( 42.0f, 0, 0, 0, 0, 0 );
		EXPECT_EQ( true, thing_1 != thing_2 );

		thing_1.set_all_to_sawtooth( 42.0f );
		thing_2.set_all_to_sawtooth( 42.0f );
		EXPECT_EQ( false, thing_1 != thing_2 );

		comparison_type = comparison_type_restore;
	}
	catch( const std::exception& err )
	{
		handle_exception(err);
	}
}

/*****************************************************/
/*****************************************************/
TEST_F(test_harness_tests_buffer, equivalence_operator_returns_correct_result_with_rms) {
	try
	{
		bool comparison_type_restore = comparison_type;
		comparison_type = root_mean_square;

		size_t dimensions = 3;
		size_t lengths[3] = { 16, 32, 64 };
		size_t* stride_null = NULL;
		size_t batch = 2;
		size_t distance = 0;

		// complex test
		buffer<float> thing_1( dimensions, lengths, stride_null, batch, distance, layout::complex_interleaved, CLFFT_OUTOFPLACE );
		buffer<float> thing_2( dimensions, lengths, stride_null, batch, distance, layout::complex_planar, CLFFT_OUTOFPLACE );

		thing_1.set_all_to_sawtooth(1.0f);
		thing_2.set_all_to_sawtooth(1.0f);
		EXPECT_EQ( true, thing_1 == thing_2 );

		thing_2.set_one_data_point( 42.0f, 0.0f, 0, 0, 0, 0 );
		EXPECT_EQ( false, thing_1 == thing_2 );

		thing_1.set_all_to_sawtooth(1.0f);
		thing_2.set_all_to_sawtooth(1.0f);
		thing_2.set_one_data_point( 16.0f, 0.0f, 15, 31, 63, 1 );
		EXPECT_EQ( false, thing_1 == thing_2 );

		thing_1.set_all_to_sawtooth(1.0f);
		thing_2.set_all_to_sawtooth(1.0f);
		thing_2.set_one_data_point( 96.0f, 7.8f, 7, 16, 12, 1 );
		EXPECT_EQ( false, thing_1 == thing_2 );

		thing_1.set_all_to_sawtooth( 42.0f );
		thing_2.set_all_to_sawtooth( 42.0f );
		EXPECT_EQ( true, thing_1 == thing_2 );

		//---------------------------------------------------//

		// real test
		buffer<double> thing_3( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_OUTOFPLACE );
		buffer<double> thing_4( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_OUTOFPLACE );

		thing_3.set_all_to_sawtooth(1.0f);
		thing_4.set_all_to_sawtooth(1.0f);
		EXPECT_EQ( true, thing_3 == thing_4 );

		thing_4.set_one_data_point( 42.0f, 0, 0, 0, 0 );
		EXPECT_EQ( false, thing_3 == thing_4 );

		thing_3.set_all_to_sawtooth( 42.0f );
		thing_4.set_all_to_sawtooth( 42.0f );
		EXPECT_EQ( true, thing_3 == thing_4 );

		comparison_type = comparison_type_restore;
	}
	catch( const std::exception& err )
	{
		handle_exception(err);
	}
}

/*****************************************************/
/*****************************************************/
TEST_F(test_harness_tests_buffer, inequivalence_operator_returns_correct_result_with_rms) {
	try
	{
		bool comparison_type_restore = comparison_type;
		comparison_type = root_mean_square;

		size_t dimensions = 3;
		size_t lengths[3] = { 16, 32, 64 };
		size_t* stride_null = NULL;
		size_t batch = 2;
		size_t distance = 0;

		buffer<float> thing_1( dimensions, lengths, stride_null, batch, distance, layout::complex_interleaved, CLFFT_OUTOFPLACE );

		buffer<float> thing_2( dimensions, lengths, stride_null, batch, distance, layout::complex_planar, CLFFT_OUTOFPLACE );

		thing_1.set_all_to_sawtooth(1.0f);
		thing_2.set_all_to_sawtooth(1.0f);
		EXPECT_EQ( false, thing_1 != thing_2 );

		thing_2.set_one_data_point( 42.0f, 0, 0, 0, 0, 0 );
		EXPECT_EQ( true, thing_1 != thing_2 );

		thing_1.set_all_to_sawtooth( 42.0f );
		thing_2.set_all_to_sawtooth( 42.0f );
		EXPECT_EQ( false, thing_1 != thing_2 );

		comparison_type = comparison_type_restore;
	}
	catch( const std::exception& err )
	{
		handle_exception(err);
	}
}

/*****************************************************/
/*****************************************************/
TEST_F(test_harness_tests_buffer, equivalence_should_fail_given_non_matching_complexities) {
	try
	{
		size_t dimensions = 1;
		size_t lengths[3] = { 5, 1, 1 };
		size_t* stride_null = NULL;
		size_t batch = 1;
		size_t distance = 0;

		buffer<float> thing_1( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_OUTOFPLACE );
		buffer<float> thing_2( dimensions, lengths, stride_null, batch, distance, layout::complex_planar, CLFFT_OUTOFPLACE );
		EXPECT_EQ( false, thing_1 == thing_2 );

		buffer<float> thing_3( dimensions, lengths, stride_null, batch, distance, layout::complex_planar, CLFFT_OUTOFPLACE );
		buffer<float> thing_4( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_OUTOFPLACE );
		EXPECT_EQ( false, thing_3 == thing_4 );

		buffer<float> thing_5( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_OUTOFPLACE );
		buffer<float> thing_6( dimensions, lengths, stride_null, batch, distance, layout::complex_interleaved, CLFFT_OUTOFPLACE );
		EXPECT_EQ( false, thing_5 == thing_6 );

		buffer<float> thing_7( dimensions, lengths, stride_null, batch, distance, layout::complex_interleaved, CLFFT_OUTOFPLACE );
		buffer<float> thing_8( dimensions, lengths, stride_null, batch, distance, layout::real, CLFFT_OUTOFPLACE );
		EXPECT_EQ( false, thing_7 == thing_8 );
	}
	catch( const std::exception& err )
	{
		handle_exception(err);
	}
}