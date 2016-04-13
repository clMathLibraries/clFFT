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
#include <complex>
#include "clFFT.h"
#include "../client/openCL.misc.h"
#include "test_constants.h"

class clfft_UnitTest : public ::testing::Test {
protected:
	clfft_UnitTest(){}
	virtual ~clfft_UnitTest(){}
	virtual void SetUp()
	{
		lengths[ 0 ] = 32;
		lengths[ 1 ] = 32;
		lengths[ 2 ] = 32;

		commandQueueFlags = 0;

		size_t memSizeBytes = lengths[ 0 ] * lengths[ 1 ] * lengths[ 2 ] * sizeof( std::complex< float > );

		device_id = initializeCL( g_device_type, g_device_id, g_platform_id, context, printInfo );
		createOpenCLCommandQueue( context,
								  commandQueueFlags,
								  queue,
								  device_id,
								  memSizeBytes, 1, &cl_mem_input,
								  memSizeBytes, 1, &cl_mem_output
								);

		outEvent	= NULL;

		clfftCreateDefaultPlan( &test_plan, context, CLFFT_1D, lengths );
	}

	virtual void TearDown()
	{
		if( test_plan != 0 )
		{
			clfftDestroyPlan( &test_plan );
            clfftTeardown();
		}

		cleanupCL( &context, &queue, 1, &cl_mem_input, 1, &cl_mem_output, &outEvent );
	}

	clfftPlanHandle test_plan;
	size_t lengths[3];

	//	We need a valid context for clfftCreateDefaultPlan to work
	cl_context			context;
	cl_command_queue	queue;
	std::vector< cl_device_id >	device_id;
	cl_event			outEvent;
	static const bool printInfo = false;
	cl_uint commandQueueFlags;

	//	These are not used, they are only placeholders for initializeCL
	cl_mem	cl_mem_input;
	cl_mem	cl_mem_output;
};

TEST_F(clfft_UnitTest, get_plan_context_should_get_a_context) {
    cl_context the_context = NULL;

    EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanContext( test_plan, &the_context ) );
    if( the_context == NULL )
        EXPECT_EQ( "context is null :(", "context should not be null");
}

TEST_F(clfft_UnitTest, copyPlan_should_copy_plan) {
	clfftPlanHandle copied_plan;
	cl_context new_context = NULL;

	lengths[0] = 8;
	lengths[1] = 16;

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_2D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_OUTOFPLACE ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanPrecision( test_plan, CLFFT_SINGLE ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanScale( test_plan, CLFFT_FORWARD, 42.0f ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanScale( test_plan, CLFFT_BACKWARD, 0.24f ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 8*16, 8*16 ) );

	size_t clStrides[ ] = { 1, 8 };
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanInStride( test_plan, CLFFT_2D, clStrides ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanOutStride( test_plan, CLFFT_2D, clStrides ) );

	// TODO need to have created context by now; clfftPlanHandle is no longer a pointer
	clfftCopyPlan( &copied_plan, context, test_plan );

	//EXPECT_EQ( false, copied_plan.baked );
	//EXPECT_EQ( CLFFT_2D, copied_plan.dim );
	//EXPECT_EQ( CLFFT_COMPLEX_INTERLEAVED, copied_plan.inputLayout );
	//EXPECT_EQ( CLFFT_COMPLEX_INTERLEAVED, copied_plan.outputLayout );
	//EXPECT_EQ( CLFFT_OUTOFPLACE, copied_plan.placeness );
	//EXPECT_EQ( CLFFT_SINGLE, copied_plan.precision );
	////TODO check context here
	//EXPECT_FLOAT_EQ( 42.0f, copied_plan.forwardScale );
	//EXPECT_FLOAT_EQ( 0.24f, copied_plan.backwardScale );
	//EXPECT_EQ( 8*16, copied_plan.pitch );
	//EXPECT_EQ( 2, copied_plan.length.size() );
	//EXPECT_EQ( 8, copied_plan.length[0] );
	//EXPECT_EQ( 16, copied_plan.length[1] );
	//EXPECT_EQ( 2, copied_plan.inStride.size() );
	//EXPECT_EQ( 1, copied_plan.inStride[0] );
	//EXPECT_EQ( 8, copied_plan.inStride[1] );
	//EXPECT_EQ( 2, copied_plan.outStride.size() );
	//EXPECT_EQ( 1, copied_plan.outStride[0] );
	//EXPECT_EQ( 8, copied_plan.outStride[1] );

	clfftDestroyPlan( &copied_plan );
}

TEST_F(clfft_UnitTest, copyPlan_should_increase_context_reference_count) {
	//TODO me
}

//TODO need to promote some things in client.cpp to a library to write this
//TEST_F(clfft_UnitTest, getPlanContext_should_yield_appropriate_values) {
//}

TEST_F(clfft_UnitTest, getPlanBatchSize_should_yield_appropriate_values) {
	size_t batch_size;

	lengths[0] = 2;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanBatchSize( test_plan, 1 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanBatchSize( test_plan, &batch_size ) );
	EXPECT_EQ( 1, batch_size );

	lengths[0] = 4;
	lengths[1] = 2;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanBatchSize( test_plan, 8 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanBatchSize( test_plan, &batch_size ) );
	EXPECT_EQ( 8, batch_size );

	lengths[0] = 4;
	lengths[1] = 2;
	lengths[2] = 8;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanBatchSize( test_plan, 16 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanBatchSize( test_plan, &batch_size ) );
	EXPECT_EQ( 16, batch_size );
}

TEST_F(clfft_UnitTest, setPlanBatchSize_should_set_batch_size_correctly) {
	size_t batch_size;
	lengths[0] = 1;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanBatchSize( test_plan, 1 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanBatchSize( test_plan, &batch_size ) );
	EXPECT_EQ( 1, batch_size );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanBatchSize( test_plan, 2 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanBatchSize( test_plan, &batch_size ) );
	EXPECT_EQ( 2, batch_size );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanBatchSize( test_plan, 16 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanBatchSize( test_plan, &batch_size ) );
	EXPECT_EQ( 16, batch_size );
}

TEST_F(clfft_UnitTest, getPlanPrecision_should_yield_appropriate_values) {
	clfftPrecision precision;

	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanPrecision( test_plan, CLFFT_SINGLE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanPrecision( test_plan, &precision ) );
	EXPECT_EQ( CLFFT_SINGLE, precision );
}

TEST_F(clfft_UnitTest, setPlanPrecision_should_set_precision_to_supported_values) {

	clfftPrecision precision;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanPrecision( test_plan, CLFFT_SINGLE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanPrecision( test_plan, &precision ) );
	EXPECT_EQ( CLFFT_SINGLE, precision );
}

TEST_F(clfft_UnitTest, setPlanPrecision_should_fail_to_set_unsupported_values) {
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetPlanPrecision( test_plan, CLFFT_SINGLE_FAST ) );
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetPlanPrecision( test_plan, CLFFT_DOUBLE_FAST ) );
}

TEST_F(clfft_UnitTest, getPlanScale_should_yield_appropriate_values) {
	cl_float scale;

	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanScale( test_plan, CLFFT_FORWARD, 1.414f ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanScale( test_plan, CLFFT_BACKWARD, 2.718f ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanScale( test_plan, CLFFT_FORWARD, &scale ) );
	EXPECT_FLOAT_EQ( 1.414f, scale );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanScale( test_plan, CLFFT_BACKWARD, &scale ) );
	EXPECT_FLOAT_EQ( 2.718f, scale );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanScale( test_plan, CLFFT_MINUS, &scale ) );
	EXPECT_FLOAT_EQ( 1.414f, scale );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanScale( test_plan, CLFFT_PLUS, &scale ) );
	EXPECT_FLOAT_EQ( 2.718f, scale );
}

TEST_F(clfft_UnitTest, getPlanScale_should_fail_on_invalid_direction) {
	cl_float scale;

	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftGetPlanScale( test_plan, ENDDIRECTION, &scale ) );
}

TEST_F(clfft_UnitTest, setPlanScale_should_set_scale_correctly) {
	cl_float scale;

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanScale( test_plan, CLFFT_FORWARD, 1.57f ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanScale( test_plan, CLFFT_FORWARD, &scale ) );
	EXPECT_FLOAT_EQ( 1.57f, scale );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanScale( test_plan, CLFFT_BACKWARD, 3.14f ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanScale( test_plan, CLFFT_BACKWARD, &scale ) );
	EXPECT_FLOAT_EQ( 3.14f, scale );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanScale( test_plan, CLFFT_MINUS, 4.71f ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanScale( test_plan, CLFFT_MINUS, &scale ) );
	EXPECT_FLOAT_EQ( 4.71f, scale );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanScale( test_plan, CLFFT_PLUS, 6.28f ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanScale( test_plan, CLFFT_PLUS, &scale ) );
	EXPECT_FLOAT_EQ( 6.28f, scale );
}

TEST_F(clfft_UnitTest, setPlanScale_should_fail_on_invalid_direction) {

	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanScale( test_plan, ENDDIRECTION, 42.0 ) );
}

TEST_F(clfft_UnitTest, setPlanDimLength_should_set_dimensions_to_supported_values) {
	cl_uint	lengthSize	= 0;
	clfftDim	dim;
	size_t	testLengths[ 3 ];

	lengths[0] = 1;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_1D ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDim( test_plan, &dim, &lengthSize ) );
	EXPECT_EQ( 1, lengthSize );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_1D, testLengths ) );
	EXPECT_EQ( CLFFT_1D, dim );
	EXPECT_EQ( 1, testLengths[0] );


	lengths[0] = 2;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_1D, testLengths ) );
	EXPECT_EQ( 2, testLengths[0] );

	lengths[0] = 4;
	lengths[1] = 8;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_2D ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDim( test_plan, &dim, &lengthSize ) );
	EXPECT_EQ( CLFFT_2D, dim );
	EXPECT_EQ( 2, lengthSize );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_2D, testLengths ) );
	EXPECT_EQ( 4, testLengths[0] );
	EXPECT_EQ( 8, testLengths[1] );

	lengths[0] = 32;
	lengths[1] = 64;
	lengths[2] = 128;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_3D ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDim( test_plan, &dim, &lengthSize ) );
	EXPECT_EQ( CLFFT_3D, dim );
	EXPECT_EQ( 3, lengthSize );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_3D, testLengths ) );
	EXPECT_EQ( 32, testLengths[0] );
	EXPECT_EQ( 64, testLengths[1] );
	EXPECT_EQ( 128, testLengths[2] );

	lengths[0] = 2;
	lengths[1] = 3;
	lengths[2] = 5;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_3D ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDim( test_plan, &dim, &lengthSize ) );
	EXPECT_EQ( CLFFT_3D, dim );
	EXPECT_EQ( 3, lengthSize );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_3D, testLengths ) );
	EXPECT_EQ( 2, testLengths[0] );
	EXPECT_EQ( 3, testLengths[1] );
	EXPECT_EQ( 5, testLengths[2] );

	lengths[0] = 4;
	lengths[1] = 9;
	lengths[2] = 25;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_3D ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDim( test_plan, &dim, &lengthSize ) );
	EXPECT_EQ( CLFFT_3D, dim );
	EXPECT_EQ( 3, lengthSize );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_3D, testLengths ) );
	EXPECT_EQ( 4, testLengths[0] );
	EXPECT_EQ( 9, testLengths[1] );
	EXPECT_EQ( 25, testLengths[2] );

	lengths[0] = 10;
	lengths[1] = 144;
	lengths[2] = 2700;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_3D ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDim( test_plan, &dim, &lengthSize ) );
	EXPECT_EQ( CLFFT_3D, dim );
	EXPECT_EQ( 3, lengthSize );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_3D, testLengths ) );
	EXPECT_EQ( 10, testLengths[0] );
	EXPECT_EQ( 144, testLengths[1] );
	EXPECT_EQ( 2700, testLengths[2] );
}

TEST_F(clfft_UnitTest, setPlanDimLength_should_fail_if_a_length_is_set_to_zero) {
	lengths[0] = 0;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );

	lengths[0] = 4;
	lengths[1] = 0;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_2D ) );
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );

	lengths[0] = 0;
	lengths[1] = 4;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );

	lengths[0] = 0;
	lengths[1] = 0;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );

	lengths[0] = 0;
	lengths[1] = 4;
	lengths[2] = 4;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_3D ) );
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );

	lengths[0] = 4;
	lengths[1] = 0;
	lengths[2] = 4;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );

	lengths[0] = 4;
	lengths[1] = 4;
	lengths[2] = 0;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );

	lengths[0] = 0;
	lengths[1] = 0;
	lengths[2] = 0;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
}

TEST_F(clfft_UnitTest, setPlanDimLength_should_fail_on_radices_that_have_non_supported_factors) {
	// currently only factors of 2, 3, 5, 7, 11, and 13 are supported
	lengths[0] = 2*3*5*7*11*13*19;
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );

	lengths[0] = 2*2*3*3*5*5*5*5*13;
	lengths[1] = 17;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_2D ) );
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );

	lengths[0] = 5*23;
	lengths[1] = 2*2*3;
	lengths[2] = 5*3*2*2*2*2*2*2*2*7*29;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_3D ) );
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
}

TEST_F(clfft_UnitTest, setPlanDimLength_should_set_values_to_second_set_when_called_twice) {
	cl_uint	lengthSize	= 0;
	clfftDim	dim;
	size_t	testLengths[ 1 ];

	lengths[0] = 2;
	lengths[1] = 4;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_2D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_1D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDim( test_plan, &dim, &lengthSize ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_1D, testLengths ) );
	EXPECT_EQ( CLFFT_1D, dim );
	EXPECT_EQ( 1, lengthSize );
	EXPECT_EQ( 2, testLengths[ 0 ] );
}

TEST_F(clfft_UnitTest, getPlanDimLength_should_yield_correct_values) {
	clfftDim dim;
	cl_uint	lengthSize	= 0;
	size_t gotten_lengths[3];

	lengths[0] = 1;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_1D, gotten_lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDim( test_plan, &dim, &lengthSize ) );
	EXPECT_EQ( CLFFT_1D, dim );
	EXPECT_EQ( 1, gotten_lengths[0] );

	lengths[0] = 2;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_1D, gotten_lengths ) );
	EXPECT_EQ( 2, gotten_lengths[0] );

	lengths[0] = 1;
	lengths[1] = 1;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_2D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_2D, gotten_lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDim( test_plan, &dim, &lengthSize ) );
	EXPECT_EQ( CLFFT_2D, dim );
	EXPECT_EQ( 1, gotten_lengths[0] );
	EXPECT_EQ( 1, gotten_lengths[1] );

	lengths[0] = 2;
	lengths[1] = 4;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_2D, gotten_lengths ) );
	EXPECT_EQ( 2, gotten_lengths[0] );
	EXPECT_EQ( 4, gotten_lengths[1] );

	lengths[0] = 1;
	lengths[1] = 1;
	lengths[2] = 1;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_3D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_3D, gotten_lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDim( test_plan, &dim, &lengthSize ) );
	EXPECT_EQ( CLFFT_3D, dim );
	EXPECT_EQ( 1, gotten_lengths[0] );
	EXPECT_EQ( 1, gotten_lengths[1] );
	EXPECT_EQ( 1, gotten_lengths[2] );

	lengths[0] = 2;
	lengths[1] = 4;
	lengths[2] = 8;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanLength( test_plan, CLFFT_3D, gotten_lengths ) );
	EXPECT_EQ( CLFFT_3D, dim );
	EXPECT_EQ( 2, gotten_lengths[0] );
	EXPECT_EQ( 4, gotten_lengths[1] );
	EXPECT_EQ( 8, gotten_lengths[2] );
}

TEST_F(clfft_UnitTest, getPlanLength_should_fail_when_passed_null_pointer) {
	EXPECT_EQ( CLFFT_INVALID_HOST_PTR, clfftGetPlanLength( test_plan, CLFFT_1D, NULL ) );
}

TEST_F(clfft_UnitTest, getPlanInStride_should_fail_when_passed_null_pointer) {
	EXPECT_EQ( CLFFT_INVALID_HOST_PTR, clfftGetPlanInStride( test_plan, CLFFT_1D, NULL ) );
}

TEST_F(clfft_UnitTest, getPlanOutStride_should_fail_when_passed_null_pointer) {
	EXPECT_EQ( CLFFT_INVALID_HOST_PTR, clfftGetPlanOutStride( test_plan, CLFFT_1D, NULL ) );
}

TEST_F(clfft_UnitTest, createDefaultPlan_should_fail_when_passed_null_pointer) {
	EXPECT_EQ( CLFFT_INVALID_HOST_PTR, clfftCreateDefaultPlan( &test_plan, context, CLFFT_1D, NULL));
}

TEST_F(clfft_UnitTest, createDefaultPlan_should_fail_when_passed_length_of_0) {
    size_t length[3] = {1,1,1};

    length[0] = 0;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftCreateDefaultPlan( &test_plan, context, CLFFT_1D, length));

    length[0] = 1;
    length[1] = 0;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftCreateDefaultPlan( &test_plan, context, CLFFT_2D, length));
    length[0] = 0;
    length[1] = 1;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftCreateDefaultPlan( &test_plan, context, CLFFT_2D, length));
    length[0] = 0;
    length[1] = 0;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftCreateDefaultPlan( &test_plan, context, CLFFT_2D, length));

    length[0] = 1;
    length[1] = 1;
    length[2] = 0;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftCreateDefaultPlan( &test_plan, context, CLFFT_3D, length));
    length[0] = 1;
    length[1] = 0;
    length[2] = 1;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftCreateDefaultPlan( &test_plan, context, CLFFT_3D, length));
    length[0] = 0;
    length[1] = 1;
    length[2] = 1;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftCreateDefaultPlan( &test_plan, context, CLFFT_3D, length));
    length[0] = 0;
    length[1] = 0;
    length[2] = 0;
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftCreateDefaultPlan( &test_plan, context, CLFFT_3D, length));
}

TEST_F(clfft_UnitTest, createDefaultPlan_should_fail_when_passed_invalid_dimension) {
    size_t length[3] = {1,1,1};

	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftCreateDefaultPlan( &test_plan, context, ENDDIMENSION, length));
}

TEST_F(clfft_UnitTest, createDefaultPlan_should_fail_when_passed_unsupported_length) {
    size_t length[3] = {1,1,1};

    length[0] = 17;
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftCreateDefaultPlan( &test_plan, context, CLFFT_1D, length));

    length[0] = 23;
    length[1] = 1;
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftCreateDefaultPlan( &test_plan, context, CLFFT_2D, length));
    length[0] = 1;
    length[1] = 34;
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftCreateDefaultPlan( &test_plan, context, CLFFT_2D, length));
    length[0] = 19;
    length[1] = 22;
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftCreateDefaultPlan( &test_plan, context, CLFFT_2D, length));

    length[0] = 1;
    length[1] = 97;
    length[2] = 221;
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftCreateDefaultPlan( &test_plan, context, CLFFT_3D, length));
    length[0] = 1;
    length[1] = 17;
    length[2] = 1;
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftCreateDefaultPlan( &test_plan, context, CLFFT_3D, length));
    length[0] = 87;
    length[1] = 1;
    length[2] = 1;
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftCreateDefaultPlan( &test_plan, context, CLFFT_3D, length));
    length[0] = 5;
    length[1] = 6;
    length[2] = 17;
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftCreateDefaultPlan( &test_plan, context, CLFFT_3D, length));
}

TEST_F(clfft_UnitTest, setPlanInStride_should_set_input_strides_to_supported_values) {
	size_t strides[ ] = { 1, 16, 16*32 };
	size_t gotten_strides[3];

	lengths[0] = 16;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanInStride( test_plan, CLFFT_1D, strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanInStride( test_plan, CLFFT_1D, gotten_strides ) );
	EXPECT_EQ( 1, gotten_strides[0] );

	lengths[0] = 16;
	lengths[1] = 32;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_2D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanInStride( test_plan, CLFFT_2D, strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanInStride( test_plan, CLFFT_2D, gotten_strides ) );
	EXPECT_EQ( 1, gotten_strides[0] );
	EXPECT_EQ( 16, gotten_strides[1] );

	lengths[0] = 16;
	lengths[1] = 32;
	lengths[2] = 64;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_3D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanInStride( test_plan, CLFFT_3D, strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanInStride( test_plan, CLFFT_3D, gotten_strides ) );
	EXPECT_EQ( 1, gotten_strides[0] );
	EXPECT_EQ( 16, gotten_strides[1] );
	EXPECT_EQ( 16*32, gotten_strides[2] );
}

TEST_F(clfft_UnitTest, getPlanInStride_should_yield_correct_values) {
	size_t input_strides[ ] = { 1, 8, 8*16 };
	size_t output_strides[ ] = { 3, 99, 456789 };
	size_t gotten_strides[ 3 ];

	size_t x;

	lengths[0] = 8;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanInStride( test_plan, CLFFT_1D, input_strides ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanOutStride( test_plan, CLFFT_1D, output_strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanInStride( test_plan, CLFFT_1D, &x ) );
	EXPECT_EQ( 1, x );

	lengths[0] = 8;
	lengths[1] = 16;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_2D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanInStride( test_plan, CLFFT_2D, input_strides) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanOutStride( test_plan, CLFFT_2D, output_strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanInStride( test_plan, CLFFT_2D, gotten_strides ) );
	EXPECT_EQ( 1, gotten_strides[ 0 ] );
	EXPECT_EQ( 8, gotten_strides[ 1 ] );

	lengths[0] = 8;
	lengths[1] = 16;
	lengths[2] = 32;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_3D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanInStride( test_plan, CLFFT_3D, input_strides) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanOutStride( test_plan, CLFFT_3D, output_strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanInStride( test_plan, CLFFT_3D, gotten_strides ) );
	EXPECT_EQ( 1, gotten_strides[ 0 ] );
	EXPECT_EQ( 8, gotten_strides[ 1 ] );
	EXPECT_EQ( 8*16, gotten_strides[ 2 ] );
}

TEST_F(clfft_UnitTest, setPlanOutStride_should_set_output_strides_to_supported_values) {
	size_t strides[ ] = { 1, 16, 16*32 };
	size_t gotten_strides[3];

	lengths[0] = 16;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanOutStride( test_plan, CLFFT_1D, strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanOutStride( test_plan, CLFFT_1D, gotten_strides ) );
	EXPECT_EQ( 1, gotten_strides[0] );

	lengths[0] = 16;
	lengths[1] = 32;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_2D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanOutStride( test_plan, CLFFT_2D, strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanOutStride( test_plan, CLFFT_2D, gotten_strides ) );
	EXPECT_EQ( 1, gotten_strides[0] );
	EXPECT_EQ( 16, gotten_strides[1] );

	lengths[0] = 16;
	lengths[1] = 32;
	lengths[2] = 64;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_3D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanOutStride( test_plan, CLFFT_3D, strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanOutStride( test_plan, CLFFT_3D, gotten_strides ) );
	EXPECT_EQ( 1, gotten_strides[0] );
	EXPECT_EQ( 16, gotten_strides[1] );
	EXPECT_EQ( 16*32, gotten_strides[2] );
}

TEST_F(clfft_UnitTest, getPlanOutStride_should_yield_correct_values) {
	size_t input_strides[ ] = { 3, 99, 456789 };
	size_t output_strides[ ] = { 1, 8, 8*16 };
	size_t gotten_strides[ 3 ];

	size_t x;

	lengths[0] = 8;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanInStride( test_plan, CLFFT_1D, input_strides ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanOutStride( test_plan, CLFFT_1D, output_strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanOutStride( test_plan, CLFFT_1D, &x ) );
	EXPECT_EQ( 1, x );

	lengths[0] = 8;
	lengths[1] = 16;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_2D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanInStride( test_plan, CLFFT_2D, input_strides) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanOutStride( test_plan, CLFFT_2D, output_strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanOutStride( test_plan, CLFFT_2D, gotten_strides ) );
	EXPECT_EQ( 1, gotten_strides[ 0 ] );
	EXPECT_EQ( 8, gotten_strides[ 1 ] );

	lengths[0] = 8;
	lengths[1] = 16;
	lengths[2] = 32;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_3D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_3D, lengths ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanInStride( test_plan, CLFFT_3D, input_strides) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanOutStride( test_plan, CLFFT_3D, output_strides ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanOutStride( test_plan, CLFFT_3D, gotten_strides ) );
	EXPECT_EQ( 1, gotten_strides[ 0 ] );
	EXPECT_EQ( 8, gotten_strides[ 1 ] );
	EXPECT_EQ( 8*16, gotten_strides[ 2 ] );
}

TEST_F(clfft_UnitTest, setPlanDistance_should_set_distance_to_supported_values) {
	size_t inDistance, outDistance;
	lengths[0] = 8;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 8+10, 8+2 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDistance( test_plan, &inDistance, &outDistance ) );
	EXPECT_EQ( 18, inDistance );
	EXPECT_EQ( 10, outDistance );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 8+3, 8+11 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDistance( test_plan, &inDistance, &outDistance ) );
	EXPECT_EQ( 11, inDistance );
	EXPECT_EQ( 19, outDistance );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 8, 8 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDistance( test_plan, &inDistance, &outDistance ) );
	EXPECT_EQ( 8, inDistance );
	EXPECT_EQ( 8, outDistance );

	lengths[0] = 2;
	lengths[1] = 2;
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDim( test_plan, CLFFT_2D ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 2*2+10, 2*2+2 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDistance( test_plan, &inDistance, &outDistance ) );
	EXPECT_EQ( 14, inDistance );
	EXPECT_EQ( 6, outDistance );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 2*2+3, 2*2+11 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDistance( test_plan, &inDistance, &outDistance ) );
	EXPECT_EQ( 7, inDistance );
	EXPECT_EQ( 15, outDistance );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 2*2, 2*2 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDistance( test_plan, &inDistance, &outDistance ) );
	EXPECT_EQ( 4, inDistance );
	EXPECT_EQ( 4, outDistance );
}

TEST_F(clfft_UnitTest, setPlanDistance_should_fail_to_set_pitch_to_smaller_than_one_dataset) {
	lengths[0] = 32;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 32-30, 32-30 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 32-16, 32-16 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 32-1, 32-1 ) );

	lengths[0] = 32;
	lengths[1] = 32;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_1D, lengths ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 32*32-30, 32*32-30 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 32*32-16, 32*32-16 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 32*32-1, 32*32-1 ) );
}

TEST_F(clfft_UnitTest, getPlanDistance_should_yield_correct_values) {
	size_t inDistance;
	size_t outDistance;

	lengths[0] = 2;
	lengths[1] = 2;
	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );

	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( test_plan, 4, 6 ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanDistance( test_plan, &inDistance, &outDistance ) );
	EXPECT_EQ( 4, inDistance );
	EXPECT_EQ( 6, outDistance );
}

TEST_F(clfft_UnitTest, setLayout_and_getLayout_should_set_and_get_layouts_respectively) {
	clfftLayout	in, out;

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetLayout( test_plan, &in, &out ) );
	EXPECT_EQ( CLFFT_COMPLEX_INTERLEAVED, in );
	EXPECT_EQ( CLFFT_COMPLEX_INTERLEAVED, out );

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetLayout( test_plan, &in, &out ) );
	EXPECT_EQ( CLFFT_COMPLEX_PLANAR, in );
	EXPECT_EQ( CLFFT_COMPLEX_PLANAR, out );

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_PLANAR ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetLayout( test_plan, &in, &out ) );
	EXPECT_EQ( CLFFT_COMPLEX_INTERLEAVED, in );
	EXPECT_EQ( CLFFT_COMPLEX_PLANAR, out );

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_INTERLEAVED ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetLayout( test_plan, &in, &out ) );
	EXPECT_EQ( CLFFT_COMPLEX_PLANAR, in );
	EXPECT_EQ( CLFFT_COMPLEX_INTERLEAVED, out );

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetLayout( test_plan, &in, &out ) );
	EXPECT_EQ( CLFFT_REAL, in );
	EXPECT_EQ( CLFFT_HERMITIAN_INTERLEAVED, out );

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_REAL, CLFFT_HERMITIAN_PLANAR ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetLayout( test_plan, &in, &out ) );
	EXPECT_EQ( CLFFT_REAL, in );
	EXPECT_EQ( CLFFT_HERMITIAN_PLANAR, out );

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_HERMITIAN_PLANAR, CLFFT_REAL ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetLayout( test_plan, &in, &out ) );
	EXPECT_EQ( CLFFT_HERMITIAN_PLANAR, in );
	EXPECT_EQ( CLFFT_REAL, out );

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetLayout( test_plan, &in, &out ) );
	EXPECT_EQ( CLFFT_HERMITIAN_INTERLEAVED, in );
	EXPECT_EQ( CLFFT_REAL, out );
}

TEST_F(clfft_UnitTest, setLayout_should_fail_on_invalid_argument_values) {
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetLayout( test_plan, CLFFT_COMPLEX_INTERLEAVED, ENDLAYOUT ) );
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetLayout( test_plan, ENDLAYOUT, CLFFT_COMPLEX_INTERLEAVED ) );
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetLayout( test_plan, ENDLAYOUT, ENDLAYOUT ) );

	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetLayout( test_plan, static_cast<clfftLayout>(ENDLAYOUT+42), CLFFT_COMPLEX_INTERLEAVED ) );
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetLayout( test_plan, CLFFT_COMPLEX_INTERLEAVED, static_cast<clfftLayout>(ENDLAYOUT+42) ) );
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetLayout( test_plan, static_cast<clfftLayout>(ENDLAYOUT+42), static_cast<clfftLayout>(ENDLAYOUT+42)) );
}

TEST_F(clfft_UnitTest, valid_layout_combinations_for_real_to_complex_should_succeed_on_bake_1) {
	// in place can go from real <-> hermitian interleaved

	clfftCreateDefaultPlan( &test_plan, context, CLFFT_1D, lengths );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_INPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftBakePlan(test_plan, 1, &queue, NULL, NULL ));
}

TEST_F(clfft_UnitTest, valid_layout_combinations_for_real_to_complex_should_succeed_on_bake_2) {
	clfftCreateDefaultPlan( &test_plan, context, CLFFT_1D, lengths );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_INPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftBakePlan(test_plan, 1, &queue, NULL, NULL ));
}

TEST_F(clfft_UnitTest, valid_layout_combinations_for_real_to_complex_should_succeed_on_bake_3) {
	// out of place can go from real <-> hermitian interleaved or planar
	clfftCreateDefaultPlan( &test_plan, context, CLFFT_1D, lengths );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_OUTOFPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftBakePlan(test_plan, 1, &queue, NULL, NULL ));
}

TEST_F(clfft_UnitTest, valid_layout_combinations_for_real_to_complex_should_succeed_on_bake_4) {
	clfftCreateDefaultPlan( &test_plan, context, CLFFT_1D, lengths );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_OUTOFPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_REAL, CLFFT_HERMITIAN_PLANAR ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftBakePlan(test_plan, 1, &queue, NULL, NULL ));
}

TEST_F(clfft_UnitTest, valid_layout_combinations_for_real_to_complex_should_succeed_on_bake_5) {
	clfftCreateDefaultPlan( &test_plan, context, CLFFT_1D, lengths );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_OUTOFPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftBakePlan(test_plan, 1, &queue, NULL, NULL ));
}

TEST_F(clfft_UnitTest, valid_layout_combinations_for_real_to_complex_should_succeed_on_bake_6) {
	clfftCreateDefaultPlan( &test_plan, context, CLFFT_1D, lengths );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_OUTOFPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_HERMITIAN_PLANAR, CLFFT_REAL ) );
	ASSERT_EQ( CLFFT_SUCCESS, clfftBakePlan(test_plan, 1, &queue, NULL, NULL ));
}

TEST_F(clfft_UnitTest, invalid_layout_combinations_for_real_to_complex_should_fail) {
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetLayout( test_plan, CLFFT_REAL, CLFFT_REAL ) );
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetLayout( test_plan, CLFFT_REAL, CLFFT_COMPLEX_INTERLEAVED ) );
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetLayout( test_plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_REAL ) );
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetLayout( test_plan, CLFFT_REAL, CLFFT_COMPLEX_PLANAR ) );
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetLayout( test_plan, CLFFT_COMPLEX_PLANAR, CLFFT_REAL ) );
}

TEST_F(clfft_UnitTest, real_to_planar_should_fail_on_bake) {
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_INPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_REAL, CLFFT_HERMITIAN_PLANAR ) );
	EXPECT_EQ( CLFFT_INVALID_PLAN, clfftBakePlan(test_plan, 1, &queue, NULL, NULL ));
}

TEST_F(clfft_UnitTest, planar_to_real_should_fail_on_bake) {
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_INPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( test_plan, CLFFT_HERMITIAN_PLANAR, CLFFT_REAL ) );
	EXPECT_EQ( CLFFT_INVALID_PLAN, clfftBakePlan(test_plan, 1, &queue, NULL, NULL ));
}

TEST_F(clfft_UnitTest, setResultLocation_should_set_placeness_to_supported_values) {
	clfftResultLocation	place;

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_INPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetResultLocation( test_plan, &place ) );
	EXPECT_EQ( CLFFT_INPLACE, place );

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_OUTOFPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetResultLocation( test_plan, &place ) );
	EXPECT_EQ( CLFFT_OUTOFPLACE, place );
}

TEST_F(clfft_UnitTest, setResultLocation_should_fail_on_invalid_argument_values) {
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetResultLocation( test_plan, ENDPLACE ) );
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetResultLocation( test_plan, static_cast<clfftResultLocation>(ENDPLACE+42) ) );
}

TEST_F(clfft_UnitTest, getResultLocation_should_yield_correct_values) {
	clfftResultLocation placeness;

	ASSERT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_INPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetResultLocation( test_plan, &placeness ) );
	EXPECT_EQ( CLFFT_INPLACE, placeness );

	ASSERT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( test_plan, CLFFT_OUTOFPLACE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetResultLocation( test_plan, &placeness ) );
	EXPECT_EQ( CLFFT_OUTOFPLACE, placeness );
}

TEST_F(clfft_UnitTest, SetPlanTransposeResult_should_set_resulttransposed_to_supported_values) {
	clfftResultTransposed transposed;

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanTransposeResult( test_plan, CLFFT_TRANSPOSED ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanTransposeResult( test_plan, &transposed ) );
	EXPECT_EQ( CLFFT_TRANSPOSED, transposed );

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanTransposeResult( test_plan, CLFFT_NOTRANSPOSE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanTransposeResult( test_plan, &transposed ) );
	EXPECT_EQ( CLFFT_NOTRANSPOSE, transposed );
}

TEST_F(clfft_UnitTest, setPlanTransposeResult_should_fail_on_invalid_argument_values) {
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanTransposeResult( test_plan, ENDTRANSPOSED ) );
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanTransposeResult( test_plan, static_cast<clfftResultTransposed>(ENDTRANSPOSED+42) ) );
}

TEST_F(clfft_UnitTest, getPlanTransposeResult_should_yield_correct_values) {
	clfftResultTransposed transposed;

	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanTransposeResult( test_plan, CLFFT_TRANSPOSED ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanTransposeResult( test_plan, &transposed ) );
	EXPECT_EQ( CLFFT_TRANSPOSED, transposed );

	ASSERT_EQ( CLFFT_SUCCESS, clfftSetPlanTransposeResult( test_plan, CLFFT_NOTRANSPOSE ) );
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanTransposeResult( test_plan, &transposed ) );
	EXPECT_EQ( CLFFT_NOTRANSPOSE, transposed );
}

TEST_F(clfft_UnitTest, getTmpBufSize_should_fail_on_unbaked_plan) {
	size_t buffersize;
	EXPECT_EQ( CLFFT_INVALID_OPERATION, clfftGetTmpBufSize( test_plan, &buffersize ) );
}

TEST_F(clfft_UnitTest, getTmpBufSize_should_succeed_on_baked_plan) {
	size_t buffersize;
	ASSERT_EQ( CLFFT_SUCCESS, clfftBakePlan(test_plan, 1, &queue, NULL, NULL ));
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetTmpBufSize( test_plan, &buffersize ) );
}

TEST_F(clfft_UnitTest, bake_plan_should_fail_to_bake_multi_GPU_plan) {
	cl_uint number_of_gpus = 2;
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftBakePlan(test_plan, number_of_gpus, &queue, NULL, NULL ));
}

void CL_CALLBACK foofies( clfftPlanHandle spoon, void* fork){}

TEST_F(clfft_UnitTest, bake_plan_should_fail_on_non_null_function_pointer) {
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftBakePlan(test_plan, 1, &queue, &foofies, NULL ));
}

TEST_F(clfft_UnitTest, bake_plan_should_fail_on_non_null_user_data) {
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftBakePlan(test_plan, 1, &queue, NULL, (void*)0xf00f1e5 ));
}

TEST_F(clfft_UnitTest, set_dimension_should_fail_on_invalid_value) {
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetPlanDim( test_plan, ENDDIMENSION ) );
}

TEST_F(clfft_UnitTest, set_precision_should_fail_on_invalid_value) {
	EXPECT_EQ( CLFFT_INVALID_ARG_VALUE, clfftSetPlanPrecision( test_plan, ENDPRECISION ) );
}

TEST_F(clfft_UnitTest, set_length_should_fail_on_null_pointer) {
    size_t* lengths = NULL;
	EXPECT_EQ( CLFFT_INVALID_HOST_PTR, clfftSetPlanLength( test_plan, CLFFT_2D, lengths ) );
}

TEST_F(clfft_UnitTest, set_length_should_fail_on_invalid_dimension) {
    size_t lengths[3] = {2,4,8};
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetPlanLength( test_plan, ENDDIMENSION, lengths ) );
}

TEST_F(clfft_UnitTest, get_length_should_fail_on_invalid_dimension) {
    size_t lengths[3] = {2,4,8};
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftGetPlanLength( test_plan, ENDDIMENSION, lengths ) );
}

TEST_F(clfft_UnitTest, set_in_stride_should_fail_on_null_pointer) {
    size_t* strides = NULL;
	EXPECT_EQ( CLFFT_INVALID_HOST_PTR, clfftSetPlanInStride( test_plan, CLFFT_2D, strides ) );
}

TEST_F(clfft_UnitTest, set_in_stride_should_fail_on_invalid_dimension) {
    size_t strides[3] = {2,4,8};
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetPlanInStride( test_plan, ENDDIMENSION, strides ) );
}

TEST_F(clfft_UnitTest, get_in_stride_should_fail_on_invalid_dimension) {
    size_t strides[3] = {2,4,8};
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftGetPlanInStride( test_plan, ENDDIMENSION, strides ) );
}

TEST_F(clfft_UnitTest, set_out_stride_should_fail_on_null_pointer) {
    size_t* strides = NULL;
	EXPECT_EQ( CLFFT_INVALID_HOST_PTR, clfftSetPlanOutStride( test_plan, CLFFT_2D, strides ) );
}

TEST_F(clfft_UnitTest, set_out_stride_should_fail_on_invalid_dimension) {
    size_t strides[3] = {2,4,8};
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftSetPlanOutStride( test_plan, ENDDIMENSION, strides ) );
}

TEST_F(clfft_UnitTest, get_out_stride_should_fail_on_invalid_dimension) {
    size_t strides[3] = {2,4,8};
	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftGetPlanOutStride( test_plan, ENDDIMENSION, strides ) );
}

TEST_F(clfft_UnitTest, enqueue_transform_should_fail_with_num_queues_and_events_greater_than_1) {
    cl_mem* cl_mem_input = NULL;
    cl_mem* cl_mem_output = NULL;

	EXPECT_EQ( CLFFT_NOTIMPLEMENTED, clfftEnqueueTransform(
		test_plan, CLFFT_FORWARD, 2, &queue, 0, NULL, &outEvent, cl_mem_input, cl_mem_output, NULL ));
}

TEST_F(clfft_UnitTest, get_version_should_get_a_version_number) {
    cl_uint major = 0xb00f1e5;
    cl_uint minor = 0xd00f1e5;
    cl_uint patch = 0xf00f1e5;
	EXPECT_EQ( CLFFT_SUCCESS, clfftGetVersion( &major, &minor, &patch ) );
    EXPECT_NE( 0xb00f1e5, major);
    EXPECT_NE( 0xd00f1e5, minor);
    EXPECT_NE( 0xf00f1e5, patch);
}

TEST_F(clfft_UnitTest, setup_should_succeed_given_a_setup_data_pointer) {
    clfftSetupData data;
    data.major = 6;
    data.minor = 8;
    data.patch = 42;
    data.debugFlags = 0xf1a95;

	EXPECT_EQ( CLFFT_SUCCESS, clfftSetup( &data ) );
}
