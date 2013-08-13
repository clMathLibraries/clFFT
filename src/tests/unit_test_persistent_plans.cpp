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

//TODO persistent plan feature : caching kernel binaries for later reload
#if defined(_WIN32)

#include <gtest/gtest.h>
#include <complex>
#include "clFFT.h"
#include "../client/openCL.misc.h"
#include "clfft.typedefs.h"

class clfft_PersistentPlans : public ::testing::Test {
protected:
	clfft_PersistentPlans(){}
	virtual ~clfft_PersistentPlans(){}
	virtual void SetUp(){}
	virtual void TearDown(){}
};

void test_persistent_plan( size_t * length, clfftLayout input_layout, clfftLayout output_layout, clfftResultLocation result_location )
{
	try
	{
		clfftDim cl_dimension;
		if( length[dimy] == 1 && length[dimz] == 1 )
			cl_dimension = CLFFT_1D;
		else if( length[dimz] == 1 )
			cl_dimension = CLFFT_2D;
		else
			cl_dimension = CLFFT_3D;

		fftw_dim fftw_dimension = static_cast<fftw_dim>(cl_dimension);

		{
			clfft_single write_fft( cl_dimension, length, NULL, 1, 0);
			write_fft.input.set_all_data_points_on_all_passes_to_value(1.0f,0.0f);
			write_fft.set_forward_transform();

			if( result_location == CLFFT_INPLACE )
				write_fft.set_in_place();
			else
				write_fft.set_out_of_place();

			write_fft.input_layout(input_layout);
			write_fft.output_layout(output_layout);
			write_fft.write_plan_to_file("wakkawakka.fft");
		}

		clfft_single read_fft( cl_dimension, length, NULL, 1, 0);
		read_fft.input.set_all_data_points_on_all_passes_to_value(1.0f,0.0f);
		read_fft.set_forward_transform();
		read_fft.read_plan_from_file("wakkawakka.fft");
		read_fft.transform();

		fftw_single reference( fftw_dimension, length);
		reference.set_forward_transform();
		reference.data.set_all_data_points_on_all_passes_to_value(1.0f,0.0f);
		reference.transform();

		if( read_fft.placeness() == CLFFT_INPLACE )
			EXPECT_EQ( true, read_fft.input.is_equal_to( reference.data));
		else
			EXPECT_EQ( true, read_fft.output.is_equal_to( reference.data));
	}
	catch( const std::exception& err )
	{
		handle_exception(err);
	}
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_1d_inplace_interleaved)
{
	size_t length[3] = {1024,1,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_1d_inplace_interleaved)
{
	size_t length[3] = {32768,1,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_normal_2d_inplace_interleaved)
{
	size_t length[3] = {1024,1024,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_normal_2d_inplace_interleaved)
{
	size_t length[3] = {4096,2,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_large_2d_inplace_interleaved)
{
	size_t length[3] = {4096,4096,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_large_2d_inplace_interleaved)
{
	size_t length[3] = {2,4096,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_normal_normal_3d_inplace_interleaved)
{
	size_t length[3] = {32,32,32};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_mixed_radices_3d_inplace_interleaved)
{
	size_t length[3] = {2*3*5,2*3*5,2*3*5};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_1d_inplace_planar)
{
	size_t length[3] = {1024,1,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_1d_inplace_planar)
{
	size_t length[3] = {32768,1,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_normal_2d_inplace_planar)
{
	size_t length[3] = {1024,1024,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_normal_2d_inplace_planar)
{
	size_t length[3] = {4096,2,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_large_2d_inplace_planar)
{
	size_t length[3] = {4096,4096,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_large_2d_inplace_planar)
{
	size_t length[3] = {2,4096,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_normal_normal_3d_inplace_planar)
{
	size_t length[3] = {32,32,32};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_mixed_radices_3d_inplace_planar)
{
	size_t length[3] = {2*3*5,2*3*5,2*3*5};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_INPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_1d_outofplace_interleaved)
{
	size_t length[3] = {1024,1,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_1d_outofplace_interleaved)
{
	size_t length[3] = {32768,1,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_normal_2d_outofplace_interleaved)
{
	size_t length[3] = {1024,1024,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_normal_2d_outofplace_interleaved)
{
	size_t length[3] = {4096,2,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_large_2d_outofplace_interleaved)
{
	size_t length[3] = {4096,4096,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_large_2d_outofplace_interleaved)
{
	size_t length[3] = {2,4096,1};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_normal_normal_3d_outofplace_interleaved)
{
	size_t length[3] = {32,32,32};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_mixed_radices_3d_outofplace_interleaved)
{
	size_t length[3] = {2*3*5,2*3*5,2*3*5};
	test_persistent_plan( length, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_1d_outofplace_planar)
{
	size_t length[3] = {1024,1,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_1d_outofplace_planar)
{
	size_t length[3] = {32768,1,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_normal_2d_outofplace_planar)
{
	size_t length[3] = {1024,1024,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_normal_2d_outofplace_planar)
{
	size_t length[3] = {4096,2,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_large_large_2d_outofplace_planar)
{
	size_t length[3] = {4096,4096,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_large_2d_outofplace_planar)
{
	size_t length[3] = {2,4096,1};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_normal_normal_normal_3d_outofplace_planar)
{
	size_t length[3] = {32,32,32};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_OUTOFPLACE );
}

TEST_F(clfft_PersistentPlans, DISABLED_mixed_radices_3d_outofplace_planar)
{
	size_t length[3] = {2*3*5,2*3*5,2*3*5};
	test_persistent_plan( length, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR, CLFFT_OUTOFPLACE );
}
#endif
