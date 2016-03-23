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


#include <algorithm>
#include <vector>
#include <numeric>
#include <memory>
#include <time.h>
#include <gtest/gtest.h>

#include "clFFT.h"
#include "test_constants.h"
#include "fftw_transform.h"
#include "cl_transform.h"
#include "typedefs.h"
#include "accuracy_test_common.h"

namespace DirectedTest {

	layout::buffer_layout_t cl_layout_to_buffer_layout(clfftLayout cl_layout)
	{
		if (cl_layout == CLFFT_REAL)
			return layout::real;
		else if (cl_layout == CLFFT_HERMITIAN_PLANAR)
			return layout::hermitian_planar;
		else if (cl_layout == CLFFT_COMPLEX_PLANAR)
			return layout::complex_planar;
		else if (cl_layout == CLFFT_HERMITIAN_INTERLEAVED)
			return layout::hermitian_interleaved;
		else if (cl_layout == CLFFT_COMPLEX_INTERLEAVED)
			return layout::complex_interleaved;
		else
			throw std::runtime_error("invalid cl_layout");
	}


	struct ParametersPackedRealInplaceInterleaved {
		
		// directed inputs
		size_t batch_size;
		clfftPrecision precision;
		clfftDirection direction;
		clfftDim dimensions;
		std::vector<size_t> lengths;


		// calculated
		std::vector<size_t> input_strides;
		std::vector<size_t> output_strides;
		size_t input_distance;
		size_t output_distance;
		clfftLayout input_layout;
		clfftLayout output_layout;

		bool is_r2c()
		{
			if (input_layout == CLFFT_REAL) return true;
			else return false;
		}

		bool is_c2r()
		{
			if (output_layout == CLFFT_REAL) return true;
			else return false;
		}

		ParametersPackedRealInplaceInterleaved(	clfftPrecision precision_in,
												clfftDirection direction_in,
												clfftDim dimensions_in,
												const std::vector<size_t> &lengths_in,
												size_t batch_size_in)
			: precision(precision_in)
			, direction(direction_in)
			, dimensions(dimensions_in)
			, batch_size(batch_size_in)
		{
			try
			{
				for (size_t i = 0; i < lengths_in.size(); i++)
					lengths.push_back(lengths_in[i]);

				input_strides.push_back(1);
				output_strides.push_back(1);

				if ((direction_in == CLFFT_FORWARD) || (direction_in == CLFFT_MINUS))
				{
					input_layout = CLFFT_REAL;
					output_layout = CLFFT_HERMITIAN_INTERLEAVED;

					input_distance = 2 * (1 + lengths[0]/2);
					output_distance = 1 + lengths[0] / 2;
				}
				else
				{
					input_layout = CLFFT_HERMITIAN_INTERLEAVED;
					output_layout = CLFFT_REAL;

					input_distance = 1 + lengths[0] / 2;
					output_distance = 2 * (1 + lengths[0] / 2);
				}

				for (size_t i = 1; i < lengths.size(); i++)
				{
					input_strides.push_back(input_distance);
					output_strides.push_back(output_distance);

					input_distance *= lengths[i];
					output_distance *= lengths[i];
				}

				if( is_r2c() )
				{
					// check for ok
					if( dimensions >= 2 )
						if( input_strides[1] != 2 * output_strides[1] )
							throw std::runtime_error( "invalid stride y generated for r2c" );

					if( dimensions >= 3 )
						if( input_strides[2] != 2 * output_strides[2] )
							throw std::runtime_error( "invalid stride z generated for r2c" );

					if( input_distance != 2 * output_distance )
						throw std::runtime_error( "invalid distance generated for r2c" );
				}

				if( is_c2r() )
				{
					// check for ok
					if( dimensions >= 2 )
						if( output_strides[1] != 2 * input_strides[1] )
							throw std::runtime_error( "invalid stride y generated for c2r" );

					if( dimensions >= 3 )
						if( output_strides[2] != 2 * input_strides[2] )
							throw std::runtime_error( "invalid stride z generated for c2r" );

					if( output_distance != 2 * input_distance )
						throw std::runtime_error( "invalid distance generated for c2r" );
				}

			}
			catch( const std::exception& err )
			{
				handle_exception(err);
			}
		}
	}; //struct ParametersPackedRealInplaceInterleaved


	class TestListGenerator
	{
	private:
		std::vector<ParametersPackedRealInplaceInterleaved> data_sets;
		const size_t *supported_length;
		size_t size_supported_length;

		void supported_length_data()
		{
			// This array must be kept sorted in the ascending order
			static const size_t supported_length_array[] = {
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28,
				30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 75, 80,
				81, 84, 90, 96, 98, 100, 105, 108, 112, 120, 125, 126, 128, 135, 140, 144,
				147, 150, 160, 162, 168, 175, 180, 189, 192, 196, 200, 210, 216, 224, 225,
				240, 243, 245, 250, 252, 256, 270, 280, 288, 294, 300, 315, 320, 324, 336,
				343, 350, 360, 375, 378, 384, 392, 400, 405, 420, 432, 441, 448, 450, 480,
				486, 490, 500, 504, 512, 525, 540, 560, 567, 576, 588, 600, 625, 630, 640,
				648, 672, 675, 686, 700, 720, 729, 735, 750, 756, 768, 784, 800, 810, 840,
				864, 875, 882, 896, 900, 945, 960, 972, 980, 1000, 1008, 1024, 1029, 1050,
				1080, 1120, 1125, 1134, 1152, 1176, 1200, 1215, 1225, 1250, 1260, 1280, 1296,
				1323, 1344, 1350, 1372, 1400, 1440, 1458, 1470, 1500, 1512, 1536, 1568, 1575,
				1600, 1620, 1680, 1701, 1715, 1728, 1750, 1764, 1792, 1800, 1875, 1890, 1920,
				1944, 1960, 2000, 2016, 2025, 2048, 2058, 2100, 2160, 2187, 2205, 2240, 2250,
				2268, 2304, 2352, 2400, 2401, 2430, 2450, 2500, 2520, 2560, 2592, 2625, 2646,
				2688, 2700, 2744, 2800, 2835, 2880, 2916, 2940, 3000, 3024, 3072, 3087, 3125,
				3136, 3150, 3200, 3240, 3360, 3375, 3402, 3430, 3456, 3500, 3528, 3584, 3600,
				3645, 3675, 3750, 3780, 3840, 3888, 3920, 3969, 4000, 4032, 4050, 4096 };

			supported_length = supported_length_array;
			size_supported_length = sizeof(supported_length_array) / sizeof(supported_length_array[0]);
		}

		void generate_1d(clfftDirection dir)
		{
			for (size_t i = 0; i < size_supported_length; i++)
			{
				std::vector<size_t> length;
				length.push_back(supported_length[i]);
				data_sets.push_back(ParametersPackedRealInplaceInterleaved(CLFFT_SINGLE, dir, CLFFT_1D, length, 1));
			}
		}

		void generate_2d(clfftDirection dir)
		{
			for (size_t i = 0; i < size_supported_length; i++)
			{
				std::vector<size_t> length;
				length.push_back(supported_length[i]);
				length.push_back(supported_length[i]);
				data_sets.push_back(ParametersPackedRealInplaceInterleaved(CLFFT_SINGLE, dir, CLFFT_2D, length, 1));
			}
		}

		void generate_3d(clfftDirection dir)
		{
			for (size_t i = 0; i < size_supported_length; i++)
			{
				std::vector<size_t> length;
				length.push_back(supported_length[i]);
				length.push_back(supported_length[i]);
				length.push_back(supported_length[i]);
				data_sets.push_back(ParametersPackedRealInplaceInterleaved(CLFFT_SINGLE, dir, CLFFT_3D, length, 1));

				const size_t max_3d_length = 256;
				if (supported_length[i] == max_3d_length) break;
			}
		}

	public:
		TestListGenerator(clfftDim dimension, clfftDirection direction)
		{
			supported_length = NULL;
			size_supported_length = 0;
			supported_length_data();

			switch (dimension)
			{
			case CLFFT_1D: generate_1d(direction); break;
			case CLFFT_2D: generate_2d(direction); break;
			case CLFFT_3D: generate_3d(direction); break;
			}
		}

		std::vector<ParametersPackedRealInplaceInterleaved> & parameter_sets() { return data_sets; }

	}; //class TestListGenerator

} //namespace DirectedTest


class accuracy_test_directed : public ::testing::TestWithParam<DirectedTest::ParametersPackedRealInplaceInterleaved> {
	protected:
		accuracy_test_directed() {}
		virtual ~accuracy_test_directed() {}
		virtual void SetUp() {}
		virtual void TearDown() {}

		virtual void accuracy_test_directed_packed_real_inplace_interleaved()
		{
			try
			{
				DirectedTest::ParametersPackedRealInplaceInterleaved params = GetParam();
				RecordProperty("batch_size", (int)params.batch_size);
				RecordProperty("precision", params.precision);
				RecordProperty("direction", params.direction);
				RecordProperty("dimensions", params.dimensions);
				RecordProperty("length_x", (int)params.lengths[0]);
				if (params.dimensions >= CLFFT_2D) RecordProperty("length_y", (int)params.lengths[1]);
				if (params.dimensions >= CLFFT_3D) RecordProperty("length_z", (int)params.lengths[2]);

				if (params.input_strides.empty())
				{
					RecordProperty("input_strides", 0);
				}
				else
				{
					RecordProperty("input_stride_x", (int)params.input_strides[0]);
					if (params.dimensions >= CLFFT_2D) RecordProperty("input_stride_y", (int)params.input_strides[1]);
					if (params.dimensions >= CLFFT_3D) RecordProperty("input_stride_z", (int)params.input_strides[2]);
				}

				if (params.output_strides.empty())
				{
					RecordProperty("output_strides", 0);
				}
				else
				{
					RecordProperty("output_stride_x", (int)params.output_strides[0]);
					if (params.dimensions >= CLFFT_2D) RecordProperty("output_stride_y", (int)params.output_strides[1]);
					if (params.dimensions >= CLFFT_3D) RecordProperty("output_stride_z", (int)params.output_strides[2]);
				}

				RecordProperty("input_distance", (int)params.input_distance);
				RecordProperty("output_distance", (int)params.output_distance);
				RecordProperty("input_layout", params.input_layout);
				RecordProperty("output_layout", params.output_layout);



				if (params.precision == CLFFT_SINGLE)
				{
					if (params.input_layout == CLFFT_REAL)
					{
						real_to_complex<float, cl_float, fftwf_complex>(erratic,
							params.lengths,
							params.batch_size,
							params.input_strides,
							params.output_strides,
							params.input_distance,
							params.output_distance,
							DirectedTest::cl_layout_to_buffer_layout(params.output_layout),
							placeness::in_place);
					}
					else if (params.output_layout == CLFFT_REAL)
					{
						complex_to_real<float, cl_float, fftwf_complex>(erratic,
							params.lengths,
							params.batch_size,
							params.input_strides,
							params.output_strides,
							params.input_distance,
							params.output_distance,
							DirectedTest::cl_layout_to_buffer_layout(params.input_layout),
							placeness::in_place);
					}
					else
					{
						throw std::runtime_error("bad layout combination");
					}
				}
				else if (params.precision == CLFFT_DOUBLE)
				{
					if (params.input_layout == CLFFT_REAL)
					{
						real_to_complex<double, cl_double, fftw_complex>(erratic,
							params.lengths,
							params.batch_size,
							params.input_strides,
							params.output_strides,
							params.input_distance,
							params.output_distance,
							DirectedTest::cl_layout_to_buffer_layout(params.output_layout),
							placeness::in_place);
					}
					else if (params.output_layout == CLFFT_REAL)
					{
						complex_to_real<double, cl_double, fftw_complex>(erratic,
							params.lengths,
							params.batch_size,
							params.input_strides,
							params.output_strides,
							params.input_distance,
							params.output_distance,
							DirectedTest::cl_layout_to_buffer_layout(params.input_layout),
							placeness::in_place);
					}
					else
					{
						throw std::runtime_error("bad layout combination");
					}
				}
				else
				{
					throw std::runtime_error("Random test: this code path should never be executed");
				}
			}
			catch (const std::exception& err)
			{
				handle_exception(err);
			}
		}
};


TEST_P(accuracy_test_directed, real_inplace) { accuracy_test_directed_packed_real_inplace_interleaved(); }


INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_fwd_1d,
	accuracy_test_directed,
	::testing::ValuesIn(DirectedTest::TestListGenerator(CLFFT_1D, CLFFT_FORWARD).parameter_sets())
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_inv_1d,
	accuracy_test_directed,
	::testing::ValuesIn(DirectedTest::TestListGenerator(CLFFT_1D, CLFFT_BACKWARD).parameter_sets())
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_fwd_2d,
	accuracy_test_directed,
	::testing::ValuesIn(DirectedTest::TestListGenerator(CLFFT_2D, CLFFT_FORWARD).parameter_sets())
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_inv_2d,
	accuracy_test_directed,
	::testing::ValuesIn(DirectedTest::TestListGenerator(CLFFT_2D, CLFFT_BACKWARD).parameter_sets())
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_fwd_3d,
	accuracy_test_directed,
	::testing::ValuesIn(DirectedTest::TestListGenerator(CLFFT_3D, CLFFT_FORWARD).parameter_sets())
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_inv_3d,
	accuracy_test_directed,
	::testing::ValuesIn(DirectedTest::TestListGenerator(CLFFT_3D, CLFFT_BACKWARD).parameter_sets())
	);

