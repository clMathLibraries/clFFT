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
#include <list>
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

	struct ParametersPacked
	{
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

		ParametersPacked(clfftPrecision precision_in,
			clfftDirection direction_in,
			clfftDim dimensions_in,
			const std::vector<size_t> &lengths_in,
			size_t batch_size_in)
			: precision(precision_in)
			, direction(direction_in)
			, dimensions(dimensions_in)
			, batch_size(batch_size_in)
		{
			for (size_t i = 0; i < lengths_in.size(); i++)
				lengths.push_back(lengths_in[i]);
		}
	};

	struct ParametersPackedRealInplaceInterleaved : public ParametersPacked
	{
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
			: ParametersPacked(precision_in, direction_in, dimensions_in, lengths_in, batch_size_in)
		{
			try
			{
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


	struct ParametersPackedComplexInplaceInterleaved : public ParametersPacked
	{
		ParametersPackedComplexInplaceInterleaved(clfftPrecision precision_in,
			clfftDirection direction_in,
			clfftDim dimensions_in,
			const std::vector<size_t> &lengths_in,
			size_t batch_size_in)
			: ParametersPacked(precision_in, direction_in, dimensions_in, lengths_in, batch_size_in)
		{
			try
			{
				input_strides.push_back(1);
				output_strides.push_back(1);

				input_layout = CLFFT_COMPLEX_INTERLEAVED;
				output_layout = CLFFT_COMPLEX_INTERLEAVED;

				input_distance = lengths[0];
				output_distance = lengths[0];

				for (size_t i = 1; i < lengths.size(); i++)
				{
					input_strides.push_back(input_distance);
					output_strides.push_back(output_distance);

					input_distance *= lengths[i];
					output_distance *= lengths[i];
				}
			}
			catch (const std::exception& err)
			{
				handle_exception(err);
			}
		}
	}; //struct ParametersPackedComplexInplaceInterleaved

	template <class ParameterType>
	class TestListGenerator
	{
	protected:
		std::vector<ParameterType> data_sets;
		const size_t *supported_length;
		size_t size_supported_length;

		virtual void supported_length_data()
		{
			// This array must be kept sorted in the ascending order
			static const size_t supported_length_array[] = {
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 24,
				25, 26, 27, 28, 30, 32, 33, 35, 36, 39, 40, 42, 44, 45, 48, 49, 50, 52, 54,
				55, 56, 60, 63, 64, 65, 66, 70, 72, 75, 77, 78, 80, 81, 84, 88, 90, 91, 96,
				98, 99, 100, 104, 105, 108, 110, 112, 117, 120, 121, 125, 126, 128, 130, 132,
				135, 140, 143, 144, 147, 150, 154, 156, 160, 162, 165, 168, 169, 175, 176,
				180, 182, 189, 192, 195, 196, 198, 200, 208, 210, 216, 220, 224, 225, 231,
				234, 240, 242, 243, 245, 250, 252, 256, 260, 264, 270, 273, 275, 280, 286,
				288, 294, 297, 300, 308, 312, 315, 320, 324, 325, 330, 336, 338, 343, 350,
				351, 352, 360, 363, 364, 375, 378, 384, 385, 390, 392, 396, 400, 405, 416,
				420, 429, 432, 440, 441, 448, 450, 455, 462, 468, 480, 484, 486, 490, 495,
				500, 504, 507, 512, 520, 525, 528, 539, 540, 546, 550, 560, 567, 572, 576,
				585, 588, 594, 600, 605, 616, 624, 625, 630, 637, 640, 648, 650, 660, 672,
				675, 676, 686, 693, 700, 702, 704, 715, 720, 726, 728, 729, 735, 750, 756,
				768, 770, 780, 784, 792, 800, 810, 819, 825, 832, 840, 845, 847, 858, 864,
				875, 880, 882, 891, 896, 900, 910, 924, 936, 945, 960, 968, 972, 975, 980,
				990, 1000, 1001, 1008, 1014, 1024, 1029, 1040, 1050, 1053, 1056, 1078, 1080,
				1089, 1092, 1100, 1120, 1125, 1134, 1144, 1152, 1155, 1170, 1176, 1183, 1188,
				1200, 1210, 1215, 1225, 1232, 1248, 1250, 1260, 1274, 1280, 1287, 1296, 1300,
				1320, 1323, 1331, 1344, 1350, 1352, 1365, 1372, 1375, 1386, 1400, 1404, 1408,
				1430, 1440, 1452, 1456, 1458, 1470, 1485, 1500, 1512, 1521, 1536, 1540, 1560,
				1568, 1573, 1575, 1584, 1600, 1617, 1620, 1625, 1638, 1650, 1664, 1680, 1690,
				1694, 1701, 1715, 1716, 1728, 1750, 1755, 1760, 1764, 1782, 1792, 1800, 1815,
				1820, 1848, 1859, 1872, 1875, 1890, 1911, 1920, 1925, 1936, 1944, 1950, 1960,
				1980, 2000, 2002, 2016, 2025, 2028, 2048, 2058, 2079, 2080, 2100, 2106, 2112,
				2145, 2156, 2160, 2178, 2184, 2187, 2197, 2200, 2205, 2240, 2250, 2268, 2275,
				2288, 2304, 2310, 2340, 2352, 2366, 2376, 2400, 2401, 2420, 2430, 2450, 2457,
				2464, 2475, 2496, 2500, 2520, 2535, 2541, 2548, 2560, 2574, 2592, 2600, 2625,
				2640, 2646, 2662, 2673, 2688, 2695, 2700, 2704, 2730, 2744, 2750, 2772, 2800,
				2808, 2816, 2835, 2860, 2880, 2904, 2912, 2916, 2925, 2940, 2970, 3000, 3003,
				3024, 3025, 3042, 3072, 3080, 3087, 3120, 3125, 3136, 3146, 3150, 3159, 3168,
				3185, 3200, 3234, 3240, 3250, 3267, 3276, 3300, 3328, 3360, 3375, 3380, 3388,
				3402, 3430, 3432, 3456, 3465, 3500, 3510, 3520, 3528, 3549, 3564, 3575, 3584,
				3600, 3630, 3640, 3645, 3675, 3696, 3718, 3744, 3750, 3773, 3780, 3822, 3840,
				3850, 3861, 3872, 3888, 3900, 3920, 3960, 3969, 3993, 4000, 4004, 4032, 4050,
				4056, 4095, 4096 };

			supported_length = supported_length_array;
			size_supported_length = sizeof(supported_length_array) / sizeof(supported_length_array[0]);
		}

		virtual void generate_1d(clfftDirection dir, clfftPrecision precision, size_t batch)
		{
			for (size_t i = 0; i < size_supported_length; i++)
			{
				std::vector<size_t> length;
				length.push_back(supported_length[i]);
				data_sets.push_back(ParameterType(precision, dir, CLFFT_1D, length, batch));
			}
		}

		virtual void generate_2d(clfftDirection dir, clfftPrecision precision, size_t batch)
		{
			for (size_t i = 0; i < size_supported_length; i++)
			{
				std::vector<size_t> length;
				length.push_back(supported_length[i]);
				length.push_back(supported_length[i]);
				data_sets.push_back(ParameterType(precision, dir, CLFFT_2D, length, batch));
			}
		}

		virtual void generate_3d(clfftDirection dir, clfftPrecision precision, size_t batch)
		{
			for (size_t i = 0; i < size_supported_length; i++)
			{
				std::vector<size_t> length;
				length.push_back(supported_length[i]);
				length.push_back(supported_length[i]);
				length.push_back(supported_length[i]);
				data_sets.push_back(ParameterType(precision, dir, CLFFT_3D, length, batch));

				const size_t max_3d_length = 256;
				if (supported_length[i] == max_3d_length) break;
			}
		}

	public:
		TestListGenerator() : supported_length(NULL), size_supported_length(0)
		{}

		virtual std::vector<ParameterType> & parameter_sets
			(clfftDim dimension, clfftDirection direction, clfftPrecision precision, size_t batch)
		{
			supported_length_data();

			switch (dimension)
			{
			case CLFFT_1D: generate_1d(direction, precision, batch); break;
			case CLFFT_2D: generate_2d(direction, precision, batch); break;
			case CLFFT_3D: generate_3d(direction, precision, batch); break;
			}

			return data_sets;
		}

	}; //class TestListGenerator

	template <class ParameterType>
	class TestListGenerator_Pow2 : public TestListGenerator<ParameterType>
	{
	protected:
		virtual void supported_length_data()
		{
			// This array must be kept sorted in the ascending order
			static const size_t supported_length_array[] = {
				8192, 16384, 32768, 65536, 131072, 262144, 524288,
				1048576, 2097152, 4194304};

			TestListGenerator<ParameterType>::supported_length = supported_length_array;
			TestListGenerator<ParameterType>::size_supported_length = sizeof(supported_length_array) / sizeof(supported_length_array[0]);
		}
	};


	template <class ParameterType>
	class TestListGenerator_Large_Random
	{
	protected:
		std::vector<size_t> supported_length;
		std::vector<ParameterType> data_sets;

		void GenerateSizes(size_t maximum_size)
		{
			std::list<size_t> sizes;

			size_t i = 0, j = 0, k = 0, l = 0, m = 0, n = 0;
			size_t sum, sumi, sumj, sumk, suml, summ, sumn;

			sumi = 1; i = 0;
			while (1)
			{
				sumj = 1; j = 0;
				while (1)
				{
					sumk = 1; k = 0;
					while (1)
					{
						suml = 1; l = 0;
						while (1)
						{
							summ = 1; m = 0;
							while (1)
							{
								sumn = 1; n = 0;
								while (1)
								{
									sum = (sumi*sumj*sumk*suml*summ*sumn);
									if (sum > maximum_size) break;

									sizes.push_back(sum);
									n++;
									sumn *= 2;
								}

								if (n == 0) break;
								m++;
								summ *= 3;
							}

							if ((m == 0) && (n == 0)) break;
							l++;
							suml *= 5;
						}

						if ((l == 0) && (m == 0) && (n == 0)) break;
						k++;
						sumk *= 7;
					}

					if ((k == 0) && (l == 0) && (m == 0) && (n == 0)) break;
					j++;
					sumj *= 11;
				}

				if ((j == 0) && (k == 0) && (l == 0) && (m == 0) && (n == 0)) break;
				i++;
				sumi *= 13;
			}

			sizes.sort();

			for (std::list<size_t>::const_iterator a = sizes.begin(); a != sizes.end(); a++)
				supported_length.push_back(*a);
		}

	public:
		virtual std::vector<ParameterType> & parameter_sets
			(clfftDirection direction, clfftPrecision precision, size_t num_tests)
		{
			size_t maximum_size = (precision == CLFFT_SINGLE) ? 16777216 : 4194304;

			GenerateSizes(maximum_size);

			assert(supported_length.size() < RAND_MAX);

			for (size_t i = 0; i < num_tests; i++)
			{
				size_t idx = 0;

				// choose size that has a 11 or 13 in it
				do
				{
					// choose index randomly
					idx = rand() % supported_length.size();
				} while ((supported_length[idx] % 11 != 0) && (supported_length[idx] % 13 != 0));


				std::vector<size_t> length;
				length.push_back(supported_length[idx]);
				size_t batch = maximum_size / supported_length[idx];

				data_sets.push_back(ParameterType(precision, direction, CLFFT_1D, length, batch));
			}

			return data_sets;
		}
	};


	template <class ParameterType>
	class TestListGenerator_huge_chosen : public TestListGenerator<ParameterType>
	{
	protected:
		virtual void supported_length_data()
		{
			// This array must be kept sorted in the ascending order
			static const size_t supported_length_array[] = {
				25050025, 27027000, 17320303, 19487171, 4826809, 53094899, 23030293, 214358881, 62748517 };

			TestListGenerator<ParameterType>::supported_length = supported_length_array;
			TestListGenerator<ParameterType>::size_supported_length = sizeof(supported_length_array) / sizeof(supported_length_array[0]);
		}
	};


} //namespace DirectedTest

class accuracy_test_directed_base : public ::testing::TestWithParam<DirectedTest::ParametersPacked> {
protected:
	accuracy_test_directed_base() {}
	virtual ~accuracy_test_directed_base() {}
	virtual void SetUp() {}
	virtual void TearDown() {}

public:
	static void RunTest(const DirectedTest::ParametersPacked *params_ptr)
	{
		const DirectedTest::ParametersPacked &params = *params_ptr;
		try
		{
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
				else if ((params.input_layout == CLFFT_COMPLEX_INTERLEAVED || params.input_layout == CLFFT_COMPLEX_PLANAR) &&
					(params.output_layout == CLFFT_COMPLEX_INTERLEAVED || params.output_layout == CLFFT_COMPLEX_PLANAR))
				{
					complex_to_complex<float, cl_float, fftwf_complex>(erratic,
						params.direction == CLFFT_FORWARD ? direction::forward : direction::backward,
						params.lengths,
						params.batch_size,
						params.input_strides,
						params.output_strides,
						params.input_distance,
						params.output_distance,
						DirectedTest::cl_layout_to_buffer_layout(params.input_layout),
						DirectedTest::cl_layout_to_buffer_layout(params.output_layout),
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
				else if ((params.input_layout == CLFFT_COMPLEX_INTERLEAVED || params.input_layout == CLFFT_COMPLEX_PLANAR) &&
					(params.output_layout == CLFFT_COMPLEX_INTERLEAVED || params.output_layout == CLFFT_COMPLEX_PLANAR))
				{
					complex_to_complex<double, cl_double, fftw_complex>(erratic,
						params.direction == CLFFT_FORWARD ? direction::forward : direction::backward,
						params.lengths,
						params.batch_size,
						params.input_strides,
						params.output_strides,
						params.input_distance,
						params.output_distance,
						DirectedTest::cl_layout_to_buffer_layout(params.input_layout),
						DirectedTest::cl_layout_to_buffer_layout(params.output_layout),
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

class accuracy_test_directed_real : public ::testing::TestWithParam<DirectedTest::ParametersPackedRealInplaceInterleaved> {
	protected:
		accuracy_test_directed_real() {}
		virtual ~accuracy_test_directed_real() {}
		virtual void SetUp() {}
		virtual void TearDown() {}

		virtual void accuracy_test_directed_packed_real_inplace_interleaved()
		{
			DirectedTest::ParametersPackedRealInplaceInterleaved params = GetParam();
			accuracy_test_directed_base::RunTest(&params);
		}
};


TEST_P(accuracy_test_directed_real, inplace_interleaved) { accuracy_test_directed_packed_real_inplace_interleaved(); }


INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_single_1d_fwd,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_FORWARD, CLFFT_SINGLE, 19))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_single_1d_inv,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_BACKWARD, CLFFT_SINGLE, 19))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_double_1d_fwd,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_FORWARD, CLFFT_DOUBLE, 19))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_double_1d_inv,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_BACKWARD, CLFFT_DOUBLE, 19))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_single_2d_fwd,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_2D, CLFFT_FORWARD, CLFFT_SINGLE, 3))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_single_2d_inv,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_2D, CLFFT_BACKWARD, CLFFT_SINGLE, 3))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_single_3d_fwd,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_3D, CLFFT_FORWARD, CLFFT_SINGLE, 1))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_single_3d_inv,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_3D, CLFFT_BACKWARD, CLFFT_SINGLE, 1))
	);




INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_Random_single_1d_fwd,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator_Large_Random<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_FORWARD, CLFFT_SINGLE, 200))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_Random_single_1d_inv,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator_Large_Random<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_BACKWARD, CLFFT_SINGLE, 200))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_Random_double_1d_fwd,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator_Large_Random<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_FORWARD, CLFFT_DOUBLE, 200))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_Random_double_1d_inv,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator_Large_Random<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_BACKWARD, CLFFT_DOUBLE, 200))
	);




#if 1

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_pow2_single_1d_fwd,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator_Pow2<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_FORWARD, CLFFT_SINGLE, 3))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_pow2_single_1d_inv,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator_Pow2<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_BACKWARD, CLFFT_SINGLE, 3))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_pow2_double_1d_fwd,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator_Pow2<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_FORWARD, CLFFT_DOUBLE, 3))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_pow2_double_1d_inv,
	accuracy_test_directed_real,
	::testing::ValuesIn(DirectedTest::TestListGenerator_Pow2<DirectedTest::ParametersPackedRealInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_BACKWARD, CLFFT_DOUBLE, 3))
	);

#endif



class accuracy_test_directed_complex : public ::testing::TestWithParam<DirectedTest::ParametersPackedComplexInplaceInterleaved> {
protected:
	accuracy_test_directed_complex() {}
	virtual ~accuracy_test_directed_complex() {}
	virtual void SetUp() {}
	virtual void TearDown() {}

	virtual void accuracy_test_directed_packed_complex_inplace_interleaved()
	{
		DirectedTest::ParametersPackedComplexInplaceInterleaved params = GetParam();
		accuracy_test_directed_base::RunTest(&params);
	}
};


TEST_P(accuracy_test_directed_complex, inplace_interleaved) { accuracy_test_directed_packed_complex_inplace_interleaved(); }


INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_single_1d_fwd,
	accuracy_test_directed_complex,
	::testing::ValuesIn(DirectedTest::TestListGenerator<DirectedTest::ParametersPackedComplexInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_FORWARD, CLFFT_SINGLE, 101))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_single_1d_inv,
	accuracy_test_directed_complex,
	::testing::ValuesIn(DirectedTest::TestListGenerator<DirectedTest::ParametersPackedComplexInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_BACKWARD, CLFFT_SINGLE, 101))
	);


#if 0

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_huge_chosen_single_1d_fwd,
	accuracy_test_directed_complex,
	::testing::ValuesIn(DirectedTest::TestListGenerator_huge_chosen<DirectedTest::ParametersPackedComplexInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_FORWARD, CLFFT_SINGLE, 1))
	);

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest_huge_chosen_single_1d_inv,
	accuracy_test_directed_complex,
	::testing::ValuesIn(DirectedTest::TestListGenerator_huge_chosen<DirectedTest::ParametersPackedComplexInplaceInterleaved>().parameter_sets(CLFFT_1D, CLFFT_BACKWARD, CLFFT_SINGLE, 1))
	);

#endif
