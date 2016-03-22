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
			, dimensions(dimensions_in)
			, batch_size(batch_size_in)
		{
			try
			{
				for (size_t i = 0; i < lengths_in.size(); i++)
					lengths.push_back(lengths_in[i]);

				if ((direction_in == CLFFT_FORWARD) || (direction_in == CLFFT_MINUS))
				{
					input_layout = CLFFT_REAL;
					output_layout = CLFFT_HERMITIAN_INTERLEAVED;
				}
				else
				{
					input_layout = CLFFT_HERMITIAN_INTERLEAVED;
					output_layout = CLFFT_REAL;
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


	class TestListGenerator {
		private:
			std::vector<ParametersPackedRealInplaceInterleaved> data_sets;
		public:
			TestListGenerator()
			{
				generate();
			}

			std::vector<ParametersPackedRealInplaceInterleaved> & parameter_sets() { return data_sets; }

		private:
			void generate()
			{
				std::vector<size_t> length;
				length.push_back(1024);
				data_sets.push_back( ParametersPackedRealInplaceInterleaved(CLFFT_SINGLE, CLFFT_FORWARD, CLFFT_1D, length, 1 ) );
			}

	}; //class TestListGenerator

} //namespace DirectedTest


class accuracy_test_directed : public ::testing::TestWithParam<DirectedTest::ParametersPackedRealInplaceInterleaved> {
	protected:
		accuracy_test_directed(){}
		virtual ~accuracy_test_directed(){}
		virtual void SetUp(){}
		virtual void TearDown(){}
};



TEST_P( accuracy_test_directed, random_transform ) {
	try {
		DirectedTest::ParametersPackedRealInplaceInterleaved params = GetParam();
		RecordProperty("batch_size", (int)params.batch_size);
		RecordProperty("precision", params.precision);
		RecordProperty("direction", params.direction);
		RecordProperty("dimensions", params.dimensions);
		RecordProperty("length_x", (int)params.lengths[0]);
		if( params.dimensions >= CLFFT_2D) RecordProperty("length_y", (int)params.lengths[1]);
		if( params.dimensions >= CLFFT_3D) RecordProperty("length_z", (int)params.lengths[2]);

		if( params.input_strides.empty() )
		{
			RecordProperty("input_strides", 0);
		}
		else
		{
			RecordProperty("input_stride_x", (int)params.input_strides[0]);
			if( params.dimensions >= CLFFT_2D) RecordProperty("input_stride_y", (int)params.input_strides[1]);
			if( params.dimensions >= CLFFT_3D) RecordProperty("input_stride_z", (int)params.input_strides[2]);
		}

		if( params.output_strides.empty() )
		{
			RecordProperty("output_strides", 0);
		}
		else
		{
			RecordProperty("output_stride_x", (int)params.output_strides[0]);
			if( params.dimensions >= CLFFT_2D) RecordProperty("output_stride_y", (int)params.output_strides[1]);
			if( params.dimensions >= CLFFT_3D) RecordProperty("output_stride_z", (int)params.output_strides[2]);
		}

		RecordProperty("input_distance", (int)params.input_distance);
		RecordProperty("output_distance", (int)params.output_distance);
		RecordProperty("input_layout", params.input_layout);
		RecordProperty("output_layout", params.output_layout);



		if( params.precision == CLFFT_SINGLE )
		{
			if( params.input_layout == CLFFT_REAL )
			{
				real_to_complex<float, cl_float, fftwf_complex>( erratic,
																 params.lengths,
																 params.batch_size,
																 params.input_strides,
																 params.output_strides,
																 params.input_distance,
																 params.output_distance,
																 DirectedTest::cl_layout_to_buffer_layout( params.output_layout ),
																 placeness::in_place );
			}
			else if( params.output_layout == CLFFT_REAL )
			{
				complex_to_real<float, cl_float, fftwf_complex>( erratic,
																 params.lengths,
																 params.batch_size,
																 params.input_strides,
																 params.output_strides,
																 params.input_distance,
																 params.output_distance,
																 DirectedTest::cl_layout_to_buffer_layout( params.input_layout ),
																 placeness::in_place );
			}
			else
			{
				throw std::runtime_error( "bad layout combination" );
			}
		}
		else if( params.precision == CLFFT_DOUBLE )
		{
			if( params.input_layout == CLFFT_REAL )
			{
				real_to_complex<double, cl_double, fftw_complex>( erratic,
																 params.lengths,
																 params.batch_size,
																 params.input_strides,
																 params.output_strides,
																 params.input_distance,
																 params.output_distance,
																 DirectedTest::cl_layout_to_buffer_layout( params.output_layout ),
																 placeness::in_place );
			}
			else if( params.output_layout == CLFFT_REAL )
			{
				complex_to_real<double, cl_double, fftw_complex>( erratic,
																 params.lengths,
																 params.batch_size,
																 params.input_strides,
																 params.output_strides,
																 params.input_distance,
																 params.output_distance,
																 DirectedTest::cl_layout_to_buffer_layout( params.input_layout ),
																 placeness::in_place );
			}
			else
			{
				throw std::runtime_error( "bad layout combination" );
			}
		}
		else
		{
			throw std::runtime_error("Random test: this code path should never be executed");
		}
	}
	catch( const std::exception& err ) {
		handle_exception(err);
	}
}

INSTANTIATE_TEST_CASE_P(
	clfft_DirectedTest,
	accuracy_test_directed,
	::testing::ValuesIn( DirectedTest::TestListGenerator().parameter_sets())
);
