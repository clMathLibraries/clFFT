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
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>

#include "clFFT.h"
#include "test_constants.h"
#include "fftw_transform.h"
#include "cl_transform.h"
#include "typedefs.h"
#include "accuracy_test_common.h"

size_t super_duper_global_seed;

namespace ParameterizedTest {
	//TODO this is pasted from cl_transform.h
	// it should be put in one place for everybody to use
	/*****************************************************/
	layout::buffer_layout_t cl_layout_to_buffer_layout( clfftLayout cl_layout )
	{
		if( cl_layout == CLFFT_REAL )
			return layout::real;
		else if( cl_layout == CLFFT_HERMITIAN_PLANAR )
			return layout::hermitian_planar;
		else if( cl_layout == CLFFT_COMPLEX_PLANAR )
			return layout::complex_planar;
		else if( cl_layout == CLFFT_HERMITIAN_INTERLEAVED )
			return layout::hermitian_interleaved;
		else if( cl_layout == CLFFT_COMPLEX_INTERLEAVED )
			return layout::complex_interleaved;
		else
			throw std::runtime_error( "invalid cl_layout" );
	}

	const size_t one_gb = 1024 * 1024 * 1024;

	size_t size_of_one_point( clfftPrecision precision, clfftLayout layout )
	{
		size_t size_of_one_point;
		// size of one point will be 1 or 2, depending on whether the points are real or complex
		if( layout == CLFFT_COMPLEX_INTERLEAVED || layout == CLFFT_COMPLEX_PLANAR || layout == CLFFT_HERMITIAN_PLANAR || layout == CLFFT_HERMITIAN_INTERLEAVED )
			size_of_one_point = 2;
		else if( layout == CLFFT_REAL )
			size_of_one_point = 1;
		else throw std::invalid_argument("random_supported_problem_size: invalid layout provided");

		// each value in a point will be the size of a float or the size of a double, depending on the precision
		if( precision == CLFFT_SINGLE ) size_of_one_point *= sizeof(float);
		else if( precision == CLFFT_DOUBLE ) size_of_one_point *= sizeof(double);
		else throw std::invalid_argument("random_supported_problem_size: invalid precision provided");

		return size_of_one_point;
	}

	// the CPU can have a lot of memory hanging off of it. in these cases and when CPU is the cl device,
	// huge amounts of memory might be dedicated to each buffer. in practice, large buffer sizes _destroy_
	// performance (and can effectively hang the machine). that is bad. to get around this, we limit
	// each buffer to a GB
	size_t max_memory_size_for_one_buffer()
	{
		size_t max_mem = max_mem_available_on_cl_device(0);
		if( max_mem > one_gb )
			return one_gb;
		else
			return max_mem;
	}

	size_t max_problem_size_in_datapoints( clfftPrecision precision, clfftLayout layout )
	{
		// we divide by 32 to shrink things just a bit. otherwise problems take a billion hours each
		return max_memory_size_for_one_buffer() / size_of_one_point(precision,layout) / 32;
	}


	boost::mt19937 random_parameter_generator;
	boost::uniform_int<> distribution(1, INT_MAX);
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> >
		random_value(random_parameter_generator, distribution);

	size_t random_int() {
		return random_value();
	}

	size_t random_int(size_t max) {
		return random_value() % (max+1);
	}

	size_t random_int(size_t min, size_t max) {
		return (random_value() % ((max+1)-min)) + min;
	}

	std::vector<size_t> random_supported_problem_size( size_t dimensions, clfftPrecision precision, clfftLayout layout )
	{
		std::vector<size_t> lengths;

		std::vector<size_t> supported_radices;
		supported_radices.push_back(2);
		supported_radices.push_back(3);
		supported_radices.push_back(5);
		supported_radices.push_back(7);

		// total size of this problem should be some fraction of the total space available on the device
		size_t this_problem_size = random_int(1, max_problem_size_in_datapoints(precision,layout));

		size_t total_problem_size = 1;
		std::vector<size_t> factors;

		while( total_problem_size < this_problem_size )
		{
			size_t a_factor = supported_radices[random_int(0, supported_radices.size()-1)];
			if( total_problem_size * a_factor <= this_problem_size )
			{
				total_problem_size *= a_factor;
				factors.push_back(a_factor);
			}
			else
				break;
		} // problem size is now factored into some permutation of 2s, 3s, and 5s
		  // (exact combination stored in "factors"

		for( size_t i = 0; i < dimensions; ++i )
			lengths.push_back(1);

		// distribute the values in factors to each valid length value
		while( !factors.empty() )
		{
			size_t which_factor = random_int( 0, factors.size()-1 );
			size_t dim = random_int( 0, dimensions-1 );
			lengths[dim] *= factors[which_factor];
			factors.erase(factors.begin() + which_factor);
		}

		// by the time we reach the end, we've calculated the total problem size, split it up into valid radices, and
		// distributed those among the dimensions available

		if( lengths.size() != dimensions )
			throw std::runtime_error( "random_supported_problem_size: number of lengths does not corroborate number of dimensions" );

		return lengths;
	}

	struct Parameters {
		size_t batch_size;
		clfftPrecision precision;
		clfftDirection direction;
		clfftDim dimensions;
		std::vector<size_t> lengths;
		std::vector<size_t> input_strides;
		std::vector<size_t> output_strides;
		size_t input_distance;
		size_t output_distance;
		clfftLayout input_layout;
		clfftLayout output_layout;
		clfftResultLocation placeness;
		double forward_scale;
		double backward_scale;
		data_pattern pattern;
		size_t data_seed;
		// start scales at double. we can just cast to float at the cost of
		// a little precision if single precision is randomly chosen. no biggie

		//we want to define a maximum stride so that memory does not get out of control
		static const size_t max_stride = 5;
		static const size_t max_distance = 128;

		size_t total_size_in_points()
		{
			if( lengths.empty() )
				throw std::runtime_error( "you shouldn't be here!" );

			size_t total_size = 1;
			for( size_t i = 0; i < lengths.size(); i++ )
				total_size *= lengths[i];

			return total_size;
		}

		bool is_in_place()
		{
			if( placeness == CLFFT_INPLACE ) return true;
			else return false;
		}

		bool is_out_of_place()
		{
			return !is_in_place();
		}

		bool is_r2c()
		{
			if( input_layout == CLFFT_REAL ) return true;
			else return false;
		}

		bool is_c2r()
		{
			if( output_layout == CLFFT_REAL ) return true;
			else return false;
		}

		bool is_c2c()
		{
			if( ( input_layout == CLFFT_COMPLEX_INTERLEAVED || input_layout == CLFFT_COMPLEX_PLANAR )
				&& ( output_layout == CLFFT_COMPLEX_INTERLEAVED || output_layout == CLFFT_COMPLEX_PLANAR ) )
				return true;
			else
				return false;
		}

		bool fifty_percent_chance()
		{
			if( random_int(0,1) )
				return true;
			else
				return false;
		}

		void generate_x_strides()
		{
			input_strides.push_back( random_int(1,max_stride) );
			output_strides.push_back( random_int(1,max_stride) );
		}

		Parameters()
			: precision( static_cast<clfftPrecision>(random_int(CLFFT_SINGLE,CLFFT_DOUBLE)) )
			, dimensions( static_cast<clfftDim>(random_int(CLFFT_1D,ENDDIMENSION-1)) )
			, placeness( static_cast<clfftResultLocation>(random_int(CLFFT_INPLACE, CLFFT_OUTOFPLACE)) )
			, input_layout( static_cast<clfftLayout>(random_int(CLFFT_COMPLEX_INTERLEAVED, CLFFT_REAL)) )
			, forward_scale( static_cast<float>(random_int())/static_cast<float>(random_int()) )
			, backward_scale( static_cast<float>(random_int())/static_cast<float>(random_int()) )
			, pattern( erratic )
			, data_seed( random_int() )
		{
			try
			{
				// input and output layouts have strict requirements, so we'll base the output layout
				// off of our randomly selected input layout
				if( input_layout == CLFFT_REAL )
				{
					if( placeness == CLFFT_INPLACE )
						output_layout = CLFFT_HERMITIAN_INTERLEAVED;
					else if( placeness == CLFFT_OUTOFPLACE )
						output_layout = static_cast<clfftLayout>(random_int(CLFFT_HERMITIAN_INTERLEAVED, CLFFT_HERMITIAN_PLANAR));
					else
						throw std::runtime_error( "parameter generator invalid placeness" );
				}
				else if( input_layout == CLFFT_HERMITIAN_INTERLEAVED )
				{
					output_layout = CLFFT_REAL;
				}
				else if( input_layout == CLFFT_HERMITIAN_PLANAR )
				{
					// in-place transforms not supported with hermitian planar
					placeness = CLFFT_OUTOFPLACE;

					output_layout = CLFFT_REAL;
				}
				else if( input_layout == CLFFT_COMPLEX_INTERLEAVED || input_layout == CLFFT_COMPLEX_PLANAR )
				{
					// complex is a little simpler. we can do them together here
					if( placeness == CLFFT_INPLACE )
						output_layout = input_layout;
					else
						output_layout = static_cast<clfftLayout>(random_int(CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_PLANAR));
				}
				else
				{
					throw std::runtime_error( "parameter generator invalid input layout" );
				}

				direction = random_int(0,1) ? CLFFT_FORWARD : CLFFT_BACKWARD;

				lengths = random_supported_problem_size(dimensions, precision, input_layout);

				// strides and distances

				if( fifty_percent_chance() ) // about half the time, we just want unit strides
				{
					// input_strides and output_strides remain empty

					input_distance = 0;
					output_distance = 0;
				}
				else if( is_in_place() && is_r2c() )
				{
					generate_x_strides();

					// generate y strides
					if( dimensions >= 2 )
					{
						size_t random_y_stride = random_int(0,max_stride);
						input_strides.push_back( input_strides[0] * ( 1 + lengths[0] / 2 ) * 2 + 2 * random_y_stride );
						output_strides.push_back( output_strides[0] * ( 1 + lengths[0] / 2 ) + random_y_stride );

						// both strides need to be able to fit both the input and the output
						if( input_strides[1] > 2 * output_strides[1] )
						{
							output_strides[1] = input_strides[1] / 2;
						}
						else if( input_strides[1] < 2 * output_strides[1] )
						{
							input_strides[1] = 2 * output_strides[1];
						}
					}

					// generate z strides
					if( dimensions >= 3 )
					{
						output_strides.push_back( output_strides[1] * lengths[1] );
						input_strides.push_back( 2 * output_strides[2] );
					}

					// generate distance

					// 1D is a special case with distances, because we need to make sure
					// we have the extra padding that we would have otherwise gotten from
					// the y dimension
					if( dimensions == 1 )
					{
						input_distance = input_strides[0] * ( 1 + lengths[0] / 2 ) * 2;
						output_distance = output_strides[0] * ( 1 + lengths[0] / 2 );

						// both strides need to be able to fit both the input and the output
						if( input_distance > 2 * output_distance )
						{
							output_distance = input_distance / 2;
						}
						else if( input_distance < 2 * output_distance )
						{
							input_distance = 2 * output_distance;
						}

						size_t random_distance = random_int(0,max_distance);
						input_distance += 2 * random_distance;
						output_distance += random_distance;
					}
					else
					{
						output_distance = output_strides[dimensions-1] * lengths[dimensions-1] + random_int(0,max_distance);
						input_distance = 2 * output_distance;
					}

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
				else if( is_in_place() && is_c2r() )
				{
					generate_x_strides();

					// generate y strides
					if( dimensions >= 2 )
					{
						size_t random_y_stride = random_int(0,max_stride);
						output_strides.push_back( output_strides[0] * ( 1 + lengths[0] / 2 ) * 2 + 2 * random_y_stride );
						input_strides.push_back( input_strides[0] * ( 1 + lengths[0] / 2 ) + random_y_stride );

						// both strides need to be able to fit both the output and the input
						if( output_strides[1] > 2 * input_strides[1] )
						{
							input_strides[1] = output_strides[1] / 2;
						}
						else if( output_strides[1] < 2 * input_strides[1] )
						{
							output_strides[1] = 2 * input_strides[1];
						}
					}

					// generate z strides
					if( dimensions >= 3 )
					{
						input_strides.push_back( input_strides[1] * lengths[1] );
						output_strides.push_back( 2 * input_strides[2] );
					}

					// generate distance

					// 1D is a special case with distances, because we need to make sure
					// we have the extra padding that we would have otherwise gotten from
					// the y dimension
					if( dimensions == 1 )
					{
						output_distance = output_strides[0] * ( 1 + lengths[0] / 2 ) * 2;
						input_distance = input_strides[0] * ( 1 + lengths[0] / 2 );

						// both strides need to be able to fit both the output and the input
						if( output_distance > 2 * input_distance )
						{
							input_distance = output_distance / 2;
						}
						else if( output_distance < 2 * input_distance )
						{
							output_distance = 2 * input_distance;
						}

						size_t random_distance = random_int(0,max_distance);
						output_distance += 2 * random_distance;
						input_distance += random_distance;
					}
					else
					{
						input_distance = input_strides[dimensions-1] * lengths[dimensions-1] + random_int(0,max_distance);
						output_distance = 2 * input_distance;
					}

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
				else // placeness::in_place c2c or placeness::out_of_place
				{
					// input first
					if( fifty_percent_chance() )
					{
						// tightly packed input
						// leave stride vector empty

						input_distance = 0;
					}
					else // input has padding
					{
						for( int i = 0; i < dimensions; i++)
						{
							if( i == 0 )
								input_strides.push_back( random_int(1,max_stride) );
							else
								input_strides.push_back( lengths[i-1] * input_strides[i-1] + random_int(0,max_stride) );
						}

						input_distance = input_strides[dimensions-1] * lengths[dimensions-1] + random_int(max_distance);
					}

					// output next
					if( is_in_place() && is_c2c() )
					{
						output_strides = input_strides;
						output_distance = input_distance;
					}
					else if( fifty_percent_chance() )
					{
						// tightly packed output
						// leave stride vector empty

						output_distance = 0;
					}
					else // output has padding
					{
						for( int i = 0; i < dimensions; i++)
						{
							if( i == 0 )
								output_strides.push_back( random_int(1,max_stride) );
							else
								output_strides.push_back( lengths[i-1] * output_strides[i-1] + random_int(0,max_stride) );
						}

						output_distance = output_strides[dimensions-1] * lengths[dimensions-1] + random_int(max_distance);
					}
				}

				if( fifty_percent_chance() )
				{
					// we'll want batches sometimes . . .

					// limit the batch size, taking in account available space and size of each pass
					size_t current_problem_size;
					if( input_strides.empty() )
						current_problem_size = total_size_in_points();
					else
						current_problem_size = input_strides[dimensions-1] * lengths[dimensions-1];

					size_t max_problem_size = max_problem_size_in_datapoints( precision, input_layout );
					size_t max_batch_size_for_this_problem = max_problem_size / current_problem_size;
					if( max_batch_size_for_this_problem <= 1 )
						batch_size = 1;
					else
						batch_size = random_int( 1, max_batch_size_for_this_problem );
				}
				else
				{
					// . . . and sometimes we won't

					batch_size = 1;
				}
			}
			catch( const std::exception& err )
			{
				handle_exception(err);
			}
		}
	}; //struct Parameters

	struct plant_seed {
		time_t the_seed;

		plant_seed(time_t seed_in) :
			the_seed( seed_in )
		{
			random_parameter_generator.seed( static_cast<boost::uint32_t>( the_seed ) );
			::testing::Test::RecordProperty("parameter_seed", static_cast<unsigned int>(the_seed));
			std::cout << "Random test's seed is " << the_seed << std::endl;
		}
	};

	class TestParameterGenerator {
		private:
			std::vector<Parameters> data_sets;
		public:
			TestParameterGenerator(int number_of_data_sets)
			: initial_seed(random_test_parameter_seed)
			{
				generate(number_of_data_sets);
			}

			std::vector<Parameters> & parameter_sets() { return data_sets; }

		private:
			void generate(int number_of_data_sets) {
				for( int i=0; i<number_of_data_sets; i++ )
				{
					data_sets.push_back( Parameters() );
				}
			}

			ParameterizedTest::plant_seed initial_seed;
	}; //class TestParameterGenerator

} //namespace ParameterizedTest

class accuracy_test_random : public ::testing::TestWithParam<ParameterizedTest::Parameters> {
	protected:
		accuracy_test_random(){}
		virtual ~accuracy_test_random(){}
		virtual void SetUp(){}
		virtual void TearDown(){}
};

TEST_P( accuracy_test_random, random_transform ) {
	try {
		ParameterizedTest::Parameters params = GetParam();
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
		RecordProperty("placeness", params.placeness);
		RecordProperty("forward_scale", (int)params.forward_scale);
		RecordProperty("backward_scale", (int)params.backward_scale);
		RecordProperty("data_seed", (int)params.data_seed);

		// SO BAD
		super_duper_global_seed = params.data_seed;

		if( params.precision == CLFFT_SINGLE )
		{
			if( params.input_layout == CLFFT_REAL && ( params.output_layout == CLFFT_HERMITIAN_INTERLEAVED || params.output_layout == CLFFT_HERMITIAN_PLANAR ) )
			{
				real_to_complex<float, cl_float, fftwf_complex>( params.pattern,
																 params.lengths,
																 params.batch_size,
																 params.input_strides,
																 params.output_strides,
																 params.input_distance,
																 params.output_distance,
																 ParameterizedTest::cl_layout_to_buffer_layout( params.output_layout ),
																 params.placeness == CLFFT_INPLACE ? placeness::in_place : placeness::out_of_place );
			}
			else if( ( params.input_layout == CLFFT_HERMITIAN_INTERLEAVED || params.input_layout == CLFFT_HERMITIAN_PLANAR ) && params.output_layout == CLFFT_REAL )
			{
				complex_to_real<float, cl_float, fftwf_complex>( params.pattern,
																 params.lengths,
																 params.batch_size,
																 params.input_strides,
																 params.output_strides,
																 params.input_distance,
																 params.output_distance,
																 ParameterizedTest::cl_layout_to_buffer_layout( params.input_layout ),
																 params.placeness == CLFFT_INPLACE ? placeness::in_place : placeness::out_of_place );
			}
			else if( ( params.input_layout == CLFFT_COMPLEX_INTERLEAVED || params.input_layout == CLFFT_COMPLEX_PLANAR ) &&
					 ( params.output_layout == CLFFT_COMPLEX_INTERLEAVED || params.output_layout == CLFFT_COMPLEX_PLANAR ) )
			{
				complex_to_complex<float, cl_float, fftwf_complex>( params.pattern,
																	params.direction == CLFFT_FORWARD ? direction::forward : direction::backward,
																	params.lengths,
																	params.batch_size,
																	params.input_strides,
																	params.output_strides,
																	params.input_distance,
																	params.output_distance,
																	ParameterizedTest::cl_layout_to_buffer_layout( params.input_layout ),
																	ParameterizedTest::cl_layout_to_buffer_layout( params.output_layout ),
																	params.placeness == CLFFT_INPLACE ? placeness::in_place : placeness::out_of_place );
			}
			else
			{
				throw std::runtime_error( "bad layout combination" );
			}
		}
		else if( params.precision == CLFFT_DOUBLE )
		{
			if( params.input_layout == CLFFT_REAL && ( params.output_layout == CLFFT_HERMITIAN_INTERLEAVED || params.output_layout == CLFFT_HERMITIAN_PLANAR ) )
			{
				real_to_complex<double, cl_double, fftw_complex>( params.pattern,
																 params.lengths,
																 params.batch_size,
																 params.input_strides,
																 params.output_strides,
																 params.input_distance,
																 params.output_distance,
																 ParameterizedTest::cl_layout_to_buffer_layout( params.output_layout ),
																 params.placeness == CLFFT_INPLACE ? placeness::in_place : placeness::out_of_place );
			}
			else if( ( params.input_layout == CLFFT_HERMITIAN_INTERLEAVED || params.input_layout == CLFFT_HERMITIAN_PLANAR ) && params.output_layout == CLFFT_REAL )
			{
				complex_to_real<double, cl_double, fftw_complex>( params.pattern,
																 params.lengths,
																 params.batch_size,
																 params.input_strides,
																 params.output_strides,
																 params.input_distance,
																 params.output_distance,
																 ParameterizedTest::cl_layout_to_buffer_layout( params.input_layout ),
																 params.placeness == CLFFT_INPLACE ? placeness::in_place : placeness::out_of_place );
			}
			else if( ( params.input_layout == CLFFT_COMPLEX_INTERLEAVED || params.input_layout == CLFFT_COMPLEX_PLANAR ) &&
					 ( params.output_layout == CLFFT_COMPLEX_INTERLEAVED || params.output_layout == CLFFT_COMPLEX_PLANAR ) )
			{
				complex_to_complex<double, cl_double, fftw_complex>( params.pattern,
																	params.direction == CLFFT_FORWARD ? direction::forward : direction::backward,
																	params.lengths,
																	params.batch_size,
																	params.input_strides,
																	params.output_strides,
																	params.input_distance,
																	params.output_distance,
																	ParameterizedTest::cl_layout_to_buffer_layout( params.input_layout ),
																	ParameterizedTest::cl_layout_to_buffer_layout( params.output_layout ),
																	params.placeness == CLFFT_INPLACE ? placeness::in_place : placeness::out_of_place );
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
	clfft_RandomTest,
	accuracy_test_random,
	::testing::ValuesIn( ParameterizedTest::TestParameterGenerator
		((int)number_of_random_tests).parameter_sets())
);
