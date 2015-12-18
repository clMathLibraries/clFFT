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


#pragma once
#if !defined( CLFFT_FFTWTRANSFORM_H )
#define CLFFT_FFTWTRANSFORM_H

#include <vector>
#include "fftw3.h"
#include "buffer.h"
#include "../client/openCL.misc.h" // we need this to leverage the CLFFT_INPLACE and _OUTOFPLACE enums

enum fftw_direction {forward=-1, backward=+1};

enum fftw_transform_type {c2c, r2c, c2r};

template <typename T, typename fftw_T>
class fftw_wrapper
{};

template <>
class fftw_wrapper<float, fftwf_complex>
{
public:
	fftwf_plan plan;

	void make_plan( int x, int y, int z, int num_dimensions, int batch_size, fftwf_complex* input_ptr, fftwf_complex* output_ptr, int num_points_in_single_batch, fftw_direction direction, fftw_transform_type type )
	{
		// we need to swap x,y,z dimensions because of a row-column discrepancy between clfft and fftw
		int lengths[max_dimension] = {z, y, x};

		if( type == c2c )
		{
			plan = fftwf_plan_many_dft( num_dimensions,
										// because we swapped dimensions up above, we need to start
										// at the end of the array and count backwards to get the
										// correct dimensions passed in to fftw
										// e.g. if max_dimension is 3 and number_of_dimensions is 2:
										// lengths = {dimz, dimy, dimx}
										// lengths + 3 - 2 = lengths + 1
										// so we will skip dimz and pass in a pointer to {dimy, dimx}
										lengths+max_dimension-num_dimensions,
										batch_size,
										input_ptr, NULL,
										1, num_points_in_single_batch,
										output_ptr, NULL,
										1, num_points_in_single_batch,
										direction, FFTW_ESTIMATE);
		}
		else if( type == r2c )
		{
			plan = fftwf_plan_many_dft_r2c( num_dimensions,
											// because we swapped dimensions up above, we need to start
											// at the end of the array and count backwards to get the
											// correct dimensions passed in to fftw
											// e.g. if max_dimension is 3 and number_of_dimensions is 2:
											// lengths = {dimz, dimy, dimx}
											// lengths + 3 - 2 = lengths + 1
											// so we will skip dimz and pass in a pointer to {dimy, dimx}
											lengths+max_dimension-num_dimensions,
											batch_size,
											reinterpret_cast<float*>(input_ptr), NULL,
											1, num_points_in_single_batch,
											output_ptr, NULL,
											1, (x/2 + 1) * y * z,
											FFTW_ESTIMATE);
		}
		else if( type == c2r )
		{
			plan = fftwf_plan_many_dft_c2r( num_dimensions,
											// because we swapped dimensions up above, we need to start
											// at the end of the array and count backwards to get the
											// correct dimensions passed in to fftw
											// e.g. if max_dimension is 3 and number_of_dimensions is 2:
											// lengths = {dimz, dimy, dimx}
											// lengths + 3 - 2 = lengths + 1
											// so we will skip dimz and pass in a pointer to {dimy, dimx}
											lengths+max_dimension-num_dimensions,
											batch_size,
											input_ptr, NULL,
											1, (x/2 + 1) * y * z,
											reinterpret_cast<float*>(output_ptr), NULL,
											1, num_points_in_single_batch,
											FFTW_ESTIMATE);
		}
		else
			throw std::runtime_error( "invalid transform type in <float>make_plan" );
	}

	fftw_wrapper( int x, int y, int z, int num_dimensions, int batch_size, fftwf_complex* input_ptr, fftwf_complex* output_ptr, int num_points_in_single_batch, fftw_direction direction, fftw_transform_type type )
	{
		make_plan( x, y, z, num_dimensions, batch_size, input_ptr, output_ptr, num_points_in_single_batch, direction, type );
	}

	void destroy_plan()
	{
		fftwf_destroy_plan(plan);
	}

	~fftw_wrapper()
	{
		destroy_plan();
	}

	void execute()
	{
		fftwf_execute(plan);
	}
};

template <>
class fftw_wrapper<double, fftw_complex>
{
public:
	fftw_plan plan;

	void make_plan( int x, int y, int z, int num_dimensions, int batch_size, fftw_complex* input_ptr, fftw_complex* output_ptr, int num_points_in_single_batch, fftw_direction direction, fftw_transform_type type )
	{
		// we need to swap x,y,z dimensions because of a row-column discrepancy between clfft and fftw
		int lengths[max_dimension] = {z, y, x};

		if( type == c2c )
		{
			plan = fftw_plan_many_dft( num_dimensions,
									// because we swapped dimensions up above, we need to start
									// at the end of the array and count backwards to get the
									// correct dimensions passed in to fftw
									// e.g. if max_dimension is 3 and number_of_dimensions is 2:
									// lengths = {dimz, dimy, dimx}
									// lengths + 3 - 2 = lengths + 1
									// so we will skip dimz and pass in a pointer to {dimy, dimx}
									lengths+max_dimension-num_dimensions,
									batch_size,
									input_ptr, NULL,
									1, num_points_in_single_batch,
									output_ptr, NULL,
									1, num_points_in_single_batch,
									direction, FFTW_ESTIMATE);
		}
		else if( type == r2c )
		{
			plan = fftw_plan_many_dft_r2c( num_dimensions,
											// because we swapped dimensions up above, we need to start
											// at the end of the array and count backwards to get the
											// correct dimensions passed in to fftw
											// e.g. if max_dimension is 3 and number_of_dimensions is 2:
											// lengths = {dimz, dimy, dimx}
											// lengths + 3 - 2 = lengths + 1
											// so we will skip dimz and pass in a pointer to {dimy, dimx}
											lengths+max_dimension-num_dimensions,
											batch_size,
											reinterpret_cast<double*>(input_ptr), NULL,
											1, num_points_in_single_batch,
											output_ptr, NULL,
											1, (x/2 + 1) * y * z,
											FFTW_ESTIMATE);
		}
		else if( type == c2r )
		{
			plan = fftw_plan_many_dft_c2r( num_dimensions,
											// because we swapped dimensions up above, we need to start
											// at the end of the array and count backwards to get the
											// correct dimensions passed in to fftw
											// e.g. if max_dimension is 3 and number_of_dimensions is 2:
											// lengths = {dimz, dimy, dimx}
											// lengths + 3 - 2 = lengths + 1
											// so we will skip dimz and pass in a pointer to {dimy, dimx}
											lengths+max_dimension-num_dimensions,
											batch_size,
											input_ptr, NULL,
											1, (x/2 + 1) * y * z,
											reinterpret_cast<double*>(output_ptr), NULL,
											1, num_points_in_single_batch,
											FFTW_ESTIMATE);
		}
		else
			throw std::runtime_error( "invalid transform type in <double>make_plan" );
	}

	fftw_wrapper( int x, int y, int z, int num_dimensions, int batch_size, fftw_complex* input_ptr, fftw_complex* output_ptr, int num_points_in_single_batch, fftw_direction direction, fftw_transform_type type )
	{
		make_plan( x, y, z, num_dimensions, batch_size, input_ptr, output_ptr, num_points_in_single_batch, direction, type );
	}

	void destroy_plan()
	{
		fftw_destroy_plan(plan);
	}

	~fftw_wrapper()
	{
		destroy_plan();
	}

	void execute()
	{
		fftw_execute(plan);
	}
};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
template <typename T, typename fftw_T>
class fftw {
private:
	static const size_t tightly_packed_distance = 0;

	std::vector<size_t> _lengths;
	fftw_direction _direction;
	fftw_transform_type _type;
	layout::buffer_layout_t _input_layout, _output_layout;
	size_t _batch_size;
	buffer<T> input;
	buffer<T> output;
	fftw_wrapper<T, fftw_T> fftw_guts;

	T _forward_scale, _backward_scale;
public:
	/*****************************************************/
	fftw( const size_t number_of_dimensions_in, const size_t* lengths_in, const size_t batch_size_in, fftw_transform_type type_in )
		: _lengths( initialized_lengths( number_of_dimensions_in, lengths_in ) )
		, _direction( forward )
		, _type( type_in )
		, _input_layout( initialized_input_layout() )
		, _output_layout( initialized_output_layout() )
		, _batch_size( batch_size_in )
		, input( number_of_dimensions_in,
				lengths_in,
				NULL,
				batch_size_in,
				tightly_packed_distance,
				_input_layout,
				CLFFT_OUTOFPLACE )
		, output( number_of_dimensions_in,
				lengths_in,
				NULL,
				batch_size_in,
				tightly_packed_distance,
				_output_layout,
				CLFFT_OUTOFPLACE )
		, _forward_scale( 1.0f )
		, _backward_scale( 1.0f/T(input.number_of_data_points_single_batch()) )
		, fftw_guts( (int)_lengths[dimx], (int)_lengths[dimy], (int)_lengths[dimz],
					 (int)number_of_dimensions_in, (int)batch_size_in,
					 reinterpret_cast<fftw_T*>(input_ptr()),
					 reinterpret_cast<fftw_T*>(output_ptr()),
					 (int)(_lengths[dimx]*_lengths[dimy]*_lengths[dimz]), _direction, _type)
	{
		clear_data_buffer();
	}

	/*****************************************************/
	~fftw() {}

	/*****************************************************/
	layout::buffer_layout_t initialized_input_layout()
	{
		if( _type == c2c )
			return layout::complex_interleaved;
		else if( _type == r2c )
			return layout::real;
		else if( _type == c2r )
			return layout::hermitian_interleaved;
		else
			throw std::runtime_error( "invalid transform type in initialized_input_layout" );
	}

	/*****************************************************/
	layout::buffer_layout_t initialized_output_layout()
	{
		if( _type == c2c )
			return layout::complex_interleaved;
		else if( _type == r2c )
			return layout::hermitian_interleaved;
		else if( _type == c2r )
			return layout::real;
		else
			throw std::runtime_error( "invalid transform type in initialized_input_layout" );
	}

	/*****************************************************/
	std::vector<size_t> initialized_lengths( const size_t number_of_dimensions, const size_t* lengths_in )
	{
		std::vector<size_t> lengths( 3, 1 ); // start with 1, 1, 1

		for( size_t i = 0; i < number_of_dimensions; i++ )
		{
			lengths[i] = lengths_in[i];
		}

		return lengths;
	}

	/*****************************************************/
	T* input_ptr()
	{
		if( _input_layout == layout::real )
			return input.real_ptr();
		else if( _input_layout == layout::complex_interleaved )
			return input.interleaved_ptr();
		else if( _input_layout == layout::hermitian_interleaved )
			return input.interleaved_ptr();
		else
			throw std::runtime_error( "invalid layout in fftw::input_ptr" );
	}

	/*****************************************************/
	T* output_ptr()
	{
		if( _output_layout == layout::real )
			return output.real_ptr();
		else if( _output_layout == layout::complex_interleaved )
			return output.interleaved_ptr();
		else if( _output_layout == layout::hermitian_interleaved )
			return output.interleaved_ptr();
		else
			throw std::runtime_error( "invalid layout in fftw::output_ptr" );
	}

	// you must call either set_forward_transform() or
	// set_backward_transform() before setting the input buffer
	/*****************************************************/
	void set_forward_transform()
	{
		if( _type != c2c )
			throw std::runtime_error( "do not use set_forward_transform() except with c2c transforms" );

		if( _direction != forward )
		{
			_direction = forward;
			fftw_guts.destroy_plan();
			fftw_guts.make_plan((int)_lengths[dimx], (int)_lengths[dimy], (int)_lengths[dimz],
								(int)input.number_of_dimensions(), (int)input.batch_size(),
								reinterpret_cast<fftw_T*>(input.interleaved_ptr()), reinterpret_cast<fftw_T*>(output.interleaved_ptr()),
								(int)(_lengths[dimx]*_lengths[dimy]*_lengths[dimz]), _direction, _type);
		}
	}

	/*****************************************************/
	void set_backward_transform()
	{
		if( _type != c2c )
			throw std::runtime_error( "do not use set_backward_transform() except with c2c transforms" );

		if( _direction != backward )
		{
			_direction = backward;
			fftw_guts.destroy_plan();
			fftw_guts.make_plan((int)_lengths[dimx], (int)_lengths[dimy], (int)_lengths[dimz],
								(int)input.number_of_dimensions(), (int)input.batch_size(),
								reinterpret_cast<fftw_T*>(input.interleaved_ptr()), reinterpret_cast<fftw_T*>(output.interleaved_ptr()),
								(int)(_lengths[dimx]*_lengths[dimy]*_lengths[dimz]), _direction, _type);
		}
	}

	/*****************************************************/
	size_t size_of_data_in_bytes()
	{
		return input.size_in_bytes();
	}

	/*****************************************************/
	void forward_scale( T in )
	{
		_forward_scale = in;
	}

	/*****************************************************/
	void backward_scale( T in )
	{
		_backward_scale = in;
	}

	/*****************************************************/
	T forward_scale()
	{
		return _forward_scale;
	}

	/*****************************************************/
	T backward_scale()
	{
		return _backward_scale;
	}

	/*****************************************************/
	void set_all_data_to_value( T value )
	{
		input.set_all_to_value( value );
	}

	/*****************************************************/
	void set_all_data_to_value( T real_value, T imag_value )
	{
		input.set_all_to_value( real_value, imag_value );
	}

	/*****************************************************/
	void set_data_to_sawtooth(T max)
	{
		input.set_all_to_sawtooth( max );
	}

	/*****************************************************/
	void set_data_to_increase_linearly()
	{
		input.set_all_to_linear_increase();
	}

	/*****************************************************/
	void set_data_to_impulse()
	{
		input.set_all_to_impulse();
	}

	/*****************************************************/
	// yes, the "super duper global seed" is horrible
	// alas, i'll have TODO it better later
	void set_data_to_random()
	{
		input.set_all_to_random_data( 10, super_duper_global_seed );
	}

	/*****************************************************/
	void set_input_to_buffer( buffer<T> other_buffer ) {
		input = other_buffer;
	}

	void set_output_postcallback()
	{
		//postcallback user data
		buffer<T> userdata( 	output.number_of_dimensions(),
					output.lengths(),
					output.strides(),
					output.batch_size(),
					output.distance(),
					layout::real ,
					CLFFT_INPLACE
					);
		
		userdata.set_all_to_random_data(_lengths[0], 10);
		
		output *= userdata;
	}

	void set_input_precallback()
	{
		//precallback user data
		buffer<T> userdata( 	input.number_of_dimensions(),
					input.lengths(),
					input.strides(),
					input.batch_size(),
					input.distance(),
					layout::real ,
					CLFFT_INPLACE
					);
		
		userdata.set_all_to_random_data(_lengths[0], 10);
		
		input *= userdata;
	}

	void set_input_precallback_special()
	{
		//precallback user data
		buffer<T> userdata( 	input.number_of_dimensions(),
					input.lengths(),
					input.strides(),
					input.batch_size(),
					input.distance(),
					layout::real ,
					CLFFT_INPLACE
					);
		
		userdata.set_all_to_random_data(_lengths[0], 10);
		
		input.multiply_3pt_average(userdata);
	}

	void set_output_postcallback_special()
	{
		//postcallback user data
		buffer<T> userdata( 	output.number_of_dimensions(),
					output.lengths(),
					output.strides(),
					output.batch_size(),
					output.distance(),
					layout::real ,
					CLFFT_INPLACE
					);
		
		userdata.set_all_to_random_data(_lengths[0], 10);
		
		output.multiply_3pt_average(userdata);
	}

	/*****************************************************/
	void clear_data_buffer()
	{
		if( _input_layout == layout::real )
		{
			set_all_data_to_value( 0.0f );
		}
		else
		{
			set_all_data_to_value( 0.0f, 0.0f );
		}
	}

	/*****************************************************/
	void transform()
	{
		fftw_guts.execute();

		if( _type == c2c )
		{
			if( _direction == forward  ) {
				output.scale_data( static_cast<T>( forward_scale( ) ) );
			}
			else if( _direction == backward  ) {
				output.scale_data( static_cast<T>( backward_scale( ) ) );
			}
		}
		else if( _type == r2c )
		{
			output.scale_data( static_cast<T>( forward_scale( ) ) );
		}
		else if( _type == c2r )
		{
			output.scale_data( static_cast<T>( backward_scale( ) ) );
		}
		else
			throw std::runtime_error( "invalid transform type in fftw::transform()" );
	}

	/*****************************************************/
	buffer<T> & result()
	{
		return output;
	}

	/*****************************************************/
	buffer<T> & input_buffer()
	{
		return input;
	}
};

#endif
