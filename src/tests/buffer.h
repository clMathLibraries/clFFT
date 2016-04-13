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
#if !defined( CLFFT_BUFFER_H )
#define CLFFT_BUFFER_H

#include <cmath>
#include <complex>
#include <stdexcept>
#include <memory>
#include <vector>
#include <utility>
#include <sstream>
#include "../include/clFFT.h"
#include "test_constants.h"
#include <boost/random.hpp>
#include <stdint.h>
#include "buffer_memory.h"

/*****************************************************/
/*****************************************************/
template< typename T >
bool floats_are_about_equal( T a, T b) {
	// explicit check to see if a and b are both zero-ish . . .
	if( fabs(a) < 0.00001f && fabs(b) < 0.00001f) return true;
	// . . . and if not, we'll see if they're the same-ish
	return ( fabs(a-b) > fabs(a*tolerance) ) ? false : true;
}

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
struct index_t {
	size_t x, y, z, batch;

	index_t( size_t inx, size_t iny, size_t inz, size_t inbatch )
		: x(inx)
		, y(iny)
		, z(inz)
		, batch(inbatch)
	{}
};

namespace layout
{
	// buffer_layout_t will be used to let class buffer know how many instances of buffer_memory to make and their sizes
	enum buffer_layout_t
	{
		real,
		complex_interleaved,
		complex_planar,
		hermitian_interleaved,
		hermitian_planar
	};
}

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
template <class T>
class buffer {
private:
	// we need to save the requested length x, because
	// if we change the buffer from complex to real,
	// (as in a round-trip test) we need to be able to
	// get back to the original length of x. in the case
	// of an odd transform length, that's not possible
	// due to round-off error unless we explicitly save it
	size_t _requested_length_x;
	size_t _number_of_dimensions;
	size_t _batch_size;
	size_t _distance;
	layout::buffer_layout_t _layout;
	clfftResultLocation _placeness;

	std::vector< size_t > _lengths;
	std::vector< size_t > _strides;
	bool _tightly_packed_strides;
	bool _tightly_packed_distance;

	static const size_t tightly_packed = 0;

	// if real or planar:
	// _the_buffers[re] will hold the real portion
	// _the_buffers[im] will hold the imaginary portion (planar only)
	// if interleaved:
	// _the_buffers[interleaved] will hold the whole banana
	std::vector< buffer_memory< T > > _the_buffers;

	enum
	{
		interleaved = 0,
		re = 0, // real
		im = 1 // imaginary
	};

public:
	/*****************************************************/
	buffer( const size_t dimensions_in,
			const size_t* lengths_in,
			const size_t* strides_in,
			const size_t batch_size_in,
			const size_t distance_in,
			const layout::buffer_layout_t layout_in,
			const clfftResultLocation placeness_in
		  )
		: _number_of_dimensions( dimensions_in )
		, _batch_size( batch_size_in )
		, _distance( distance_in )
		, _layout( layout_in )
		, _placeness( placeness_in )
		, _lengths()
		, _strides()
		, _the_buffers()
	{
		initialize_lengths(lengths_in);
		initialize_strides(strides_in);
		initialize_distance(distance_in);
		create_buffer_memory();
		clear();
	}

	/*****************************************************/
	~buffer()
	{}

	/*****************************************************/
	// this assignment operator only copies _data_.
	// it does not change the rest of the buffer information
	// and in fact, it requires that the buffer sizes be the same going in
	buffer<T> & operator=( buffer<T> & that )
	{
		if( this->is_real() != that.is_real() ||
			this->is_hermitian() != that.is_hermitian() ||
			this->is_complex() != that.is_complex() )
		{
			throw std::runtime_error( "Buffers must be the same layout type for assignment operator" );
		}

		if( this->_number_of_dimensions != that._number_of_dimensions ||
			this->_batch_size != that._batch_size ||
			this->_lengths != that._lengths )
		{
			throw std::runtime_error( "Buffers must be the same size for assignment operator" );
		}

		if( this->is_real() )
		{
			for( size_t batch = 0; batch < batch_size(); batch++ ) {
				for( size_t z = 0; z < length(dimz); z++ ) {
					for( size_t y = 0; y < length(dimy); y++ ) {
						for( size_t x = 0; x < length(dimx); x++ ) {
							this->set_one_data_point( that.real(x,y,z,batch), x, y, z, batch );
						}
					}
				}
			}
		}
		else
		{
			for( size_t batch = 0; batch < batch_size(); batch++ ) {
				for( size_t z = 0; z < length(dimz); z++ ) {
					for( size_t y = 0; y < length(dimy); y++ ) {
						for( size_t x = 0; x < length(dimx); x++ ) {
							this->set_one_data_point( that.real(x,y,z,batch), that.imag(x,y,z,batch), x, y, z, batch );
						}
					}
				}
			}
		}

		return *this;
	}

private:
	/*****************************************************/
	void preinitialize_lengths_to_1_1_1()
	{
		_lengths.clear();

		for( int i = 0; i < max_dimension; ++i ) {
			_lengths.push_back(1);
		}
	}

	/*****************************************************/
	void initialize_lengths(const size_t* lengths_in)
	{
		preinitialize_lengths_to_1_1_1();

		for( size_t i = 0; i < _number_of_dimensions; ++i )
		{
			_lengths[i] = lengths_in[i];
		}

		_requested_length_x = _lengths[dimx];
		adjust_length_x_for_hermitian_buffers();
	}

	/*****************************************************/
	void adjust_length_x_for_hermitian_buffers()
	{
		// complex-to-complex transforms do not require any change
		// to the number of points in the buffer

		// real buffers also never require a change to the number of
		// points in the buffer

		// a hermitian buffer with a length of "X" will actually
		// have X/2 + 1 points (the other half-ish are conjugates
		// and do not need to be stored). lenY and lenZ are never
		// modified
		if( is_hermitian() )
		{
			_lengths[dimx] = _lengths[dimx] / 2 + 1;
		}
	}

	/*****************************************************/
	void preinitialize_strides_to_1_1_1()
	{
		_strides.clear();

		for( int i = 0; i < max_dimension; ++i ) {
			_strides.push_back(1);
		}
	}

	/*****************************************************/
	void initialize_strides(const size_t* strides_in)
	{
		preinitialize_strides_to_1_1_1();

		// we need to calculate the strides if tightly packed
		if( strides_in == nullptr ) {
			_strides[dimx] = 1;
			for( size_t i = 1; i < _number_of_dimensions; ++i )
			{
				_strides[i] = _strides[i-1]*_lengths[i-1];
			}

			_tightly_packed_strides = true;
		}
		// we do not need to calculate anything if the user specifies strides
		// we just copy the input strides into place
		else
		{
			for( size_t i = 0; i < _number_of_dimensions; ++i )
			{
				_strides[i] = strides_in[i];
			}

			_tightly_packed_strides = false;
		}
	}

	/*****************************************************/
	void initialize_distance(const size_t distance_in)
	{
		if( distance_in == tightly_packed )
		{
			// calculate distance if not passed in
			_distance = _lengths[_number_of_dimensions-1] * _strides[_number_of_dimensions-1];

			_tightly_packed_distance = true;
		}
		else
		{
			// or copy it if passed in
			_distance = distance_in;

			_tightly_packed_distance = false;
		}
	}

	/*****************************************************/
	void create_buffer_memory()
	{
		if( is_real() )
		{
			// just one real buffer
			_the_buffers.push_back( buffer_memory< T >( total_number_of_points_including_data_and_intervening() ) );

			increase_memory_allocation_for_real_in_place_buffers();
		}
		else if( is_planar() )
		{
			// one real buffer
			_the_buffers.push_back( buffer_memory< T >( total_number_of_points_including_data_and_intervening() ) );
			// and one imaginary buffer
			_the_buffers.push_back( buffer_memory< T >( total_number_of_points_including_data_and_intervening() ) );
		}
		else if( is_interleaved() )
		{
			// one double-wide interleaved buffer
			_the_buffers.push_back( buffer_memory< T >( 2 * total_number_of_points_including_data_and_intervening() ) );
		}
	}

	/*****************************************************/
	size_t amount_of_extra_padding_per_x()
	{
		if( length(dimx) % 2 == 0 ) // even lengths of x add 2 per row
			return 2;
		else // odd lengths of x add 1 per row
			return 1;
	}

	/*****************************************************/
	void adjust_strides_and_distance_for_in_place_real_buffer()
	{
		if( is_real() )
		{
			if( is_in_place() )
			{
				size_t amount_to_add_for_this_dimension = stride(dimx) * amount_of_extra_padding_per_x();

				// strides first
				if( number_of_dimensions() >= 2 )
				{
					_strides[dimy] += amount_to_add_for_this_dimension;
				}

				if( number_of_dimensions() == 3 )
				{
					amount_to_add_for_this_dimension *= length(dimy);
					_strides[dimz] += amount_to_add_for_this_dimension;
				}

				// distance next
				if( number_of_dimensions() == 1 )
				{
					_distance += amount_to_add_for_this_dimension;
				}
				else if( number_of_dimensions() == 2 )
				{
					_distance += ( amount_to_add_for_this_dimension * length(dimy) );
				}
				else if( number_of_dimensions() == 3 )
				{
					_distance += ( amount_to_add_for_this_dimension * length(dimz) );
				}
				else throw std::runtime_error( "invalid dimensions in adjust_strides_and_distance_for_in_place_real_buffer()" );
			}
			else throw std::runtime_error( "this buffer is out of place and shouldn't be adjusting strides" );
		}
		else throw std::runtime_error( "this buffer is unreal and shouldn't be adjusting strides" );
	}

	/*****************************************************/
	void increase_memory_allocation_for_real_in_place_buffers()
	{
		// when performing an in-place, real-to-hermitian transform,
		// we want a little extra space to account for the larger size
		// of the hermitian output.

		// each row in the X dimension should have enough space for 2 extra reals
		// (to account for the one extra complex number that will be put
		// into the buffer after the transform)

		// we don't want to change the length, because the number of points
		// in the transform isn't changing. we only want to change the
		// amount of memory reserved
		if( is_real() )
		{
			if( is_in_place() )
			{
				if( _tightly_packed_strides && _tightly_packed_distance )
				{
					// request extra memory
					_the_buffers[re].increase_allocated_memory( amount_of_extra_padding_per_x() * stride(dimx) * length(dimy) * length(dimz) * batch_size() );

					// adjust strides/distances so that the padding is at the end of each row in the Xth dimension
					adjust_strides_and_distance_for_in_place_real_buffer();
				}
			}
		}
	}

	/*****************************************************/
	size_t index( const size_t x, const size_t y=0, const size_t z=0, const size_t batch=0)
	{
		size_t interleaved_offset = 1;

		// if this buffer is interleaved, the index should actually be double what it appears.
		// interleaved_offset will accomplish this magical doubling.
		if( is_interleaved() )
			interleaved_offset = 2;

		size_t the_index = ( stride(dimx) * x + stride(dimy) * y + stride(dimz) * z + distance() * batch ) * interleaved_offset;

		return the_index;
	}

	/*****************************************************/
	size_t next_index( const size_t x, const size_t y=0, const size_t z=0, const size_t batch=0)
	{
		if( x+1 < length(dimx))
			return index( x+1, y, z, batch );
		else if( y+1 < length(dimy) )
			return index( 0, y+1, z, batch );
		else if( z+1 < length(dimz) )
			return index( 0, 0, z+1, batch );
		else if( batch+1 < batch_size() )
			return index( 0, 0, 0, batch+1 );
		else
			// we are at the last point
			// return the location immediately after the last point
			return index( 0, 0, 0, batch+1 );
	}

	/*****************************************************/
	bool points_are_about_equal( buffer<T> & other_buffer, size_t x, size_t y, size_t z, size_t batch )
	{
		if( is_real() )
			return floats_are_about_equal<T>( real(x, y, z, batch), other_buffer.real(x, y, z, batch) );
		else if( is_complex() || is_hermitian() )
			return ( floats_are_about_equal<T>( real(x, y, z, batch), other_buffer.real(x, y, z, batch) ) &&
					 floats_are_about_equal<T>( imag(x, y, z, batch), other_buffer.imag(x, y, z, batch) ) );
		else
			throw std::runtime_error( "invalid layout in points_are_about_equal()" );
	}

	/*****************************************************/
	size_t buffer_mismatches( buffer<T> & other_buffer, bool compare_method)
	{
		std::vector< index_t > mismatched_point_indices;

		if (compare_method == pointwise_compare)
		{
			for( size_t batch = 0; batch < batch_size(); batch++ )
				for( size_t z = 0; z < length(dimz); z++ )
					for( size_t y = 0; y < length(dimy); y++ )
						for( size_t x = 0; x < length(dimx); x++ )
							if( !points_are_about_equal( other_buffer, x, y, z, batch ) )
							{
								mismatched_point_indices.push_back( index_t(x, y, z, batch));
							}

			const size_t max_mismatches_output = default_number_of_mismatches_to_output;

			if( mismatched_point_indices.size() != 0 && max_mismatches_output != 0 && suppress_output == false) {
				std::cout << std::endl << std::dec << mismatched_point_indices.size() << " of " << batch_size() * number_of_data_points_single_batch()
					<<" data points did not match.  The first " << max_mismatches_output << " (max) mismatching points follow:" << std::endl;

				std::cout << std::endl << "(array index)(index) ";
				std::cout << "[test value (dec)] / [expected value (dec)]";
				std::cout << std::endl;
				for( size_t i = 0; i < max_mismatches_output && i < mismatched_point_indices.size(); i++ )
				{
					index_t mismatch = mismatched_point_indices[i];

					std::cout
						<< std::dec << "(" << mismatched_point_indices.at(i).batch << ")"
						<< std::dec << "(" << mismatched_point_indices.at(i).x << "," << mismatched_point_indices.at(i).y << "," << mismatched_point_indices.at(i).z << ") ";
					std::cout
						<< real( mismatch.x, mismatch.y, mismatch.z, mismatch.batch );

					if( is_complex() || is_hermitian() )
					{
						std::cout << "+i*" << imag( mismatch.x, mismatch.y, mismatch.z, mismatch.batch );
					}
					std::cout
						<< " / " << other_buffer.real( mismatch.x, mismatch.y, mismatch.z, mismatch.batch );

					if( is_complex() || is_hermitian() )
					{
						std::cout << "+i*" << other_buffer.imag( mismatch.x, mismatch.y, mismatch.z, mismatch.batch );
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
			}
			return mismatched_point_indices.size();
		}
		else
		{
			//RMS accuracy judgement

			size_t problem_size_per_transform = length(dimx) * length(dimy) * length(dimz);
			double rmse_tolerance_this = rmse_tolerance * sqrt((double)problem_size_per_transform / 4096.0);

			for (size_t batch = 0; batch < batch_size(); batch++) {

				double maxMag = 0.0, maxMagInv = 1.0;

				// Compute RMS error relative to maximum magnitude
				double rms = 0;

				for (size_t z = 0; z < length(dimz); z++) {
					for (size_t y = 0; y < length(dimy); y++) {
						for (size_t x = 0; x < length(dimx); x++) {
							double ex_r, ex_i, ac_r, ac_i;
							double mag;

							ex_r = other_buffer.real(x, y, z, batch);
							ac_r = real(x, y, z, batch);

							if (other_buffer.is_complex() || other_buffer.is_hermitian())
								ex_i = other_buffer.imag(x, y, z, batch);
							else
								ex_i = 0;

							if (other_buffer.is_complex() || other_buffer.is_hermitian())
								ac_i = imag(x, y, z, batch);
							else
								ac_i = 0;

							// find maximum magnitude
							mag = ex_r*ex_r + ex_i*ex_i;
							maxMag = (mag > maxMag) ? mag : maxMag;

							// compute square error
							rms += ((ex_r - ac_r)*(ex_r - ac_r) + (ex_i - ac_i)*(ex_i - ac_i));
						}
					}
				}

				if (maxMag > magnitude_lower_limit)
				{
					maxMagInv = 1.0 / maxMag;
				}

				rms = sqrt(rms*maxMagInv);

				if (fabs(rms) > rmse_tolerance_this)
				{
					if (suppress_output == false)
						std::cout << std::endl << "RMSE accuracy judgement failure -- RMSE = " << std::dec << rms <<
							", maximum allowed RMSE = " << std::dec << rmse_tolerance_this << std::endl;
					return 1;
				}
			}

			return 0;
		}
	}

public:
	/*****************************************************/
	bool operator==( buffer<T> & other_buffer )
	{
		// complexity of each dimension must be the same
		if( ( is_real() && !other_buffer.is_real() ) || ( !is_real() && other_buffer.is_real() ) ||
			( is_hermitian() && !other_buffer.is_hermitian() ) || ( !is_hermitian() && other_buffer.is_hermitian() ) ||
			( is_complex() && !other_buffer.is_complex() ) || ( !is_complex() && other_buffer.is_complex() ) )
		{
			return false;
		}

		// batch_size of the data must be the same
		if( batch_size() != other_buffer.batch_size() )
		{
			return false;
		}

		// dimensionality of the data must be the same
		if( number_of_dimensions() != other_buffer.number_of_dimensions() )
		{
			return false;
		}

		// size of each dimension must be the same
		for( size_t i = 0; i < number_of_dimensions(); ++i )
		{
			if( length(i) != other_buffer.length(i)) return false;
		}

		size_t number_deaths = 0;
		number_deaths += buffer_mismatches( other_buffer, comparison_type);

		if( number_deaths == 0 ) return true;
		else return false;
	}

	/*****************************************************/
	bool operator!=( buffer<T> & other_buffer )
	{
		return !( *this == other_buffer );
	}

	void operator*=( buffer<T> & other_buffer )
	{
		size_t the_index;
		T* base_ptr;
		T* real_ptr;
		T* imag_ptr;

		if( is_interleaved() )
		{
			base_ptr = _the_buffers[interleaved].ptr();
		}
		else if ( is_planar() )
		{
			real_ptr = _the_buffers[re].ptr();
			imag_ptr = _the_buffers[im].ptr();
		}
		else if ( is_real() )
		{
			base_ptr = _the_buffers[re].ptr();
		}

		for( size_t batch = 0; batch < batch_size(); batch++ )
			for( size_t z = 0; z < length(dimz); z++ )
				for( size_t y = 0; y < length(dimy); y++ )
					for( size_t x = 0; x < length(dimx); x++ )
					{		
						the_index = index(x, y, z, batch);
						if( is_interleaved() )
						{
							*( base_ptr + the_index ) *= other_buffer.real(x, y, z, batch);
		
							the_index = the_index + 1; // the imaginary component immediately follows the real
							if (other_buffer.is_real())
							{
								*( base_ptr + the_index ) *= other_buffer.real(x, y, z, batch);
							}
							else
							{	
								*( base_ptr + the_index ) *= other_buffer.imag(x, y, z, batch);
							}
						}
						else if ( is_planar() )
						{
							*( real_ptr + the_index ) *= other_buffer.real(x, y, z, batch);
							
							if (other_buffer.is_real())
							{
								*( imag_ptr + the_index ) *= other_buffer.real(x, y, z, batch);
							}
							else
							{
								*( imag_ptr + the_index ) *= other_buffer.imag(x, y, z, batch);
							}
						}
						else if ( is_real() )
						{
							*( base_ptr + the_index ) *= other_buffer.real(x, y, z, batch);
						}
					}
	}

	//Calculates a 3 point average of other_buffer and
	//multiplies with buffer
	//only real layout is supported for other_buffer currently
	void multiply_3pt_average( buffer<T> & other_buffer )
	{
		if (!other_buffer.is_real())
		{
			throw std::runtime_error( "only real layout is supported currently for other_buffer" );
		}

		size_t the_index, o_the_index;
		T *base_ptr, *o_base_ptr;
		T *real_ptr;
		T *imag_ptr;
		T o_prev_val, o_next_val;
		T average;

		if( is_interleaved() )
		{
			base_ptr = _the_buffers[interleaved].ptr();
		}
		else if ( is_planar() )
		{
			real_ptr = _the_buffers[re].ptr();
			imag_ptr = _the_buffers[im].ptr();
		}
		else if ( is_real() )
		{
			base_ptr = _the_buffers[re].ptr();
		}
		o_base_ptr = other_buffer.real_ptr();

		for( size_t batch = 0; batch < batch_size(); batch++ )
			for( size_t z = 0; z < length(dimz); z++ )
				for( size_t y = 0; y < length(dimy); y++ )
					for( size_t x = 0; x < length(dimx); x++ )
					{		
						the_index = index(x, y, z, batch);
						o_the_index = other_buffer.index(x, y, z, batch);
						o_prev_val = o_the_index <= 0 ? 0 : *(o_base_ptr + o_the_index - 1);
						o_next_val = o_the_index >= (other_buffer.total_number_of_points_including_data_and_intervening() - 1) ? 0 : *(o_base_ptr + o_the_index +  1);
						
						average = (o_prev_val + *(o_base_ptr + o_the_index) + o_next_val)/ 3.0f ;

						if( is_interleaved() )
						{
							*( base_ptr + the_index ) *= average;
		
							the_index = the_index + 1; // the imaginary component immediately follows the real
							*( base_ptr + the_index ) *= average;
						}
						else if ( is_planar() )
						{
							*( real_ptr + the_index ) *= average;
							
							*( imag_ptr + the_index ) *= average;
						}
						else if ( is_real() )
						{
							*( base_ptr + the_index ) *= average;
						}
					}
	}

	/*****************************************************/
	// strides and distance are those of the output (that is, the new hermitian buffer)
	void change_real_to_hermitian( const size_t* strides_in, const size_t distance_in )
	{
		if( !is_real() || !is_in_place() )
		{
			throw std::runtime_error( "can only change a real buffer used in an in-place transform to a hermitian one" );
		}

		// we currently only support hermitian interleaved for in-place transforms
		_layout = layout::hermitian_interleaved;
		adjust_length_x_for_hermitian_buffers();
		initialize_strides(strides_in);
		initialize_distance(distance_in);
	}

	/*****************************************************/
	// strides and distance are those of the output (that is, the new real buffer)
	void change_hermitian_to_real( const size_t* strides_in, const size_t distance_in )
	{
		// we currently only support hermitian interleaved for in-place transforms
		if( _layout != layout::hermitian_interleaved || !is_in_place() )
		{
			throw std::runtime_error( "can only change a hermitian interleaved buffer used in an in-place transform to a real one" );
		}

		_layout = layout::real;
		_lengths[dimx] = _requested_length_x;
		initialize_strides(strides_in);
		initialize_distance(distance_in);
	}

	/*****************************************************/
	bool is_real()
	{
		return _layout == layout::real;
	}

	/*****************************************************/
	bool is_complex()
	{
		return _layout == layout::complex_interleaved || _layout == layout::complex_planar;
	}

	/*****************************************************/
	bool is_hermitian()
	{
		return _layout == layout::hermitian_interleaved || _layout == layout::hermitian_planar;
	}

	/*****************************************************/
	bool is_planar()
	{
		return _layout == layout::complex_planar || _layout == layout::hermitian_planar;
	}

	/*****************************************************/
	bool is_interleaved()
	{
		return _layout == layout::complex_interleaved || _layout == layout::hermitian_interleaved;
	}

	/*****************************************************/
	bool is_in_place()
	{
		if( _placeness == CLFFT_INPLACE ) return true;
		else if( _placeness == CLFFT_OUTOFPLACE) return false;
		else throw std::runtime_error( "invalid placeness value in is_in_place()" );
	}

	/*****************************************************/
	T* interleaved_ptr()
	{
		if( is_interleaved() )
			return _the_buffers[interleaved].ptr();
		else
			throw std::runtime_error( "interleaved_ptr() is only available on interleaved buffers" );
	}

	/*****************************************************/
	T* real_ptr()
	{
		if( is_planar() || is_real() )
			return _the_buffers[re].ptr();
		else
			throw std::runtime_error( "real() is only available on real and planar buffers" );
	}

	/*****************************************************/
	T* imag_ptr()
	{
		if( is_planar() )
			return _the_buffers[im].ptr();
		else
			throw std::runtime_error( "imag_ptr() is only available on planar buffers" );
	}

	/*****************************************************/
	T real( const size_t x, const size_t y=0, const size_t z=0, const size_t batch=0 )
	{
		size_t this_index = index( x, y, z, batch );

		// all layouts will have a real component
		// using [re] will catch the real component for
		// layout::interleaved as well
		T this_value = _the_buffers[re][this_index];
		return this_value;
	}

	/*****************************************************/
	T imag( const size_t x, const size_t y=0, const size_t z=0, const size_t batch=0 )
	{
		size_t this_index = index( x, y, z, batch );

		if( is_real() )
			throw std::runtime_error( "imag() is not available for this real buffer" );
		else if( is_planar() )
			return _the_buffers[im][this_index];
		else if( is_interleaved() )
			// index always points to the real component of an interleaved number
			// the following memory location is the imaginary component
			return _the_buffers[interleaved][this_index + 1];
		else
			throw std::runtime_error( "invalid layout type in imag()" );
	}

	/*****************************************************/
	std::complex<T> complex( const size_t x, const size_t y=0, const size_t z=0, const size_t batch=0 )
	{
		if( is_real() )
			throw std::runtime_error( "complex() is not available for this real buffer" );
		else if( is_complex() || is_hermitian() )
		{
			std::complex<T> this_complex( real( x, y, z, batch ), imag( x, y, z, batch ) );
			return this_complex;
		}
		else
			throw std::runtime_error( "invalid layout type in complex()" );
	}

	/*****************************************************/
	size_t number_of_dimensions()
	{
		return _number_of_dimensions;
	}

	/*****************************************************/
	size_t number_of_data_points_single_batch()
	{
		size_t number_of_points = 1;
		for( size_t i = 0; i < _number_of_dimensions; ++i )
		{
			number_of_points *= length(i);
		}
		return number_of_points;
	}

	/*****************************************************/
	size_t number_of_data_points()
	{
		return number_of_data_points_single_batch() * batch_size();
	}

	/*****************************************************/
	// note that this returns the size in number of points and
	// does not take layout into consideration. this will yield
	// the same number for real, interleaved, and planar layouts.
	// whomever uses this information will need to know if they
	// want 1x buffer of this size (real), 2x buffer of this
	// size (planar), or 1x double-wide buffer (interleaved)
	size_t total_number_of_points_including_data_and_intervening()
	{
		return distance() * batch_size();
	}

	/*****************************************************/
	// note that this will return the size of ONE BUFFER in bytes
	// for real and interleaved, that doesn't change anything
	// for planar, you will get the size of the real _or_ the imaginary
	//			(which should always be the same)
	size_t size_in_bytes()
	{
		return _the_buffers[0].size_in_bytes();
	}

	/*****************************************************/
	size_t length(size_t dim)
	{
		return _lengths[dim];
	}

	/*****************************************************/
	size_t stride(size_t dim)
	{
		return _strides[dim];
	}

	/*****************************************************/
	size_t* lengths()
	{
		return &_lengths[0];
	}

	/*****************************************************/
	size_t* strides()
	{
		return &_strides[0];
	}

	/*****************************************************/
	size_t batch_size()
	{
		return _batch_size;
	}

	/*****************************************************/
	size_t distance()
	{
		return _distance;
	}

	/*****************************************************/
	void clear()
	{
		// for all batches

		if( is_real() )
			set_all_to_value( 0.0f );
		else
			set_all_to_value( 0.0f, 0.0f );
	}

	/*****************************************************/
	void set_one_data_point( T real, const size_t x, const size_t y, const size_t z, const size_t batch )
	{
		if( is_real() )
		{
			T* base_ptr = _the_buffers[re].ptr();
			size_t real_index = index(x, y, z, batch);

			*( base_ptr + real_index ) = real;
		}
		else
			throw std::runtime_error( "attempting to use real data point setter for complex or hermitian buffer" );
	}

	/*****************************************************/
	void set_one_data_point( T real, T imag, const size_t x, const size_t y, const size_t z, const size_t batch )
	{
		if( is_real() )
			throw std::runtime_error( "attempting to use complex data point setter for real buffer" );
		else if( is_interleaved() )
		{
			T* base_ptr = _the_buffers[interleaved].ptr();
			size_t real_index = index(x, y, z, batch);
			size_t imag_index = real_index + 1; // the imaginary component immediately follows the real

			*( base_ptr + real_index ) = real;
			*( base_ptr + imag_index ) = imag;
		}
		else // planar
		{
			T* real_ptr = _the_buffers[re].ptr();
			T* imag_ptr = _the_buffers[im].ptr();
			size_t the_index = index(x, y, z, batch);

			*( real_ptr + the_index ) = real;
			*( imag_ptr + the_index ) = imag;
		}
	}

	/*****************************************************/
	void set_all_to_value( T real )
	{
		// for all batches

		for( size_t batch = 0; batch < batch_size(); batch++ ) {
			for( size_t z = 0; z < length(dimz); z++ ) {
				for( size_t y = 0; y < length(dimy); y++ ) {
					for( size_t x = 0; x < length(dimx); x++ ) {
						set_one_data_point( real, x, y, z, batch );
					}
				}
			}
		}
	}

	/*****************************************************/
	void set_all_to_value( T real, T imag )
	{
		// for all batches

		for( size_t batch = 0; batch < batch_size(); batch++ ) {
			for( size_t z = 0; z < length(dimz); z++ ) {
				for( size_t y = 0; y < length(dimy); y++ ) {
					for( size_t x = 0; x < length(dimx); x++ ) {
						set_one_data_point( real, imag, x, y, z, batch );
					}
				}
			}
		}
	}

	/*****************************************************/
	void set_all_to_linear_increase()
	{
		// for all batches

		size_t val = 1;
		for( size_t batch = 0; batch < batch_size(); batch++ ) {
			for( size_t z = 0; z < length(dimz); z++ ) {
				for( size_t y = 0; y < length(dimy); y++ ) {
					for( size_t x = 0; x < length(dimx); x++ ) {
						if( is_real() )
						{
							set_one_data_point( static_cast<T>(val), x, y, z, batch );
						}

						else
						{
							set_one_data_point( static_cast<T>(val), static_cast<T>(val) + 0.5f, x, y, z, batch );
						}

						++val;
					}
				}
			}
		}
	}

	/*****************************************************/
	void set_all_to_sawtooth( T amplitude )
	{
		// for all batches

		for( size_t batch = 0; batch < batch_size(); batch++ )
		{
			for( size_t z = 0; z < length(dimz); z++ )
			{
				for( size_t y = 0; y < length(dimy); y++ )
				{
					// waveform will be 1 period of sawtooth
					size_t number_of_points_in_one_period = length(dimx);
					size_t number_of_points_on_one_line = number_of_points_in_one_period / 2;

					// the sawtooth will start at 0 and increase to amplitude at T/2
					// at T/2, value will change to -amplitude and increase back up to 0 at T
					// if there are an odd number of points in the whole period,
					// we'll make a stop at 0 in the middle of the jump
					T value = 0.0f;
					T per_point_delta = amplitude / (number_of_points_on_one_line - 1);

					for( size_t x = 0; x < number_of_points_in_one_period; x++) {
						if( is_real() )
						{
							set_one_data_point( value, x, y, z, batch);
						}
						else
						{
							// for the real value, we want the sawtooth as described above
							// for the imaginary value, we want the 2 times the inverse
							//		(so that real and imaginary don't match, possibly obscuring errors)
							set_one_data_point( value, -2.0f * value, x, y, z, batch);
						}

						// if we're at T/2, we want to saw on down to the negative amplitude . . .
						if( floats_are_about_equal( value, amplitude ) )
						{
							if( number_of_points_in_one_period % 2 != 0 ) // odd, we need to add the 0
							{
								x++;
								if( is_real() )
								{
									set_one_data_point( 0.0f, x, y, z, batch);
								}
								else
								{
									set_one_data_point( 0.0f, 0.0f, x, y, z, batch);
								}
							}
							value = -1 * amplitude;
						}
						// . . . otherwise, keep going up
						else value += per_point_delta;
					}
				}
			}
		}
	}

	/*****************************************************/
	void set_all_to_random_data( size_t max_value, size_t seed ) {
		// for all batches

		boost::mt19937 random_data_generator;
		boost::uniform_int<> distribution(1, INT_MAX);
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> >
			random_value(random_data_generator, distribution);
		random_data_generator.seed( static_cast<boost::uint32_t>( seed ) );

		for( size_t batch = 0; batch < batch_size(); batch++) {
			for( size_t z = 0; z < length(dimz); z++) {
				for( size_t y = 0; y < length(dimy); y++) {
					for( size_t x = 0; x < length(dimx); x++) {
						int val = random_value() % (max_value + 1); // pluck a random value
						if( random_value() % 2 ) val *= -1; // make it negative about 50% of the time

						if( is_real() )
						{
							set_one_data_point( static_cast<T>(val), x, y, z, batch );
						}

						else
						{
							set_one_data_point( static_cast<T>(val), static_cast<T>(val), x, y, z, batch );
						}
					}
				}
			}
		}
	}

	/*****************************************************/
	void set_all_to_impulse()
	{
		// for all batches
		clear();

		for( size_t batch = 0; batch < batch_size(); batch++ )
		{
			if( is_real() )
				set_one_data_point( static_cast<T>(number_of_data_points_single_batch()), 0, 0, 0, batch);
			else
				set_one_data_point( static_cast<T>(number_of_data_points_single_batch()), 0.0f, 0, 0, 0, batch);
		}
	}



	/*****************************************************/
	void scale_data( T scale) {
		// for all batches

		for( size_t batch = 0; batch < batch_size(); batch++ )
		{
			for( size_t z = 0; z < length(dimz); z++ )
			{
				for( size_t y = 0; y < length(dimy); y++ )
				{
					for( size_t x = 0; x < length(dimx); x++ )
					{
						if( is_real() )
						{
							T this_value = real(x, y, z, batch);
							T scaled_value = this_value * scale;
							set_one_data_point( scaled_value, x, y, z, batch );
						}
						else
						{
							T this_real = real(x, y, z, batch);
							T this_imag = imag(x, y, z, batch);

							T scaled_real = this_real * scale;
							T scaled_imag = this_imag * scale;
							set_one_data_point( scaled_real, scaled_imag, x, y, z, batch );
						}
					}
				}
			}
		}
	}

	/*****************************************************/
	void make_sure_padding_was_not_overwritten()
	{
		// check before and after memory first
		for( size_t i = 0; i < _the_buffers.size(); i++ )
		{
			_the_buffers[i].check_memory_boundaries();
		}

		if( _tightly_packed_strides && _tightly_packed_distance) return; // nothing worth checking

		size_t intervening_point_touched = 0;

		for( size_t batch = 0; batch < batch_size(); batch++)
		{
			for( size_t z = 0; z < length(dimz); z++)
			{
				for( size_t y = 0; y < length(dimy); y++)
				{
					for( size_t x = 0; x < length(dimx); x++)
					{
						size_t this_point = index(x, y, z, batch);
						size_t next_point = next_index(x, y, z, batch);

						if( is_planar() )
						{
							if( this_point < _the_buffers[re].size() && this_point + 1 != next_point)
							{
								for( size_t i = this_point+1; i < next_point; i++)
								{
									T this_real = _the_buffers[re][i];
									T this_imag = _the_buffers[im][i];

									if( nan_as_hex(this_real) != float_as_hex(this_real)
										|| nan_as_hex(this_imag) != float_as_hex(this_imag) )
									{
										++intervening_point_touched;
									}
								}
							}
						}
						else if( is_real() )
						{
							if( this_point < _the_buffers[re].size() && this_point + 1 != next_point)
							{
								for( size_t i = this_point+1; i < next_point; i++)
								{
									T this_real = _the_buffers[re][i];

									if( nan_as_hex(this_real) != float_as_hex(this_real) )
									{
										++intervening_point_touched;
									}
								}
							}
						}
						else if( is_interleaved() )
						{
							if( this_point < _the_buffers[re].size() && this_point + 1 != next_point)
							{
								// NOTE whereas real and planar initialize i = this_point+1,
								// we want this_point+2 for interleaved so that we skip the
								// imaginary value of the point
								for( size_t i = this_point+2; i < next_point; i++)
								{
									T this_real = _the_buffers[interleaved][i];

									if( nan_as_hex(this_real) != float_as_hex(this_real) )
									{
										++intervening_point_touched;
									}
								}
							}
						}
						else
							throw std::runtime_error( "invalid layout in make_sure_memory_between_data_points_was_not_touched()" );
					}
				}
			}
		}

		EXPECT_EQ( 0, intervening_point_touched );
	}
};

#endif
