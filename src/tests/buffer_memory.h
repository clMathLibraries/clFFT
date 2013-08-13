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
#if !defined( CLFFT_BUFFER_MEMORY_H )
#define CLFFT_BUFFER_MEMORY_H

#include <vector>
#include <stdexcept>
#include <stdint.h>

uint32_t float_as_hex( float a );
uint64_t float_as_hex( double a );
uint32_t nan_as_hex( float a );
uint64_t nan_as_hex( double a );

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
template <class T>
class buffer_memory {
private:
	// Each array will have a cookie of this size placed before and after it.
	// We will initialize the cookies to NaN.
	// The user can confirm the cookies after operating on the data to confirm that
	// his or her operations are respecting the boundaries of the memory.
	size_t cookie_size;

	// requested_floats is the number of floats the user requested originally.
	// This never changes, even if the memory size is increased.
	size_t requested_floats;

	// With this and cookie_size, we can calculate the size of memory the user can access.
	// Note that this will be in units of T (so 4 bytes or 8 bytes depending on float or double).
	size_t memory_size_including_cookies;

	// Interesting stuff goes here.
	std::vector<T> memory;

public:
	 /*****************************************************/
	 // requested_number_of_floats should already take into account any strides,
	 // batch size, data layout (real, complex, hermitian, interleaved, planar)
	buffer_memory( size_t requested_number_of_floats )
		: cookie_size( 4 )
		, requested_floats( requested_number_of_floats )
		, memory_size_including_cookies( requested_number_of_floats + 2 * cookie_size )
		, memory( memory_size_including_cookies )
	{
		clear();
	}

	 /*****************************************************/
	~buffer_memory() {
	}

	 /*****************************************************/
	buffer_memory<T> & operator=( const buffer_memory<T> & that )
	{
		this->cookie_size = that.cookie_size;
		this->requested_floats = that.requested_floats;
		this->memory_size_including_cookies = that.memory_size_including_cookies;
		this->memory = that.memory;

		return *this;
	}

	 /*****************************************************/
	void check_memory_boundaries() {
		for( size_t i = 0; i < cookie_size; ++i) {
			// we need to compare hex values instead of float values so that we don't get float ambiguities
			if( float_as_hex(memory[i]) != nan_as_hex(memory[0]) ||
				float_as_hex( memory[ memory.size()-1-i ] ) != nan_as_hex(memory[0]) )
			 throw std::runtime_error("some operation wrote beyond bounds of memory");
		}
	}

	 /*****************************************************/
	void clear()
	{
		memset(&memory[0], ~0x0, memory_size_including_cookies * sizeof(T));
	}

	 /*****************************************************/
	 // note that this is in units of T (float or double)
	 // also see: size_in_bytes()
	size_t size()
	{
		return size_in_bytes() / sizeof(T);
	}

	 /*****************************************************/
	 // returns the amount of memory currently allocated to the buffer in bytes
	size_t size_in_bytes()
	{
		return (memory_size_including_cookies - 2 * cookie_size) * sizeof(T);
	}

	 /*****************************************************/
	 // N.B. memory will be cleared after this
	void increase_allocated_memory( size_t amount )
	{
		size_t new_memory_size = memory_size_including_cookies + amount;

		memory.resize( new_memory_size );
		memory_size_including_cookies = new_memory_size;

		clear();
	}

	 /*****************************************************/
	T* ptr()
	{
		return &memory[0] + cookie_size;
	}

	 /*****************************************************/
	T& operator[]( size_t index ) {
		if( index >= size() )
			throw std::runtime_error( "operator[] write out of bounds" );
		return memory[0 + cookie_size + index];
	}
};

#endif
