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
#include<math.h>

#include "test_constants.h"
#include "fftw_transform.h"
#include "cl_transform.h"
#include "buffer.h"
#include "typedefs.h"
#include <stdexcept>
#include <vector>

namespace placeness
{
	enum placeness_t { in_place = CLFFT_INPLACE, out_of_place = CLFFT_OUTOFPLACE };
}

enum data_pattern { impulse, sawtooth, value, erratic };

namespace direction
{
	enum direction_t { forward, backward };
}

clfftResultLocation cl_placeness( placeness::placeness_t placeness );
clfftLayout cl_layout( layout::buffer_layout_t layout_in );


/*****************************************************/
/*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
template< class T, class cl_T, class fftw_T >
void complex_to_complex( data_pattern pattern, direction::direction_t direction,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t in_layout, layout::buffer_layout_t out_layout,
	placeness::placeness_t placeness,
	T scale = 1.0f )
{
	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(in_layout), cl_layout(out_layout),
		cl_placeness(placeness) );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, c2c );

	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		reference.set_data_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f, 2.5f );
		reference.set_all_data_to_value( 2.0f, 2.5f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		reference.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		reference.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_complex()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	if( direction == direction::forward )
	{
		test_fft.set_forward_transform();
		test_fft.forward_scale( scale );

		reference.set_forward_transform();
		reference.forward_scale( scale );
	}
	else if( direction == direction::backward )
	{
		test_fft.set_backward_transform();
		test_fft.backward_scale( scale );

		reference.set_backward_transform();
		reference.backward_scale( scale );
	}
	else
		throw std::runtime_error( "invalid direction in complex_to_complex()" );
	reference.transform();
	test_fft.transform();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

/*****************************************************/
/*****************************************************/
// complex to complex transform with precallback
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
template< class T, class cl_T, class fftw_T >
void precallback_complex_to_complex( data_pattern pattern, direction::direction_t direction,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t in_layout, layout::buffer_layout_t out_layout,
	placeness::placeness_t placeness, T scale = 1.0f, bool hasUserDatatype = false )
{
	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(in_layout), cl_layout(out_layout),
		cl_placeness(placeness) );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, c2c );

	//initialize input
	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		reference.set_data_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f, 2.5f );
		reference.set_all_data_to_value( 2.0f, 2.5f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		reference.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		reference.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_complex()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	//set precallback values
	if (hasUserDatatype)
	{
		test_fft.set_input_precallback_userdatatype();
	}
	else
	{
		test_fft.set_input_precallback();
	}
	reference.set_input_precallback();

	if( direction == direction::forward )
	{
		test_fft.set_forward_transform();
		test_fft.forward_scale( scale );

		reference.set_forward_transform();
		reference.forward_scale( scale );
	}
	else if( direction == direction::backward )
	{
		test_fft.set_backward_transform();
		test_fft.backward_scale( scale );

		reference.set_backward_transform();
		reference.backward_scale( scale );
	}
	else
		throw std::runtime_error( "invalid direction in complex_to_complex()" );

	reference.transform();
	test_fft.transform();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

/*****************************************************/
/*****************************************************/
// complex to complex transform with postcallback
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
template< class T, class cl_T, class fftw_T >
void postcallback_complex_to_complex( data_pattern pattern, direction::direction_t direction,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t in_layout, layout::buffer_layout_t out_layout,
	placeness::placeness_t placeness, T scale = 1.0f, bool hasUserDatatype = false )
{
	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(in_layout), cl_layout(out_layout),
		cl_placeness(placeness) );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, c2c );

	//initialize input
	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		reference.set_data_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f, 2.5f );
		reference.set_all_data_to_value( 2.0f, 2.5f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		reference.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		reference.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_complex()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	//set postcallback values
	if (hasUserDatatype)
	{
		//test_fft.set_input_precallback_userdatatype();
	}
	else
	{
		test_fft.set_output_postcallback();
	}
	
	if( direction == direction::forward )
	{
		test_fft.set_forward_transform();
		test_fft.forward_scale( scale );

		reference.set_forward_transform();
		reference.forward_scale( scale );
	}
	else if( direction == direction::backward )
	{
		test_fft.set_backward_transform();
		test_fft.backward_scale( scale );

		reference.set_backward_transform();
		reference.backward_scale( scale );
	}
	else
		throw std::runtime_error( "invalid direction in complex_to_complex()" );

	reference.transform();
	test_fft.transform();

	reference.set_output_postcallback();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

/*****************************************************/
/*****************************************************/
// complex to complex transform with pre and post callback
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
template< class T, class cl_T, class fftw_T >
void pre_and_post_callback_complex_to_complex( data_pattern pattern, direction::direction_t direction,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t in_layout, layout::buffer_layout_t out_layout,
	placeness::placeness_t placeness, T scale = 1.0f, bool withLDS = false)
{
	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(in_layout), cl_layout(out_layout),
		cl_placeness(placeness) );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, c2c );

	//initialize input
	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		reference.set_data_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f, 2.5f );
		reference.set_all_data_to_value( 2.0f, 2.5f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		reference.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		reference.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_complex()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	//set callback values
	if (withLDS)
	{
		unsigned int localMemSize = 64 * sizeof(T);
		test_fft.set_input_precallback(localMemSize);
		reference.set_input_precallback_special();

		test_fft.set_output_postcallback(localMemSize);
	}
	else
	{
		test_fft.set_input_precallback();
		reference.set_input_precallback();

		//set postcallback values
		test_fft.set_output_postcallback();
	}

	if( direction == direction::forward )
	{
		test_fft.set_forward_transform();
		test_fft.forward_scale( scale );

		reference.set_forward_transform();
		reference.forward_scale( scale );
	}
	else if( direction == direction::backward )
	{
		test_fft.set_backward_transform();
		test_fft.backward_scale( scale );

		reference.set_backward_transform();
		reference.backward_scale( scale );
	}
	else
		throw std::runtime_error( "invalid direction in complex_to_complex()" );

	reference.transform();
	test_fft.transform();

	//update reference for postcallback
	if (withLDS)
	{
		reference.set_output_postcallback_special();
	}
	else
	{
		reference.set_output_postcallback();
	}

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

/*****************************************************/
/*****************************************************/
// complex to complex transform with precallback function that uses LDS
template< class T, class cl_T, class fftw_T >
void precallback_complex_to_complex_lds( data_pattern pattern, direction::direction_t direction,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t in_layout, layout::buffer_layout_t out_layout,
	placeness::placeness_t placeness, 
	T scale = 1.0f )
{
	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(in_layout), cl_layout(out_layout),
		cl_placeness(placeness) );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, c2c );

	//initialize input
	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		reference.set_data_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f, 2.5f );
		reference.set_all_data_to_value( 2.0f, 2.5f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		reference.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		reference.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_complex()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	//set precallback values
	//Test assumes 64 length data
	unsigned int localMemSize = 64 * sizeof(T);
	test_fft.set_input_precallback(localMemSize);
	reference.set_input_precallback_special();

	if( direction == direction::forward )
	{
		test_fft.set_forward_transform();
		test_fft.forward_scale( scale );

		reference.set_forward_transform();
		reference.forward_scale( scale );
	}
	else if( direction == direction::backward )
	{
		test_fft.set_backward_transform();
		test_fft.backward_scale( scale );

		reference.set_backward_transform();
		reference.backward_scale( scale );
	}
	else
		throw std::runtime_error( "invalid direction in complex_to_complex()" );

	reference.transform();
	test_fft.transform();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

/*****************************************************/
/*****************************************************/
// complex to complex transform with postcallback function that uses LDS
template< class T, class cl_T, class fftw_T >
void postcallback_complex_to_complex_lds( data_pattern pattern, direction::direction_t direction,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t in_layout, layout::buffer_layout_t out_layout,
	placeness::placeness_t placeness, T scale = 1.0f )
{
	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(in_layout), cl_layout(out_layout),
		cl_placeness(placeness) );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, c2c );

	//initialize input
	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		reference.set_data_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f, 2.5f );
		reference.set_all_data_to_value( 2.0f, 2.5f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		reference.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		reference.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_complex()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	//set postcallback values
	//Test assumes 64 length data
	unsigned int localMemSize = 64 * sizeof(T);
	test_fft.set_output_postcallback(localMemSize);
		
	if( direction == direction::forward )
	{
		test_fft.set_forward_transform();
		test_fft.forward_scale( scale );

		reference.set_forward_transform();
		reference.forward_scale( scale );
	}
	else if( direction == direction::backward )
	{
		test_fft.set_backward_transform();
		test_fft.backward_scale( scale );

		reference.set_backward_transform();
		reference.backward_scale( scale );
	}
	else
		throw std::runtime_error( "invalid direction in complex_to_complex()" );

	reference.transform();
	test_fft.transform();

	reference.set_output_postcallback_special();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

/*****************************************************/
/*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
// input layout is always real
template< class T, class cl_T, class fftw_T >
void real_to_complex( data_pattern pattern,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t out_layout,
	placeness::placeness_t placeness,
	T scale = 1.0f )
{
	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(layout::real), cl_layout(out_layout),
		cl_placeness(placeness) );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, r2c );

	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		reference.set_data_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f );
		reference.set_all_data_to_value( 2.0f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		reference.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		reference.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in real_to_complex()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	test_fft.forward_scale( scale );
	reference.forward_scale( scale );

	test_fft.transform();
	reference.transform();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

/*****************************************************/
/*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
// input layout is always real
template< class T, class cl_T, class fftw_T >
void precallback_real_to_complex( data_pattern pattern,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t out_layout,
	placeness::placeness_t placeness,
	T scale = 1.0f )
{
	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(layout::real), cl_layout(out_layout),
		cl_placeness(placeness) );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, r2c );

	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		reference.set_data_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f );
		reference.set_all_data_to_value( 2.0f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		reference.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		reference.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in real_to_complex()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	test_fft.set_input_precallback();
	reference.set_input_precallback();

	test_fft.forward_scale( scale );
	reference.forward_scale( scale );

	test_fft.transform();
	reference.transform();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

/*****************************************************/
/*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
// input layout is always real
template< class T, class cl_T, class fftw_T >
void postcallback_real_to_complex( data_pattern pattern,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t out_layout,
	placeness::placeness_t placeness,
	T scale = 1.0f )
{
	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(layout::real), cl_layout(out_layout),
		cl_placeness(placeness) );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, r2c );

	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		reference.set_data_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f );
		reference.set_all_data_to_value( 2.0f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		reference.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		reference.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in real_to_complex()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	test_fft.forward_scale( scale );
	reference.forward_scale( scale );

	//set postcallback values
	test_fft.set_output_postcallback();
	
	test_fft.transform();
	reference.transform();

	reference.set_output_postcallback();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

/*****************************************************/
/*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
// output layout is always real
template< class T, class cl_T, class fftw_T >
void complex_to_real( data_pattern pattern,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t in_layout,
	placeness::placeness_t placeness,
	T scale = 1.0f )
{
	fftw<T, fftw_T> data_maker( lengths.size(), &lengths[0], batch, r2c );

	if( pattern == sawtooth )
	{
		data_maker.set_data_to_sawtooth(1.0f);
	}
	else if( pattern == value )
	{
		data_maker.set_all_data_to_value(2.0f);
	}
	else if( pattern == impulse )
	{
		data_maker.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		data_maker.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_real()" );
	}

	data_maker.transform();

	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(in_layout), cl_layout(layout::real),
		cl_placeness(placeness) );
	test_fft.set_input_to_buffer( data_maker.result() );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, c2r );
	reference.set_input_to_buffer(data_maker.result());

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	test_fft.backward_scale( scale );
	reference.backward_scale( scale );

	test_fft.transform();
	reference.transform();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

/*****************************************************/
/*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
// output layout is always real
template< class T, class cl_T, class fftw_T >
void precallback_complex_to_real( data_pattern pattern,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t in_layout,
	placeness::placeness_t placeness,
	T scale = 1.0f )
{
	fftw<T, fftw_T> data_maker( lengths.size(), &lengths[0], batch, r2c );

	if( pattern == sawtooth )
	{
		data_maker.set_data_to_sawtooth(1.0f);
	}
	else if( pattern == value )
	{
		data_maker.set_all_data_to_value(2.0f);
	}
	else if( pattern == impulse )
	{
		data_maker.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		data_maker.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_real()" );
	}

	data_maker.transform();

	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(in_layout), cl_layout(layout::real),
		cl_placeness(placeness) );
	test_fft.set_input_to_buffer( data_maker.result() );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, c2r );
	reference.set_input_to_buffer(data_maker.result());

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	test_fft.set_input_precallback();
	reference.set_input_precallback();

	test_fft.backward_scale( scale );
	reference.backward_scale( scale );

	test_fft.transform();
	reference.transform();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

template< class T, class cl_T, class fftw_T >
void postcallback_complex_to_real( data_pattern pattern,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	layout::buffer_layout_t in_layout,
	placeness::placeness_t placeness,
	T scale = 1.0f )
{
	fftw<T, fftw_T> data_maker( lengths.size(), &lengths[0], batch, r2c );

	if( pattern == sawtooth )
	{
		data_maker.set_data_to_sawtooth(1.0f);
	}
	else if( pattern == value )
	{
		data_maker.set_all_data_to_value(2.0f);
	}
	else if( pattern == impulse )
	{
		data_maker.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		data_maker.set_data_to_random();
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_real()" );
	}

	data_maker.transform();

	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		input_strides.empty() ? NULL : &input_strides[0],
		output_strides.empty() ? NULL : &output_strides[0],
		batch, input_distance, output_distance,
		cl_layout(in_layout), cl_layout(layout::real),
		cl_placeness(placeness) );
	test_fft.set_input_to_buffer( data_maker.result() );

	fftw<T, fftw_T> reference( lengths.size(), &lengths[0], batch, c2r );
	reference.set_input_to_buffer(data_maker.result());

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	test_fft.backward_scale( scale );
	reference.backward_scale( scale );

	//set postcallback values
	test_fft.set_output_postcallback();

	test_fft.transform();
	reference.transform();

	reference.set_output_postcallback();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

/*****************************************************/
/*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()

// no need to support non-unit strides and distances here
// they are covered in plenty of other places
// and just needlessly complicate things in this case
template< class T, class cl_T, class fftw_T >
void complex_to_complex_round_trip( data_pattern pattern,
									std::vector<size_t> lengths, size_t batch,
									layout::buffer_layout_t layout )
{
	placeness::placeness_t placeness = placeness::in_place;

	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		NULL, NULL,	batch, 0, 0,
		cl_layout(layout), cl_layout(layout),
		cl_placeness( placeness ) );

	buffer<T> expected( lengths.size(), &lengths[0], NULL, batch, 0, layout, CLFFT_OUTOFPLACE );

	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		expected.set_all_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f, 2.5f );
		expected.set_all_to_value( 2.0f, 2.5f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		expected.set_all_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		expected.set_all_to_random_data( 10, super_duper_global_seed );
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_complex_round_trip()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == expected );

	test_fft.set_forward_transform();
	test_fft.transform();

	// confirm that we actually did something
	bool stash_suppress_output = suppress_output;
	suppress_output = true;
	EXPECT_EQ( false, test_fft.result() == expected );
	suppress_output = stash_suppress_output;

	test_fft.set_backward_transform();
	test_fft.transform();

	EXPECT_EQ( true, test_fft.result() == expected );
}

/*****************************************************/
/*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
// no need to support non-unit strides and distances here
// they are covered in plenty of other places
// and just needlessly complicate things in this case
template< class T, class cl_T, class fftw_T >
void precallback_complex_to_complex_round_trip( data_pattern pattern,
									std::vector<size_t> lengths, size_t batch,
									layout::buffer_layout_t layout )
{
	placeness::placeness_t placeness = placeness::in_place;

	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		NULL, NULL,	batch, 0, 0,
		cl_layout(layout), cl_layout(layout),
		cl_placeness( placeness ) );

	buffer<T> expected( lengths.size(), &lengths[0], NULL, batch, 0, layout, CLFFT_OUTOFPLACE );

	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		expected.set_all_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f, 2.5f );
		expected.set_all_to_value( 2.0f, 2.5f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		expected.set_all_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		expected.set_all_to_random_data( 10, super_duper_global_seed );
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_complex_round_trip()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == expected );

	test_fft.set_input_precallback();

	//precallback user data
	buffer<T> userdata( lengths.size(), &lengths[0], NULL, batch, 0, layout::real, CLFFT_OUTOFPLACE);	
	userdata.set_all_to_random_data(lengths[0], 10);
		
	expected *= userdata;

	test_fft.set_forward_transform();
	test_fft.transform();

	// confirm that we actually did something
	bool stash_suppress_output = suppress_output;
	suppress_output = true;
	EXPECT_EQ( false, test_fft.result() == expected );
	suppress_output = stash_suppress_output;

	test_fft.refresh_plan();

	test_fft.set_backward_transform();
	test_fft.transform();

	EXPECT_EQ( true, test_fft.result() == expected );
}

/*****************************************************/
/*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
template< class T, class cl_T, class fftw_T >
void real_to_complex_round_trip( data_pattern pattern,
								 std::vector<size_t> lengths, size_t batch )
{
	placeness::placeness_t placeness = placeness::in_place;

	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		NULL, NULL,	batch, 0, 0,
		cl_layout(layout::real), cl_layout(layout::hermitian_interleaved),
		cl_placeness( placeness ) );

	buffer<T> expected( lengths.size(), &lengths[0], NULL, batch, 0, layout::real, CLFFT_OUTOFPLACE );

	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		expected.set_all_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f );
		expected.set_all_to_value( 2.0f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		expected.set_all_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		expected.set_all_to_random_data( 10, super_duper_global_seed );
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in real_to_complex_round_trip()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == expected );

	test_fft.transform();

	// confirm that we actually did something
	bool stash_suppress_output = suppress_output;
	suppress_output = true;
	EXPECT_EQ( false, test_fft.result() == expected );
	suppress_output = stash_suppress_output;

	test_fft.swap_layouts();
	test_fft.transform();

	EXPECT_EQ( true, test_fft.result() == expected );
}

/*****************************************************/
/*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
template< class T, class cl_T, class fftw_T >
void precallback_real_to_complex_round_trip( data_pattern pattern,
								 std::vector<size_t> lengths, size_t batch )
{
	placeness::placeness_t placeness = placeness::in_place;

	clfft<T, cl_T> test_fft( static_cast<clfftDim>(lengths.size()), &lengths[0],
		NULL, NULL,	batch, 0, 0,
		cl_layout(layout::real), cl_layout(layout::hermitian_interleaved),
		cl_placeness( placeness ) );

	buffer<T> expected( lengths.size(), &lengths[0], NULL, batch, 0, layout::real, CLFFT_OUTOFPLACE );

	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		expected.set_all_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f );
		expected.set_all_to_value( 2.0f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		expected.set_all_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random();
		expected.set_all_to_random_data( 10, super_duper_global_seed );
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in real_to_complex_round_trip()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == expected );

	test_fft.set_input_precallback();

	//precallback user data
	buffer<T> userdata( lengths.size(), &lengths[0], NULL, batch, 0, layout::real, CLFFT_OUTOFPLACE);	
	userdata.set_all_to_random_data(lengths[0], 10);
		
	expected *= userdata;

	test_fft.transform();

	// confirm that we actually did something
	bool stash_suppress_output = suppress_output;
	suppress_output = true;
	EXPECT_EQ( false, test_fft.result() == expected );
	suppress_output = stash_suppress_output;

	test_fft.swap_layouts();
	test_fft.transform();

	EXPECT_EQ( true, test_fft.result() == expected );
}
