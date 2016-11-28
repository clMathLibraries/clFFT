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
#if !defined( CLFFT_CLTRANSFORM_H )
#define CLFFT_CLTRANSFORM_H

#include <iostream>
#include <vector>
#include "clFFT.h"
#include "../library/private.h"
#include "../client/openCL.misc.h"
#include "buffer.h"
#include "test_constants.h"

//	Custom deleter functions for our unique_ptr smart pointer class
struct clMem_deleter
{
	template <class T> void operator()(T* clMemObj)
	{
		if( clMemObj != NULL )
			OPENCL_V_THROW( ::clReleaseMemObject( clMemObj ), "Error: In clReleaseMemObject\n" );
	};
};

struct plan_handle_deleter
{
	template <class T> void operator()(T* handle)
	{
		if( *handle )
		{
			clfftDestroyPlan( handle );
		}
		clfftTeardown( ); // when multi-GPU tests are written, this will need to occur in the gtest cleanup
	};
};

struct clEvent_deleter
{
	template <class T> void operator()(T* clEventObj)
	{
		if( clEventObj != NULL )
			OPENCL_V_THROW( clReleaseEvent( clEventObj ), "Error: In clReleaseEvent\n" );
	};
};

struct clCommQueue_deleter
{
	template <class T> void operator()(T* clQueueObj)
	{
		if( clQueueObj != NULL )
			OPENCL_V_THROW( clReleaseCommandQueue( clQueueObj ), "Error: In clReleaseCommandQueue\n" );
	};
};

struct clContext_deleter
{
	template <class T> void operator()(T* clContextObj)
	{
		if( clContextObj != NULL )
			OPENCL_V_THROW( clReleaseContext( clContextObj ), "Error: In clReleaseContext\n" );
	};
};

template <class T>
class Precision_Setter
{
public:
    Precision_Setter(clfftPlanHandle plan_handle)
    {
        throw std::runtime_error("Precision_Setter: this code path should never be executed");
    }

private:
    Precision_Setter(){}
};

template<>
class Precision_Setter<float>
{
public:
    Precision_Setter(clfftPlanHandle plan_handle)
    {
	    EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanPrecision( plan_handle, CLFFT_SINGLE ));
    }

private:
    Precision_Setter(){}
};

template<>
class Precision_Setter<double>
{
public:
    Precision_Setter(clfftPlanHandle plan_handle)
    {
		clfftStatus ret = clfftSetPlanPrecision( plan_handle, CLFFT_DOUBLE );

		//	If device does not support double precision, skip this test, don't fail it
		if( ret == CLFFT_DEVICE_NO_DOUBLE )
			throw std::runtime_error("CLFFT_DEVICE_NO_DOUBLE");

		EXPECT_EQ( CLFFT_SUCCESS, ret );
    }

private:
    Precision_Setter(){}
};

 /*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
 /*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
template <class T, class cl_T>
class clfft {
private:
	clfftLayout _input_layout, _output_layout;
	clfftResultLocation _placeness;

	buffer<T> input;
	buffer<T> output;
	
	size_t number_of_data_points;
	T _forward_scale, _backward_scale;
	cl_uint commandQueueFlags;
	bool init_failure;
	bool dataset_too_large;

	cl_device_type deviceType;
	std::unique_ptr< clfftPlanHandle, plan_handle_deleter > plan_handle;

	clfftDirection _transformation_direction;
	clfftDim dimension;

	std::vector<size_t> lengths;

	static const bool printInfo = false;

	std::unique_ptr< _cl_mem, clMem_deleter > userDataPreMem;
	std::unique_ptr< _cl_mem, clMem_deleter > userDataPostMem;

	//	OpenCL resources that need to be carefully managed
	std::unique_ptr< _cl_context, clContext_deleter > context;
	std::unique_ptr< _cl_command_queue, clCommQueue_deleter > queue;
	std::vector< std::unique_ptr< _cl_mem, clMem_deleter > > cl_mem_input;
	std::vector< std::unique_ptr< _cl_mem, clMem_deleter > > cl_mem_output;
	std::vector< cl_device_id >	device_id;
public:
	/*****************************************************/
	clfft(  const clfftDim dimensions_in, const size_t* lengths_in,
			const size_t* input_strides_in, const size_t* output_strides_in,
			const size_t batch_size_in,
			const size_t input_distance_in, const size_t output_distance_in,
			const clfftLayout input_layout_in, const clfftLayout output_layout_in,
			const clfftResultLocation placeness_in )
		try
		: _input_layout( input_layout_in )
		, _output_layout( output_layout_in )
		, _placeness( placeness_in )
		, input( 	static_cast<size_t>(dimensions_in),
					lengths_in,
					input_strides_in,
					batch_size_in,
					input_distance_in,
					cl_layout_to_buffer_layout( _input_layout ),
					_placeness
				)
		, output(	static_cast<size_t>(dimensions_in),
					lengths_in,
					output_strides_in,
					batch_size_in,
					output_distance_in,
					cl_layout_to_buffer_layout( _output_layout ),
					_placeness
				)
		, number_of_data_points( input.number_of_data_points())
		, _forward_scale( 1.0f )
		, _backward_scale( 1.0f/T(number_of_data_points) )
		, commandQueueFlags( 0 )
		, init_failure( false )
		, dataset_too_large( false )
		, deviceType( 0 )
		, plan_handle( new clfftPlanHandle )
		, _transformation_direction( ENDDIRECTION )
		, dimension( dimensions_in )

	{
		if( _placeness == CLFFT_INPLACE )
		{
			if( ( is_real( _input_layout ) && is_planar( _output_layout ) ) ||
				( is_planar( _input_layout ) && is_real( _output_layout ) ) )
			{
				throw std::runtime_error( "in-place transforms may not be real<->planar" );
			}
		}

		*plan_handle = 0;
		clfftSetupData setupData;
		clfftInitSetupData( &setupData );
		clfftSetup( &setupData );

		for( int i = 0; i < max_dimension; i++ )
		{
			if( i < dimension )
				lengths.push_back( lengths_in[i] );
			else
				lengths.push_back( 1 );
		}

		initialize_openCL();
		initialize_plan();
	}
	catch( const std::exception& ) {
		throw;
	}

	/*****************************************************/
	~clfft()
	{}

	/*****************************************************/
	bool is_real( const clfftLayout layout )
	{
		return layout == CLFFT_REAL;
	}

	/*****************************************************/
	bool is_planar( const clfftLayout layout )
	{
		return (layout == CLFFT_COMPLEX_PLANAR || layout == CLFFT_HERMITIAN_PLANAR);
	}

	/*****************************************************/
	bool is_interleaved( const clfftLayout layout )
	{
		return (layout == CLFFT_COMPLEX_INTERLEAVED || layout == CLFFT_HERMITIAN_INTERLEAVED);
	}

	/*****************************************************/
	bool is_complex( const clfftLayout layout )
	{
		return (layout == CLFFT_COMPLEX_INTERLEAVED || layout == CLFFT_COMPLEX_PLANAR);
	}

	/*****************************************************/
	bool is_hermitian( const clfftLayout layout )
	{
		return (layout == CLFFT_HERMITIAN_INTERLEAVED || layout == CLFFT_HERMITIAN_PLANAR);
	}

	/*****************************************************/
	void initialize_openCL() {
		try
		{
			cl_context tempContext = NULL;
			device_id = initializeCL(
				g_device_type,
				g_device_id,
				g_platform_id,
				tempContext,
				printInfo
			);
			context = std::unique_ptr< _cl_context, clContext_deleter >( tempContext );

			if( input.size_in_bytes() > cl_device_max_memory_to_allocate(0) ||
				output.size_in_bytes() > cl_device_max_memory_to_allocate(0))
			{
				throw std::runtime_error("problem too large for device");
			}

			cl_int status = 0;
			queue	= std::unique_ptr< _cl_command_queue, clCommQueue_deleter >(
					::clCreateCommandQueue( context.get( ), device_id[ 0 ], commandQueueFlags, &status ) );
			OPENCL_V_THROW( status, "Creating Command Queue ( ::clCreateCommandQueue() )" );

			// make the new buffer
			const size_t bufferSizeBytes = input.size_in_bytes( );

			for( cl_int i = 0; i < CLFFT_COMPLEX_INTERLEAVED; ++i )
			{
				cl_int status = 0;
				std::unique_ptr< _cl_mem, clMem_deleter > inBuff(
						::clCreateBuffer( context.get( ), CL_MEM_READ_WRITE, bufferSizeBytes, NULL, &status) );
				OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer() )" );

				cl_mem_input.push_back( std::move( inBuff ) );

				std::unique_ptr< _cl_mem, clMem_deleter > outBuff(
						::clCreateBuffer( context.get( ), CL_MEM_READ_WRITE, bufferSizeBytes, NULL, &status) );
				OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer() )" );

				cl_mem_output.push_back( std::move( outBuff ) );
			}
		}
		catch( const std::exception& )
		{
			throw;
		}
	}

	/*****************************************************/
	void initialize_plan()
	{
		EXPECT_EQ( CLFFT_SUCCESS, clfftCreateDefaultPlan( plan_handle.get(), context.get( ), dimension, &lengths[0] ) );
		set_layouts( _input_layout, _output_layout );
		placeness( _placeness );
		EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanInStride( *plan_handle, dimension, input.strides()));
		EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanOutStride( *plan_handle, dimension, output.strides()));
		EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanBatchSize( *plan_handle, input.batch_size()));
		EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanDistance( *plan_handle, input.distance(), output.distance()));
		Precision_Setter<T> setter(*plan_handle);
	}

	/*****************************************************/
	std::string input_strides_plaintext()
	{
		size_t strides[3];
		clfftGetPlanInStride( *plan_handle, dimension, &strides[0] );

		std::ostringstream my_strides_stream;

		for( int i = 0; i < dimension; i++ )
			my_strides_stream << strides[i] << " ";

		std::string my_strides( my_strides_stream.str() );
		my_strides.erase( my_strides.end() - 1 ); // chomp off trailing space

		return my_strides;
	}

	/*****************************************************/
	std::string output_strides_plaintext()
	{
		size_t strides[3];
		clfftGetPlanOutStride( *plan_handle, dimension, &strides[0] );

		std::ostringstream my_strides_stream;

		for( int i = 0; i < dimension; i++ )
			my_strides_stream << strides[i] << " ";

		std::string my_strides( my_strides_stream.str() );
		my_strides.erase( my_strides.end() - 1 ); // chomp off trailing space

		return my_strides;
	}

	/*****************************************************/
	std::string lengths_plaintext()
	{
		size_t lengths[3];
		clfftGetPlanLength( *plan_handle, dimension, &lengths[0] );

		std::ostringstream my_lengths_stream;

		for( int i = 0; i < dimension; i++ )
			my_lengths_stream << lengths[i] << " ";

		std::string my_lengths( my_lengths_stream.str() );
		my_lengths.erase( my_lengths.end() - 1 ); // chomp off trailing space

		return my_lengths;
	}

	/*****************************************************/
	std::string layout_plaintext( clfftLayout layout )
	{
		switch( layout )
		{
		case CLFFT_REAL:
			return "real";
		case CLFFT_HERMITIAN_INTERLEAVED:
			return "hermitian interleaved";
		case CLFFT_HERMITIAN_PLANAR:
			return "hermitian planar";
		case CLFFT_COMPLEX_INTERLEAVED:
			return "complex interleaved";
		case CLFFT_COMPLEX_PLANAR:
			return "complex planar";
		default:
			throw std::runtime_error( "invalid layout in layout_plaintext()" );
		}
	}

	/*****************************************************/
	void refresh_plan()
	{
		clfftDestroyPlan(plan_handle.get());
		initialize_plan();
	}

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

	/*****************************************************/
	void verbose_output()
	{
		if(verbose)
		{
			std::cout << "transform parameters as seen by clfft:" << std::endl;

			clfftDim dim;
			cl_uint dimensions;
			clfftGetPlanDim( *plan_handle, &dim, &dimensions );

			std::cout << dimensions << " dimension(s): " << lengths_plaintext() << std::endl;

			size_t batch;
			clfftGetPlanBatchSize( *plan_handle, &batch );
			std::cout << "batch: " << batch << std::endl;

			clfftPrecision precision;
			clfftGetPlanPrecision( *plan_handle, &precision );
			if( precision == CLFFT_SINGLE ) std::cout << "single precision" << std::endl;
			else if( precision == CLFFT_DOUBLE ) std::cout << "double precision" << std::endl;
			else throw std::runtime_error( "can't figure out the precision in verbose_output()" );

			if( placeness() == CLFFT_INPLACE ) std::cout << "in-place" << std::endl;
			else std::cout << "out-of-place" << std::endl;

			get_layouts();
			std::cout << layout_plaintext(_input_layout) << " -> " << layout_plaintext(_output_layout) << std::endl;

			std::cout << "input stride(s): " << input_strides_plaintext() << std::endl;
			std::cout << "output stride(s): " << output_strides_plaintext() << std::endl;

			size_t input_distance, output_distance;
			clfftGetPlanDistance( *plan_handle, &input_distance, &output_distance );
			std::cout << "input distance: " << input_distance << std::endl;
			std::cout << "output distance: " << output_distance << std::endl;
		}
	}

	/*****************************************************/
	clfftResultLocation placeness() {
		clfftResultLocation res;
		EXPECT_EQ( CLFFT_SUCCESS, clfftGetResultLocation( *plan_handle, &res ) );
		return res;
	}

	/*****************************************************/
	void set_forward_transform() {
		_transformation_direction = CLFFT_FORWARD;
	}

	/*****************************************************/
	void set_backward_transform() {
		_transformation_direction = CLFFT_BACKWARD;
	}

	/*****************************************************/
	void set_transposed() {
		EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanTransposeResult( *plan_handle, CLFFT_TRANSPOSED ) );
	}

	/*****************************************************/
	void set_layouts( clfftLayout new_input_layout, clfftLayout new_output_layout )
	{
		cl_mem_input.clear( );
		cl_mem_output.clear( );

		// make the new input buffer
		const size_t input_buffer_size_in_bytes = input.size_in_bytes();

		size_t number_of_input_buffers;

		if( is_planar( new_input_layout ) )
			number_of_input_buffers = 2;
		else if( is_real( new_input_layout ) || is_interleaved( new_input_layout ) )
			number_of_input_buffers = 1;
		else
			throw std::runtime_error( "we shouldn't make it here [set_layouts(), input]" );

		for( size_t i = 0; i < number_of_input_buffers; ++i )
		{
			cl_int status = 0;
			std::unique_ptr< _cl_mem, clMem_deleter > buff(
				::clCreateBuffer( context.get( ), CL_MEM_READ_WRITE, input_buffer_size_in_bytes, NULL, &status) );
			OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer() )" );

			cl_mem_input.push_back( std::move( buff ) );
		}

		// make the new output buffer
		const size_t output_buffer_size_in_bytes = output.size_in_bytes();

		size_t number_of_output_buffers;

		if( is_planar( new_output_layout ) )
			number_of_output_buffers = 2;
		else if( is_real( new_output_layout ) || is_interleaved( new_output_layout ) )
			number_of_output_buffers = 1;
		else
			throw std::runtime_error( "we shouldn't make it here [set_layouts(), input]" );

		for( size_t i = 0; i < number_of_output_buffers; ++i )
		{
			cl_int status = 0;
			std::unique_ptr< _cl_mem, clMem_deleter > buff(
				::clCreateBuffer( context.get( ), CL_MEM_READ_WRITE, output_buffer_size_in_bytes, NULL, &status) );
			OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer() )" );

			cl_mem_output.push_back( std::move( buff ) );
		}

		EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( *plan_handle, new_input_layout, new_output_layout ) );
		get_layouts();
	}

	/*****************************************************/
	// swap_layouts should only be used with in-place real-to-complex or complex-to-real transforms
	void swap_layouts()
	{
		get_layouts();
		clfftLayout new_input_layout = _output_layout;
		clfftLayout new_output_layout = _input_layout;

		EXPECT_EQ( CLFFT_SUCCESS, clfftSetLayout( *plan_handle, new_input_layout, new_output_layout ) );
		get_layouts();

		refresh_plan();
	}

	/*****************************************************/
	clfftLayout input_layout() {
		get_layouts();
		return _input_layout;
	}

	/*****************************************************/
	clfftLayout output_layout() {
		get_layouts();
		return _output_layout;
	}

	/*****************************************************/
	void forward_scale( T in ) {
		EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanScale( *plan_handle, CLFFT_FORWARD, static_cast<float>( in ) ) );
		_forward_scale = forward_scale();
	}

	/*****************************************************/
	void backward_scale( T in ) {
		EXPECT_EQ( CLFFT_SUCCESS, clfftSetPlanScale( *plan_handle, CLFFT_BACKWARD, static_cast<float>( in ) ) );
		_backward_scale = backward_scale();
	}

	/*****************************************************/
	T forward_scale() {
		cl_T scale;
		EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanScale( *plan_handle, CLFFT_FORWARD, reinterpret_cast<cl_float*>(&scale) ));
		return scale;
	}

	/*****************************************************/
	T backward_scale() {
		cl_T scale;
		EXPECT_EQ( CLFFT_SUCCESS, clfftGetPlanScale( *plan_handle, CLFFT_BACKWARD, reinterpret_cast<cl_float*>(&scale) ));
		return scale;
	}

	/*****************************************************/
	void set_input_to_value( T real )
	{
		input.set_all_to_value( real );
	}

	/*****************************************************/
	void set_input_to_value( T real, T imag )
	{
		input.set_all_to_value( real, imag );
	}

	/*****************************************************/
	void set_input_to_sawtooth(T max) {
		input.set_all_to_sawtooth(max);
	}

	/*****************************************************/
	void set_input_to_impulse() {
		input.set_all_to_impulse();
	}

	/*****************************************************/
	// yes, the "super duper global seed" is horrible
	// alas, i'll have TODO it better later
	void set_input_to_random()
	{
		input.set_all_to_random_data( 10, super_duper_global_seed );
	}

	/*****************************************************/
	void set_input_to_buffer( buffer<T> other_buffer ) {
		input = other_buffer;
	}

	/*****************************************************/
	void set_input_precallback(unsigned int localMemSize = 0) {
		cl_int status = 0;
		clfftPrecision precision;
		clfftGetPlanPrecision( *plan_handle, &precision );

		const char* precallbackstr;
		
		if (localMemSize > 0)
		{
			//Test for LDS in precallback function
			precallbackstr = STRINGIFY(PRE_MULVAL_LDS);
		}
		else
		{
			if (input.is_interleaved() )
			{
				precallbackstr = (precision == CLFFT_SINGLE) ? STRINGIFY(PRE_MULVAL) : STRINGIFY(PRE_MULVAL_DP);
			}
			else if (input.is_planar())
			{
				precallbackstr = (precision == CLFFT_SINGLE) ? STRINGIFY(PRE_MULVAL_PLANAR) : STRINGIFY(PRE_MULVAL_PLANAR_DP);
			}
			else if (input.is_real())
			{
				precallbackstr = (precision == CLFFT_SINGLE) ? STRINGIFY(PRE_MULVAL_REAL) : STRINGIFY(PRE_MULVAL_REAL_DP);
			}
		}

		//precallback user data
		buffer<T> userdata( 	static_cast<size_t>(dimension),
					input.lengths(),
					input.strides(),
					input.batch_size(),
					input.distance(),
					layout::real,
					_placeness
					);

		userdata.set_all_to_random_data(lengths[0], 10);
		
		// make the new buffer
		const size_t bufferSizeBytes = userdata.size_in_bytes( );

		std::unique_ptr< _cl_mem, clMem_deleter > userdataBuff( clCreateBuffer( context.get( ), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
													bufferSizeBytes, userdata.real_ptr(), &status) );
		OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer() )" );

		userDataPreMem = std::move(userdataBuff);
		cl_mem uptr = userDataPreMem.get();

		//Register the callback
		OPENCL_V_THROW (clfftSetPlanCallback(*plan_handle, "mulval_pre", precallbackstr, localMemSize, PRECALLBACK, &uptr, 1), "clFFTSetPlanCallback failed");
	}

		/*****************************************************/
	void set_input_precallback_userdatatype() {
		cl_int status = 0;

		const char* precallbackstr = STRINGIFY(PRE_MULVAL_UDT);

		size_t totalPts = input.total_number_of_points_including_data_and_intervening();

		buffer<T> temp( 	static_cast<size_t>(dimension),
					input.lengths(),
					input.strides(),
					input.batch_size(),
					input.distance(),
					layout::real,
					_placeness
					);

		temp.set_all_to_random_data(lengths[0], 10);

		std::vector<USER_DATA> userdata(totalPts);
		size_t the_index;
		for( size_t batch = 0; batch < input.batch_size(); batch++) 
			for( size_t z = 0; z < input.length(dimz); z++) 
				for( size_t y = 0; y < input.length(dimy); y++) 
					for( size_t x = 0; x < input.length(dimx); x++) 
					{
						the_index = ( input.stride(dimx) * x + input.stride(dimy) * y + input.stride(dimz) * z + input.distance() * batch );

						userdata[the_index].scalar1 = (float)temp.real(x, y, z, batch);
						userdata[the_index].scalar2 = 1;
					}

		std::unique_ptr< _cl_mem, clMem_deleter > userdataBuff( clCreateBuffer(context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
													sizeof(USER_DATA) * totalPts, (void*)&userdata[0], &status) );
		OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer() )" );

		userDataPreMem = std::move(userdataBuff);
		cl_mem uptr = userDataPreMem.get();

		//Register the callback
		OPENCL_V_THROW (clfftSetPlanCallback(*plan_handle, "mulval_pre", precallbackstr, 0, PRECALLBACK, &uptr, 1), "clFFTSetPlanCallback failed");
	}

		/*****************************************************/
	void set_output_postcallback(unsigned int localMemSize = 0) {
		cl_int status = 0;
		clfftPrecision precision;
		clfftGetPlanPrecision( *plan_handle, &precision );

		const char* postcallbackstr;
		
		if (localMemSize > 0)
		{
			//Test for LDS in postcallback function
			postcallbackstr = STRINGIFY(POST_MULVAL_LDS);
		}
		else
		{
			if (output.is_interleaved() )
			{
				postcallbackstr = (precision == CLFFT_SINGLE) ? STRINGIFY(POST_MULVAL) : STRINGIFY(POST_MULVAL_DP);
			}
			else if (output.is_planar())
			{
				postcallbackstr = (precision == CLFFT_SINGLE) ? STRINGIFY(POST_MULVAL_PLANAR) : STRINGIFY(POST_MULVAL_PLANAR_DP);
			}
			else if (output.is_real())
			{
				postcallbackstr = (precision == CLFFT_SINGLE) ? STRINGIFY(POST_MULVAL_REAL) : STRINGIFY(POST_MULVAL_REAL_DP);
			}
		}

		//post-callback user data
		buffer<T> userdata( 	static_cast<size_t>(dimension),
					output.lengths(),
					output.strides(),
					output.batch_size(),
					output.distance(),
					layout::real,
					_placeness
					);

		userdata.set_all_to_random_data(lengths[0], 10);
		
		// make the new buffer
		const size_t bufferSizeBytes = userdata.size_in_bytes( );

		std::unique_ptr< _cl_mem, clMem_deleter > userdataBuff(clCreateBuffer(context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			bufferSizeBytes, userdata.real_ptr(), &status));
		OPENCL_V_THROW(status, "Creating Buffer ( ::clCreateBuffer() )");

		userDataPostMem = std::move(userdataBuff);
		cl_mem uptr = userDataPostMem.get();

		//Register the post-callback
		OPENCL_V_THROW (clfftSetPlanCallback(*plan_handle, "mulval_post", postcallbackstr, localMemSize, POSTCALLBACK, &uptr, 1), "clFFTSetPlanCallback failed");
	}

	/*****************************************************/
	bool device_list_has_devices() {
		return !device_id.empty();
	}

	/*****************************************************/
	// returns true if the memory required for input + output (if applicable) + intermediate (if applicable) buffers
	// is too large compared with the OpenCL device's memory size
	bool total_memory_footprint_is_too_large_for_device() {
		throw_if_device_list_is_empty();

		// In order to call clfftEnqueueTransform, we need to pass naked pointers
		cl_command_queue tempQueue = queue.get( );
		size_t buffer_size = 0;

		EXPECT_EQ( CLFFT_SUCCESS, clfftBakePlan(*plan_handle, 1, &tempQueue, NULL, NULL ));
		EXPECT_EQ( CLFFT_SUCCESS, clfftGetTmpBufSize(*plan_handle, &buffer_size ));

		cl_ulong total_memory_size = input.size_in_bytes() + buffer_size;

		// we are only going to include the result space if the transform is out of place
		if( placeness() == CLFFT_OUTOFPLACE )
		{
			total_memory_size += output.size_in_bytes();
		}

		cl_ulong global_memory_size = cl_device_max_global_memory(0);

		// we don't want to bog down the CPU with ginormous problem sizes
		// so we chop the global memory way down to keep things manageable
		if( g_device_type == CL_DEVICE_TYPE_CPU )
		{
			global_memory_size /= 8;
		}

		return total_memory_size > global_memory_size;
	}

	/*****************************************************/
	void throw_if_total_memory_footprint_is_too_large_for_device()
	{
		if( total_memory_footprint_is_too_large_for_device() )
		{
			throw std::runtime_error("problem too large for device");
		}
	}

	/*****************************************************/
	void throw_if_device_list_is_empty()
	{
		if( !device_list_has_devices() ) {
			throw std::runtime_error("device list is empty at transform");
		}
	}

	/*****************************************************/
	void transform(bool explicit_intermediate_buffer = use_explicit_intermediate_buffer) {
		verbose_output();

		throw_if_device_list_is_empty();

		cl_int status;

		// In order to call clfftEnqueueTransform, we need to pass naked pointers
		cl_command_queue tempQueue = queue.get( );
		std::unique_ptr< _cl_event, clEvent_deleter > tempEvent;
		std::unique_ptr< _cl_mem, clMem_deleter > intermediate_buffer;

		throw_if_total_memory_footprint_is_too_large_for_device();

		write_local_input_buffer_to_gpu();
		if( placeness() == CLFFT_OUTOFPLACE )
			write_local_output_buffer_to_gpu();

		try
		{
			size_t buffer_size = 0;
			EXPECT_EQ( CLFFT_SUCCESS, clfftBakePlan(*plan_handle, 1, &tempQueue, NULL, NULL ));
			EXPECT_EQ( CLFFT_SUCCESS, clfftGetTmpBufSize(*plan_handle, &buffer_size ));

			if( explicit_intermediate_buffer )
			{
				// the buffer size is already stashed above
				// now we want to make the intermediate buffer to pass in (if necessary)
				if (buffer_size)
				{
					// because unique_ptrs are funky, we have to create a temp_buffer
					// and then std::move it to the intermediate_buffer
					std::unique_ptr< _cl_mem, clMem_deleter > temp_buffer(
						::clCreateBuffer( context.get( ),
								  CL_MEM_READ_WRITE,
								  buffer_size,
								  NULL,
								  &status) );
					OPENCL_V_THROW( status, "Creating intermediate Buffer ( ::clCreateBuffer() )" );

					intermediate_buffer = std::move( temp_buffer );
				}
			}

			cl_mem	tempInput[2];
			cl_mem	tempOutput[2];
			for( cl_uint i = 0; i < cl_mem_input.size( ); ++i )
				tempInput[ i ] = cl_mem_input[ i ].get( );

			for( cl_uint i = 0; i < cl_mem_output.size( ); ++i )
				tempOutput[ i ] = cl_mem_output[ i ].get( );

			cl_event tevent = NULL;
			if( buffer_size )
			{
				status = clfftEnqueueTransform(*plan_handle,
								  _transformation_direction,
								  1,
								  &tempQueue,
								  0,
								  NULL,
								  &tevent,
								  &tempInput[ 0 ],
								  &tempOutput[ 0 ],
								  intermediate_buffer.get() );
			}
			else
			{
				status = clfftEnqueueTransform(*plan_handle,
								  _transformation_direction,
								  1,
								  &tempQueue,
								  0,
								  NULL,
								  &tevent,
								  &tempInput[ 0 ],
								  &tempOutput[ 0 ],
								  NULL );
			}
            clFinish(tempQueue);
			tempEvent.reset(tevent); tevent = NULL;

			if( status != CLFFT_SUCCESS )
			{
				throw std::runtime_error(prettyPrintclFFTStatus(status).c_str());
			}

			// wait for the kernel call to finish execution
			const cl_event revent = tempEvent.get();
            cl_int wait_status = clWaitForEvents(1, &revent);
			if( wait_status == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST )
			{
				cl_int error_code;
				clGetEventInfo( revent, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &error_code, NULL );
				throw std::runtime_error(prettyPrintclFFTStatus(error_code).c_str());
			}
            else if( wait_status != CL_SUCCESS )
            {
				throw std::runtime_error(prettyPrintclFFTStatus(wait_status).c_str());
            }
		}
		catch (const std::exception& ) {
			std::cout << "Exception occurred during clfftEnqueueTransform"
					  << __FILE__ << __LINE__ << std::endl;
			throw;
		}

		if( in_place() ) {
			capture_input();
		}
		else {
			capture_output();
		}

		get_layouts();
		if( placeness() == CLFFT_INPLACE )
		{
			if( is_real( _input_layout ) && is_hermitian( _output_layout ) )
			{
				input.change_real_to_hermitian( output.strides(), output.distance() );
			}
			else if( is_hermitian( _input_layout ) && is_real( _output_layout ) )
			{
				input.change_hermitian_to_real( output.strides(), output.distance() );
			}
		}

		// there's no way to know if in-place transforms have written in bad places,
		// because depending on input and output strides, the state of the memory
		// between points is not necessarily the NaN that we set it to
		if( _placeness != CLFFT_INPLACE )
		{
			input.make_sure_padding_was_not_overwritten();
			output.make_sure_padding_was_not_overwritten();
		}
	}

	/*****************************************************/
	size_t maximum_problem_size() {
		int device_index = 0;
		//N.B. if this class ever needs to support more than one device at once
		//(i.e., multiple GPUs or CPU+GPU), device index will need to be variable
		//to choose the device of interest
		return cl_device_max_memory_to_allocate(device_index)/(sizeof(T)*2);
		//TODO *2 needs to be either *1 or *2, depending, once real numbers are implemented in clfft
	}

	/*****************************************************/
	size_t number_of_opencl_devices() {
		return device_id.size();
	}


	/*****************************************************/
	bool initialize_failed() {
		return init_failure;
	}

	/*****************************************************/
	bool dataset_is_too_large_for_device() {
		return dataset_too_large;
	}

	/*****************************************************/
	buffer<T> & input_buffer()
	{
		return input;
	}

	/*****************************************************/
	buffer<T> & output_buffer()
	{
		return output;
	}

	/*****************************************************/
	buffer<T> & result()
	{
		if( placeness() == CLFFT_INPLACE )
			return input;
		else if( placeness() == CLFFT_OUTOFPLACE )
			return output;
		else
			throw std::runtime_error( "invalid placeness" );
	}

private:
	/*****************************************************/
	void get_layouts() {
		EXPECT_EQ( CLFFT_SUCCESS, clfftGetLayout( *plan_handle, &_input_layout, &_output_layout ) );
	}

	/*****************************************************/
	// after transform() is run:
	//   if in-place transformation -- the results will be in the input buffer
	//	 otherwise -- the results will be in the output buffer
	void placeness( clfftResultLocation placeness )
	{
		EXPECT_EQ( CLFFT_SUCCESS, clfftSetResultLocation( *plan_handle, placeness ) );
	}

	/*****************************************************/
	bool in_place() {
		clfftResultLocation placeness;
		clfftGetResultLocation( *plan_handle, &placeness );
		return (placeness == CLFFT_INPLACE) ? true : false;
	}

	/*****************************************************/
	void capture_output() {
		if( is_planar( output_layout() ) ) {
			OPENCL_V_THROW( clEnqueueReadBuffer( queue.get( ), cl_mem_output[REAL].get( ), CL_TRUE, 0,
					output.size_in_bytes(), output.real_ptr(), 0, NULL, NULL), "reading output buffer - planar real ( ::clEnqueueReadBuffer() )" );
			OPENCL_V_THROW( clEnqueueReadBuffer( queue.get( ), cl_mem_output[IMAG].get( ), CL_TRUE, 0,
					output.size_in_bytes(), output.imag_ptr(), 0, NULL, NULL), "reading output buffer - planar imaginary ( ::clEnqueueReadBuffer() )" );
		}
		else if( is_interleaved( output_layout() ) ) {
			OPENCL_V_THROW( clEnqueueReadBuffer( queue.get( ), cl_mem_output[0].get( ), CL_TRUE, 0,
					output.size_in_bytes(), output.interleaved_ptr(), 0, NULL, NULL), "reading output buffer - interleaved ( ::clEnqueueReadBuffer() )" );
		}
		else if( is_real( output_layout() ) ) {
			OPENCL_V_THROW( clEnqueueReadBuffer( queue.get( ), cl_mem_output[REAL].get( ), CL_TRUE, 0,
					output.size_in_bytes(), output.real_ptr(), 0, NULL, NULL), "reading output buffer - planar real ( ::clEnqueueReadBuffer() )" );
		}
		else
		{
			throw std::runtime_error( "we shouldn't make it here [capture_output()]" );
		}
	}

	/*****************************************************/
	void capture_input() {
		if( is_planar( input_layout() ) ) {
			OPENCL_V_THROW( clEnqueueReadBuffer( queue.get( ), cl_mem_input[REAL].get( ), CL_TRUE, 0,
					input.size_in_bytes(), input.real_ptr(), 0, NULL, NULL), "reading input buffer - planar real ( ::clEnqueueReadBuffer() )" );
			OPENCL_V_THROW( clEnqueueReadBuffer( queue.get( ), cl_mem_input[IMAG].get( ), CL_TRUE, 0,
					input.size_in_bytes(), input.imag_ptr(), 0, NULL, NULL), "reading input buffer - planar imaginary ( ::clEnqueueReadBuffer() )" );
		}
		else if( is_interleaved ( input_layout() ) ) {
			OPENCL_V_THROW( clEnqueueReadBuffer( queue.get( ), cl_mem_input[0].get( ), CL_TRUE, 0,
					input.size_in_bytes(), input.interleaved_ptr(), 0, NULL, NULL), "reading input buffer - interleaved ( ::clEnqueueReadBuffer() )" );
		}
		else if( is_real( input_layout() ) ) {
			OPENCL_V_THROW( clEnqueueReadBuffer( queue.get( ), cl_mem_input[REAL].get( ), CL_TRUE, 0,
					input.size_in_bytes(), input.real_ptr(), 0, NULL, NULL), "reading input buffer - planar real ( ::clEnqueueReadBuffer() )" );
		}
		else
		{
			throw std::runtime_error( "we shouldn't make it here [capture_input()]" );
		}
	}

	/*****************************************************/
	void write_local_output_buffer_to_gpu() {
		if( is_planar( output_layout() ) ) {
			OPENCL_V_THROW( clEnqueueWriteBuffer( queue.get( ), cl_mem_output[REAL].get( ), CL_TRUE, 0,
					output.size_in_bytes(), output.real_ptr(), 0, NULL, NULL), "writing output buffer - planar real ( ::clEnqueueWriteBuffer() )" );
			OPENCL_V_THROW( clEnqueueWriteBuffer( queue.get( ), cl_mem_output[IMAG].get( ), CL_TRUE, 0,
					output.size_in_bytes(), output.imag_ptr(), 0, NULL, NULL), "writing output buffer - planar imaginary ( ::clEnqueueWriteBuffer() )" );
		}
		else if( is_interleaved ( output_layout() ) ) {
			OPENCL_V_THROW( clEnqueueWriteBuffer( queue.get( ), cl_mem_output[0].get( ), CL_TRUE, 0,
					output.size_in_bytes(), output.interleaved_ptr(), 0, NULL, NULL), "writing output buffer - interleaved ( ::clEnqueueWriteBuffer() )" );
		}
		else if( is_real( output_layout() ) ) {
			OPENCL_V_THROW( clEnqueueWriteBuffer( queue.get( ), cl_mem_output[REAL].get( ), CL_TRUE, 0,
					output.size_in_bytes(), output.real_ptr(), 0, NULL, NULL), "writing output buffer - planar real ( ::clEnqueueWriteBuffer() )" );
		}
		else
		{
			throw std::runtime_error( "we shouldn't make it here [write_local_output_buffer_to_gpu()]" );
		}
	}

	/*****************************************************/
	void write_local_input_buffer_to_gpu() {
		if( is_planar( input_layout() ) ) {
			OPENCL_V_THROW( clEnqueueWriteBuffer( queue.get( ), cl_mem_input[REAL].get( ), CL_TRUE, 0,
					input.size_in_bytes(), input.real_ptr(), 0, NULL, NULL), "writing input buffer - planar real ( ::clEnqueueWriteBuffer() )" );
			OPENCL_V_THROW( clEnqueueWriteBuffer( queue.get( ), cl_mem_input[IMAG].get( ), CL_TRUE, 0,
					input.size_in_bytes(), input.imag_ptr(), 0, NULL, NULL), "writing input buffer - planar imaginary ( ::clEnqueueWriteBuffer() )" );
		}
		else if( is_interleaved( input_layout() ) ) {
			OPENCL_V_THROW( clEnqueueWriteBuffer( queue.get( ), cl_mem_input[0].get( ), CL_TRUE, 0,
					input.size_in_bytes(), input.interleaved_ptr(), 0, NULL, NULL), "writing input buffer - interleaved ( ::clEnqueueWriteBuffer() )" );
		}
		else if( is_real( input_layout() ) ) {
			OPENCL_V_THROW( clEnqueueWriteBuffer( queue.get( ), cl_mem_input[REAL].get( ), CL_TRUE, 0,
					input.size_in_bytes(), input.real_ptr(), 0, NULL, NULL), "writing input buffer - planar real ( ::clEnqueueWriteBuffer() )" );
		}
		else
		{
			throw std::runtime_error( "we shouldn't make it here [write_local_input_buffer_to_gpu()]" );
		}
	}


	/*****************************************************/
	cl_ulong cl_device_max_memory_to_allocate(size_t device_index) {
		if( number_of_opencl_devices() == 0 || device_index > number_of_opencl_devices() )
		{
			return 0;
		}
		else
		{
			cl_ulong device_max_to_allocate = 0;
			OPENCL_V_THROW( ::clGetDeviceInfo( device_id[device_index], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( cl_ulong ), &device_max_to_allocate, NULL ),
				"Getting CL_DEVICE_MAX_MEM_ALLOC_SIZE device info ( ::clGetDeviceInfo() )" );

			return device_max_to_allocate;
		}
	}


	/*****************************************************/
	cl_ulong cl_device_max_global_memory(size_t device_index) {
		if( number_of_opencl_devices() == 0 || device_index > number_of_opencl_devices() )
		{
			return 0;
		}
		else
		{
			cl_ulong global_mem_size = 0;
			OPENCL_V_THROW( ::clGetDeviceInfo( device_id[device_index], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( cl_ulong ), &global_mem_size, NULL ),
				"Getting CL_DEVICE_GLOBAL_MEM_SIZE device info ( ::clGetDeviceInfo() )" );

			return global_mem_size;
		}
	}

	#if defined(PERSISTENT_PLANS_FEATURE_HAS_BEEN_DEFEATURED_WHICH_MEANS_IT_IS_NO_LONGER_A_FEATURE)
	/*****************************************************/
	void write_plan_to_file(std::string filename)
	{
		cl_command_queue tempQueue = queue.get( );
		EXPECT_EQ( CLFFT_SUCCESS, clfftBakePlan(*plan_handle, 1, &tempQueue, NULL, NULL ));
		// we need to make sure the plan is baked before we write it out, or we won't get any juicy binaries along with it

		clfftWritePlanToDisk(*plan_handle, filename.c_str());
	}

	/*****************************************************/
	void read_plan_from_file(std::string filename)
	{
		clfftReadPlanFromDisk( *plan_handle, filename.c_str() );

		// if we've changed from the default for input and output layouts, we need to re-set the layouts to make sure buffers get set up completely
		set_layouts( input_layout(), output_layout() );
	}
	#endif
};

#endif
