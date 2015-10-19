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


// clfft.client.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <functional>
#include <cmath>

#include "client.h"
#include "../library/private.h"
#include "openCL.misc.h"
#include "../statTimer/statisticalTimer.extern.h"
#include "../include/sharedLibrary.h"
#include "../include/unicode.compatibility.h"

namespace po = boost::program_options;

//	This is used with the program_options class so that the user can type an integer on the command line
//	and we store into an enum varaible
template<class _Elem, class _Traits>
std::basic_istream<_Elem, _Traits> & operator>> (std::basic_istream<_Elem, _Traits> & stream, clfftLayout & layout)
{
	cl_uint tmp;
	stream >> tmp;
	layout = clfftLayout(tmp);
	return stream;
}


template < typename T >
int transform( size_t* lengths, const size_t *inStrides, const size_t *outStrides, size_t batch_size,
				clfftLayout in_layout, clfftLayout out_layout,
				clfftResultLocation place, clfftPrecision precision, clfftDirection dir,
				cl_device_type deviceType, cl_int deviceId, cl_int platformId, bool printInfo,
				cl_uint command_queue_flags, cl_uint profile_count,
				std::auto_ptr< clfftSetupData > setupData )
{
	//	Our command line does not specify what dimension FFT we wish to transform; we decode
	//	this from the lengths that the user specifies for X, Y, Z.  A length of one means that
	//	The user does not want that dimension.

	const size_t max_dimensions = 3;
	size_t strides[ 4 ];
	size_t o_strides[ 4 ];
	size_t fftVectorSize = 0;
	size_t fftVectorSizePadded = 0;
	size_t fftBatchSize = 0;
	size_t outfftVectorSize = 0;
	size_t outfftVectorSizePadded = 0;
	size_t outfftBatchSize = 0;
	size_t size_of_input_buffers_in_bytes = 0;
	size_t size_of_output_buffers_in_bytes = 0;
	cl_uint number_of_output_buffers = 0;
	clfftDim	dim = CLFFT_1D;
	cl_mem input_cl_mem_buffers [2] = { NULL, NULL };
	cl_mem output_cl_mem_buffers[2] = { NULL, NULL };
	std::vector< cl_device_id > device_id;
	cl_context context;
	cl_command_queue queue;
	clfftPlanHandle plan_handle;

	for (unsigned u = 0; u < max_dimensions; ++u) {
		if (0 != lengths[u])
			continue;
		lengths[u] = 1;
	}

	if( lengths[ 1 ] > 1 )
	{
		dim	= CLFFT_2D;
	}
	if( lengths[ 2 ] > 1 )
	{
		dim	= CLFFT_3D;
	}

	strides[ 0 ] = inStrides[0];
	strides[ 1 ] = inStrides[1];
	strides[ 2 ] = inStrides[2];
	strides[ 3 ] = inStrides[3];

	o_strides[ 0 ] = outStrides[0];
	o_strides[ 1 ] = outStrides[1];
	o_strides[ 2 ] = outStrides[2];
	o_strides[ 3 ] = outStrides[3];

	fftVectorSize = lengths[0] * lengths[1] * lengths[2];
	fftVectorSizePadded = strides[3];
	fftBatchSize = fftVectorSizePadded * batch_size;

	size_t Nt = 1 + lengths[0]/2;

	if(place == CLFFT_INPLACE)
	{
		outfftVectorSize = fftVectorSize;
		outfftVectorSizePadded = fftVectorSizePadded;
		outfftBatchSize = fftBatchSize;
	}
	else
	{
		outfftVectorSize = lengths[0] * lengths[1] * lengths[2];
		outfftVectorSizePadded = o_strides[3];
		outfftBatchSize = outfftVectorSizePadded * batch_size;
	}


	// Real to complex case
	if( (in_layout == CLFFT_REAL) || (out_layout == CLFFT_REAL) )
	{
		fftVectorSizePadded = strides[3];
		fftBatchSize = fftVectorSizePadded * batch_size;

		outfftVectorSizePadded = o_strides[3];
		outfftBatchSize = outfftVectorSizePadded * batch_size;

		fftVectorSize = lengths[0] * lengths[1] * lengths[2];
		outfftVectorSize = fftVectorSize;

	}


	switch( out_layout )
	{
	case CLFFT_COMPLEX_INTERLEAVED:
		number_of_output_buffers = 1;
		size_of_output_buffers_in_bytes = outfftBatchSize * sizeof( std::complex< T > );
		break;
	case CLFFT_COMPLEX_PLANAR:
		number_of_output_buffers = 2;
		size_of_output_buffers_in_bytes = outfftBatchSize * sizeof(T);
		break;
	case CLFFT_HERMITIAN_INTERLEAVED:
		number_of_output_buffers = 1;
		size_of_output_buffers_in_bytes = outfftBatchSize * sizeof( std::complex< T > );
		break;
	case CLFFT_HERMITIAN_PLANAR:
		number_of_output_buffers = 2;
		size_of_output_buffers_in_bytes = outfftBatchSize * sizeof(T);
		break;
	case CLFFT_REAL:
		number_of_output_buffers = 1;
		size_of_output_buffers_in_bytes = outfftBatchSize * sizeof(T);
		break;
	}

	// Fill the input buffers
	switch( in_layout )
	{
	case CLFFT_COMPLEX_INTERLEAVED:
		{
			//	This call creates our openCL context and sets up our devices; expected to throw on error
			size_of_input_buffers_in_bytes = fftBatchSize * sizeof( std::complex< T > );

			device_id = initializeCL( deviceType, deviceId, platformId, context, printInfo );
			createOpenCLCommandQueue( context,
				command_queue_flags, queue,
				device_id,
				size_of_input_buffers_in_bytes, 1, input_cl_mem_buffers,
				size_of_output_buffers_in_bytes, number_of_output_buffers, output_cl_mem_buffers);

			std::vector< std::complex< T > > input( fftBatchSize );

			// set zero
			for( cl_uint i = 0; i < fftBatchSize; ++i )
			{
				input[ i ] = 0;
			}

			// impulse test case
			for(size_t b = 0; b < batch_size; b++)
			{
				size_t p3 = b * strides[3];
				for(size_t k = 0; k < lengths[2]; k++)
				{
					size_t p2 = p3 + k * strides[2];
					for(size_t j = 0; j < lengths[1]; j++)
					{
						size_t p1 = p2 + j * strides[1];
						for(size_t i = 0; i < lengths[0]; i++)
						{
							size_t p0 = p1 + i * strides[0];
							input[p0] = 1;
						}
					}
				}
			}


			OPENCL_V_THROW( clEnqueueWriteBuffer( queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, size_of_input_buffers_in_bytes, &input[ 0 ],
				0, NULL, NULL ),
				"clEnqueueWriteBuffer failed" );

		}
		break;
	case CLFFT_COMPLEX_PLANAR:
		{
			//	This call creates our openCL context and sets up our devices; expected to throw on error
			size_of_input_buffers_in_bytes = fftBatchSize * sizeof( T );

			device_id = initializeCL( deviceType, deviceId, platformId, context, printInfo );
			createOpenCLCommandQueue( context,
				command_queue_flags, queue,
				device_id,
				size_of_input_buffers_in_bytes, 2, input_cl_mem_buffers,
				size_of_output_buffers_in_bytes, number_of_output_buffers, output_cl_mem_buffers);

			std::vector< T > real( fftBatchSize );
			std::vector< T > imag( fftBatchSize );

			// set zero
			for( cl_uint i = 0; i < fftBatchSize; ++i )
			{
				real[ i ] = 0;
				imag[ i ] = 0;
			}

			// impulse test case
			for(size_t b = 0; b < batch_size; b++)
			{
				size_t p3 = b * strides[3];
				for(size_t k = 0; k < lengths[2]; k++)
				{
					size_t p2 = p3 + k * strides[2];
					for(size_t j = 0; j < lengths[1]; j++)
					{
						size_t p1 = p2 + j * strides[1];
						for(size_t i = 0; i < lengths[0]; i++)
						{
							size_t p0 = p1 + i * strides[0];
							real[p0] = 1;
						}
					}
				}
			}


			OPENCL_V_THROW( clEnqueueWriteBuffer( queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, size_of_input_buffers_in_bytes, &real[ 0 ],
				0, NULL, NULL ),
				"clEnqueueWriteBuffer failed" );
			OPENCL_V_THROW( clEnqueueWriteBuffer( queue, input_cl_mem_buffers[ 1 ], CL_TRUE, 0, size_of_input_buffers_in_bytes, &imag[ 0 ],
				0, NULL, NULL ),
				"clEnqueueWriteBuffer failed" );
		}
		break;
	case CLFFT_HERMITIAN_INTERLEAVED:
		{
			//	This call creates our openCL context and sets up our devices; expected to throw on error
			size_of_input_buffers_in_bytes = fftBatchSize * sizeof( std::complex< T > );

			device_id = initializeCL( deviceType, deviceId, platformId, context, printInfo );
			createOpenCLCommandQueue( context,
				command_queue_flags, queue,
				device_id,
				size_of_input_buffers_in_bytes, 1, input_cl_mem_buffers,
				size_of_output_buffers_in_bytes, number_of_output_buffers, output_cl_mem_buffers);

			std::vector< std::complex< T > > input( fftBatchSize );

			// set zero
			for( cl_uint i = 0; i < fftBatchSize; ++i )
			{
				input[ i ] = 0;
			}

			// impulse test case
			for(size_t b = 0; b < batch_size; b++)
			{
				size_t p3 = b * strides[3];
				input[p3] = static_cast<T>(outfftVectorSize);

			}


			OPENCL_V_THROW( clEnqueueWriteBuffer( queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, size_of_input_buffers_in_bytes, &input[ 0 ],
				0, NULL, NULL ),
				"clEnqueueWriteBuffer failed" );
		}
		break;
	case CLFFT_HERMITIAN_PLANAR:
		{
			//	This call creates our openCL context and sets up our devices; expected to throw on error
			size_of_input_buffers_in_bytes = fftBatchSize * sizeof( T );

			device_id = initializeCL( deviceType, deviceId, platformId, context, printInfo );
			createOpenCLCommandQueue( context,
				command_queue_flags, queue,
				device_id,
				size_of_input_buffers_in_bytes, 2, input_cl_mem_buffers,
				size_of_output_buffers_in_bytes, number_of_output_buffers, output_cl_mem_buffers);

			std::vector< T > real( fftBatchSize );
			std::vector< T > imag( fftBatchSize );

			// set zero
			for( cl_uint i = 0; i < fftBatchSize; ++i )
			{
				real[ i ] = 0;
				imag[ i ] = 0;
			}

			// impulse test case
			for(size_t b = 0; b < batch_size; b++)
			{
				size_t p3 = b * strides[3];
				real[p3] = static_cast<T>(outfftVectorSize);
			}



			OPENCL_V_THROW( clEnqueueWriteBuffer( queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, size_of_input_buffers_in_bytes, &real[ 0 ],
				0, NULL, NULL ),
				"clEnqueueWriteBuffer failed" );
			OPENCL_V_THROW( clEnqueueWriteBuffer( queue, input_cl_mem_buffers[ 1 ], CL_TRUE, 0, size_of_input_buffers_in_bytes, &imag[ 0 ],
				0, NULL, NULL ),
				"clEnqueueWriteBuffer failed" );
		}
		break;
	case CLFFT_REAL:
		{
			//	This call creates our openCL context and sets up our devices; expected to throw on error
			size_of_input_buffers_in_bytes = fftBatchSize * sizeof( T );

			device_id = initializeCL( deviceType, deviceId, platformId, context, printInfo );
			createOpenCLCommandQueue( context,
				command_queue_flags, queue,
				device_id,
				size_of_input_buffers_in_bytes, 1, input_cl_mem_buffers,
				size_of_output_buffers_in_bytes, number_of_output_buffers, output_cl_mem_buffers);

			std::vector< T > real( fftBatchSize );

			// set zero
			for( cl_uint i = 0; i < fftBatchSize; ++i )
			{
				real[ i ] = 0;
			}

			// impulse test case
			for(size_t b = 0; b < batch_size; b++)
			{
				size_t p3 = b * strides[3];
				for(size_t k = 0; k < lengths[2]; k++)
				{
					size_t p2 = p3 + k * strides[2];
					for(size_t j = 0; j < lengths[1]; j++)
					{
						size_t p1 = p2 + j * strides[1];
						for(size_t i = 0; i < lengths[0]; i++)
						{
							size_t p0 = p1 + i * strides[0];
							real[p0] = 1;
						}
					}
				}
			}


			OPENCL_V_THROW( clEnqueueWriteBuffer( queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, size_of_input_buffers_in_bytes, &real[ 0 ],
				0, NULL, NULL ),
				"clEnqueueWriteBuffer failed" );
		}
		break;
	default:
		{
			throw std::runtime_error( "Input layout format not yet supported" );
		}
		break;
	}

	//	Discover and load the timer module if present
	void* timerLibHandle = LoadSharedLibrary( "lib", "StatTimer", false );
	if( timerLibHandle == NULL )
	{
		terr << _T( "Could not find the external timing library; timings disabled" ) << std::endl;
	}

	//	Timer module discovered and loaded successfully
	//	Initialize function pointers to call into the shared module
	PFGETSTATTIMER get_timer = reinterpret_cast< PFGETSTATTIMER > ( LoadFunctionAddr( timerLibHandle, "getStatTimer" ) );


	OPENCL_V_THROW( clfftSetup( setupData.get( ) ), "clfftSetup failed" );
	OPENCL_V_THROW( clfftCreateDefaultPlan( &plan_handle, context, dim, lengths ), "clfftCreateDefaultPlan failed" );

	//	Default plan creates a plan that expects an inPlace transform with interleaved complex numbers
	OPENCL_V_THROW( clfftSetResultLocation( plan_handle, place ), "clfftSetResultLocation failed" );
	OPENCL_V_THROW( clfftSetLayout( plan_handle, in_layout, out_layout ), "clfftSetLayout failed" );
	OPENCL_V_THROW( clfftSetPlanBatchSize( plan_handle, batch_size ), "clfftSetPlanBatchSize failed" );
	OPENCL_V_THROW( clfftSetPlanPrecision( plan_handle, precision ), "clfftSetPlanPrecision failed" );

	OPENCL_V_THROW (clfftSetPlanInStride  ( plan_handle, dim, strides ), "clfftSetPlanInStride failed" );
	OPENCL_V_THROW (clfftSetPlanOutStride ( plan_handle, dim, o_strides ), "clfftSetPlanOutStride failed" );
	OPENCL_V_THROW (clfftSetPlanDistance  ( plan_handle, strides[ 3 ], o_strides[ 3 ]), "clfftSetPlanDistance failed" );

	// Set backward scale factor to 1.0 for non real FFTs to do correct output checks
	if(dir == CLFFT_BACKWARD && in_layout != CLFFT_REAL && out_layout != CLFFT_REAL)
		OPENCL_V_THROW (clfftSetPlanScale( plan_handle, CLFFT_BACKWARD, (cl_float)1.0f ), "clfftSetPlanScale failed" );

	OPENCL_V_THROW( clfftBakePlan( plan_handle, 1, &queue, NULL, NULL ), "clfftBakePlan failed" );

	//get the buffersize
	size_t buffersize=0;
	OPENCL_V_THROW( clfftGetTmpBufSize(plan_handle, &buffersize ), "clfftGetTmpBufSize failed" );

	//allocate the intermediate buffer
	cl_mem clMedBuffer=NULL;

	if (buffersize)
	{
		cl_int medstatus;
		clMedBuffer = clCreateBuffer ( context, CL_MEM_READ_WRITE, buffersize, 0, &medstatus);
		OPENCL_V_THROW( medstatus, "Creating intmediate Buffer failed" );
	}

	switch( in_layout )
	{
	case CLFFT_COMPLEX_INTERLEAVED:
	case CLFFT_COMPLEX_PLANAR:
	case CLFFT_HERMITIAN_INTERLEAVED:
	case CLFFT_HERMITIAN_PLANAR:
	case CLFFT_REAL:
		break;
	default:
		//	Don't recognize input layout
		return CLFFT_INVALID_ARG_VALUE;
	}

	switch( out_layout )
	{
	case CLFFT_COMPLEX_INTERLEAVED:
	case CLFFT_COMPLEX_PLANAR:
	case CLFFT_HERMITIAN_INTERLEAVED:
	case CLFFT_HERMITIAN_PLANAR:
	case CLFFT_REAL:
		break;
	default:
		//	Don't recognize output layout
		return CLFFT_INVALID_ARG_VALUE;
	}

	if (( place == CLFFT_INPLACE )
	&&  ( in_layout != out_layout )) {
		switch( in_layout )
		{
		case CLFFT_COMPLEX_INTERLEAVED:
			{
				if( (out_layout == CLFFT_COMPLEX_PLANAR) || (out_layout == CLFFT_HERMITIAN_PLANAR) )
				{
					throw std::runtime_error( "Cannot use the same buffer for interleaved->planar in-place transforms" );
				}
				break;
			}
		case CLFFT_COMPLEX_PLANAR:
			{
				if( (out_layout == CLFFT_COMPLEX_INTERLEAVED) || (out_layout == CLFFT_HERMITIAN_INTERLEAVED) )
				{
					throw std::runtime_error( "Cannot use the same buffer for planar->interleaved in-place transforms" );
				}
				break;
			}
		case CLFFT_HERMITIAN_INTERLEAVED:
			{
				if( out_layout != CLFFT_REAL )
				{
					throw std::runtime_error( "Cannot use the same buffer for interleaved->planar in-place transforms" );
				}
				break;
			}
		case CLFFT_HERMITIAN_PLANAR:
			{
				throw std::runtime_error( "Cannot use the same buffer for planar->interleaved in-place transforms" );
				break;
			}
		case CLFFT_REAL:
			{
				if( (out_layout == CLFFT_COMPLEX_PLANAR) || (out_layout == CLFFT_HERMITIAN_PLANAR) )
				{
					throw std::runtime_error( "Cannot use the same buffer for interleaved->planar in-place transforms" );
				}
				break;
			}
		}
	}


	cl_mem * BuffersOut = ( place == CLFFT_INPLACE ) ? NULL : &output_cl_mem_buffers[ 0 ];

	// Execute once for basic functional test
	OPENCL_V_THROW( clfftEnqueueTransform( plan_handle, dir, 1, &queue, 0, NULL, NULL,
		&input_cl_mem_buffers[ 0 ], BuffersOut, clMedBuffer ),
		"clfftEnqueueTransform failed" );

	OPENCL_V_THROW( clFinish( queue ), "clFinish failed" );
	

	//	Create and initialize our timer class, if the external timer shared library loaded
	baseStatTimer* timer = NULL;
	size_t	clFFTID = 0;
	if( get_timer )
	{
		timer = get_timer( CLFFT_GPU );
		timer->Reserve( 1, profile_count );
		timer->setNormalize( true );

		clFFTID	= timer->getUniqueID( "clFFT", 0 );
	}

	cl_event *outEvent = new cl_event[profile_count];
	for( cl_uint i = 0; i < profile_count; ++i ) outEvent[i] = 0;

	if(profile_count > 1)
	{
		Timer tr;		
		tr.Start();
		for( cl_uint i = 0; i < profile_count; ++i )
		{
			if( timer ) timer->Start( clFFTID );

			OPENCL_V_THROW( clfftEnqueueTransform( plan_handle, dir, 1, &queue, 0, NULL, &outEvent[i],
				&input_cl_mem_buffers[ 0 ], BuffersOut, clMedBuffer ),
				"clfftEnqueueTransform failed" );

			if( timer ) timer->Stop( clFFTID );
		}
		OPENCL_V_THROW( clWaitForEvents ( profile_count, outEvent ), "clWaitForEvents  failed" );

		double wtime = tr.Sample()/((double)profile_count);

		OPENCL_V_THROW( clFinish( queue ), "clFinish failed" );

		size_t totalLen = 1;
		for(int i=0; i<dim; i++) totalLen *= lengths[i];

		double constMultiplier = 1.0;
		if( (in_layout == CLFFT_REAL ) || (out_layout == CLFFT_REAL) )
			constMultiplier = 2.5;
		else
			constMultiplier = 5.0;

		double opsconst = constMultiplier * (double)totalLen * log((double)totalLen) / log(2.0);


		tout << "\nExecution wall time: " << 1000.0*wtime << " ms" << std::endl;
		tout << "Execution gflops: " << ((double)batch_size * opsconst)/(1000000000.0*wtime) << std::endl;

	}

	if(clMedBuffer) clReleaseMemObject(clMedBuffer);

	if( timer && (command_queue_flags & CL_QUEUE_PROFILING_ENABLE) )
	{
		//	Remove all timings that are outside of 2 stddev (keep 65% of samples); we ignore outliers to get a more consistent result
		timer->pruneOutliers( 2.0 );
		timer->Print( );
		timer->Reset( );
	}

	/*****************/
	FreeSharedLibrary( timerLibHandle );

	for( cl_uint i = 0; i < profile_count; ++i )
	{
		if(outEvent[i])
			clReleaseEvent(outEvent[i]);
	}

	delete[] outEvent;

	// Read and check output data
	// This check is not valid if the FFT is executed multiple times inplace.
	//
	if (( place == CLFFT_OUTOFPLACE )
	||  ( profile_count == 1))
	{
		bool checkflag= false;
		switch( out_layout )
		{
		case CLFFT_HERMITIAN_INTERLEAVED:
		case CLFFT_COMPLEX_INTERLEAVED:
			{
				std::vector< std::complex< T > > output( outfftBatchSize );

				if( place == CLFFT_INPLACE )
				{
					OPENCL_V_THROW( clEnqueueReadBuffer( queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, size_of_input_buffers_in_bytes, &output[ 0 ],
						0, NULL, NULL ),
						"Reading the result buffer failed" );
				}
				else
				{
					OPENCL_V_THROW( clEnqueueReadBuffer( queue, BuffersOut[ 0 ], CL_TRUE, 0, size_of_output_buffers_in_bytes, &output[ 0 ],
						0, NULL, NULL ),
						"Reading the result buffer failed" );
				}

				//check output data
				for( cl_uint i = 0; i < outfftBatchSize; ++i )
				{
					if (0 == (i % outfftVectorSizePadded))
					{
						if (output[i].real() != outfftVectorSize)
						{
							checkflag = true;
							break;
						}
					}
					else
					{
						if (output[ i ].real() != 0)
						{
							checkflag = true;
							break;
						}
					}

					if (output[ i ].imag() != 0)
					{
						checkflag = true;
						break;
					}
				}
			}
			break;
		case CLFFT_HERMITIAN_PLANAR:
		case CLFFT_COMPLEX_PLANAR:
			{
				std::valarray< T > real( outfftBatchSize );
				std::valarray< T > imag( outfftBatchSize );

				if( place == CLFFT_INPLACE )
				{
					OPENCL_V_THROW( clEnqueueReadBuffer( queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, size_of_input_buffers_in_bytes, &real[ 0 ],
						0, NULL, NULL ),
						"Reading the result buffer failed" );
					OPENCL_V_THROW( clEnqueueReadBuffer( queue, input_cl_mem_buffers[ 1 ], CL_TRUE, 0, size_of_input_buffers_in_bytes, &imag[ 0 ],
						0, NULL, NULL ),
						"Reading the result buffer failed" );
				}
				else
				{
					OPENCL_V_THROW( clEnqueueReadBuffer( queue, BuffersOut[ 0 ], CL_TRUE, 0, size_of_output_buffers_in_bytes, &real[ 0 ],
						0, NULL, NULL ),
						"Reading the result buffer failed" );
					OPENCL_V_THROW( clEnqueueReadBuffer( queue, BuffersOut[ 1 ], CL_TRUE, 0, size_of_output_buffers_in_bytes, &imag[ 0 ],
						0, NULL, NULL ),
						"Reading the result buffer failed" );
				}

				//  Check output data
				for( cl_uint i = 0; i < outfftBatchSize; ++i )
				{
					if (0 == (i % outfftVectorSizePadded))
					{
						if (real[i] != outfftVectorSize)
						{
							checkflag = true;
							break;
						}
					}
					else
					{
						if (real[i] != 0)
						{
							checkflag = true;
							break;
						}
					}

					if (imag[i] != 0)
					{
						checkflag = true;
						break;
					}
				}
			}
			break;
		case CLFFT_REAL:
			{
				std::valarray< T > real( outfftBatchSize );

				if( place == CLFFT_INPLACE )
				{
					OPENCL_V_THROW( clEnqueueReadBuffer( queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, size_of_input_buffers_in_bytes, &real[ 0 ],
						0, NULL, NULL ),
						"Reading the result buffer failed" );
				}
				else
				{
					OPENCL_V_THROW( clEnqueueReadBuffer( queue, BuffersOut[ 0 ], CL_TRUE, 0, size_of_output_buffers_in_bytes, &real[ 0 ],
						0, NULL, NULL ),
						"Reading the result buffer failed" );
				}

				////check output data

				for(size_t b = 0; b < batch_size; b++)
				{
					size_t p3 = b * o_strides[3];
					for(size_t k = 0; k < lengths[2]; k++)
					{
						size_t p2 = p3 + k * o_strides[2];
						for(size_t j = 0; j < lengths[1]; j++)
						{
							size_t p1 = p2 + j * o_strides[1];
							for(size_t i = 0; i < lengths[0]; i++)
							{
								size_t p0 = p1 + i * o_strides[0];

								if (real[p0] != 1)
								{
									checkflag = true;
									break;
								}

							}
						}
					}
				}
			}
			break;
		default:
			{
				throw std::runtime_error( "Input layout format not yet supported" );
			}
			break;
		}

		if (checkflag)
		{
			std::cout << "\n\n\t\tInternal Client Test *****FAIL*****" << std::endl;
		}
		else
		{
			std::cout << "\n\n\t\tInternal Client Test *****PASS*****" << std::endl;
		}
	}

	OPENCL_V_THROW( clfftDestroyPlan( &plan_handle ), "clfftDestroyPlan failed" );
	OPENCL_V_THROW( clfftTeardown( ), "clfftTeardown failed" );

	cleanupCL( &context, &queue, countOf( input_cl_mem_buffers ), input_cl_mem_buffers, countOf( output_cl_mem_buffers ), output_cl_mem_buffers, NULL );
	return 0;
}

int _tmain( int argc, _TCHAR* argv[] )
{
	//	This helps with mixing output of both wide and narrow characters to the screen
	std::ios::sync_with_stdio( false );

	//	Define MEMORYREPORT on windows platfroms to enable debug memory heap checking
#if defined( MEMORYREPORT ) && defined( _WIN32 )
	TCHAR logPath[ MAX_PATH ];
	::GetCurrentDirectory( MAX_PATH, logPath );
	::_tcscat_s( logPath, _T( "\\MemoryReport.txt") );

	//	We leak the handle to this file, on purpose, so that the ::_CrtSetReportFile() can output it's memory
	//	statistics on app shutdown
	HANDLE hLogFile;
	hLogFile = ::CreateFile( logPath, GENERIC_WRITE,
		FILE_SHARE_READ|FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL );

	::_CrtSetReportMode( _CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW | _CRTDBG_MODE_DEBUG );
	::_CrtSetReportMode( _CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW | _CRTDBG_MODE_DEBUG );
	::_CrtSetReportMode( _CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG );

	::_CrtSetReportFile( _CRT_ASSERT, hLogFile );
	::_CrtSetReportFile( _CRT_ERROR, hLogFile );
	::_CrtSetReportFile( _CRT_WARN, hLogFile );

	int tmp = ::_CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
	tmp |= _CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF;
	::_CrtSetDbgFlag( tmp );

	//	By looking at the memory leak report that is generated by this debug heap, there is a number with
	//	{} brackets that indicates the incremental allocation number of that block.  If you wish to set
	//	a breakpoint on that allocation number, put it in the _CrtSetBreakAlloc() call below, and the heap
	//	will issue a bp on the request, allowing you to look at the call stack
	//	::_CrtSetBreakAlloc( 1833 );

#endif /* MEMORYREPORT */

	//	OpenCL state 
	cl_device_type		deviceType	= CL_DEVICE_TYPE_ALL;
	cl_int				deviceId = 0;
	cl_int				platformId = 0;

	//	FFT state

	clfftResultLocation	place = CLFFT_INPLACE;
	clfftLayout	inLayout  = CLFFT_COMPLEX_INTERLEAVED;
	clfftLayout	outLayout = CLFFT_COMPLEX_INTERLEAVED;
	clfftPrecision precision = CLFFT_SINGLE;
	clfftDirection dir = CLFFT_FORWARD;
	size_t lengths[ 3 ] = {1,1,1};
	size_t iStrides[ 4 ] = {0,0,0,0};
	size_t oStrides[ 4 ] = {0,0,0,0};
	cl_uint profile_count = 0;

	cl_uint command_queue_flags = 0;
	size_t batchSize = 1;


	//	Initialize flags for FFT library
	std::auto_ptr< clfftSetupData > setupData( new clfftSetupData );
	OPENCL_V_THROW( clfftInitSetupData( setupData.get( ) ),
		"clfftInitSetupData failed" );

	try
	{
		// Declare the supported options.
		po::options_description desc( "clFFT client command line options" );
		desc.add_options()
			( "help,h",        "produces this help message" )
			( "version,v",     "Print queryable version information from the clFFT library" )
			( "clinfo,i",      "Print queryable information of all the OpenCL runtimes and devices" )
			( "printChosen",   "Print queryable information of the selected OpenCL runtime and device" )
			( "gpu,g",         "Force selection of OpenCL GPU devices only" )
			( "cpu,c",         "Force selection of OpenCL CPU devices only" )
			( "all,a",         "Force selection of all OpenCL devices (default)" )
			( "platform",      po::value< cl_int >( &platformId )->default_value( 0 ),   "Select a specific OpenCL platform id as it is reported by clinfo" )
			( "device",        po::value< cl_int >( &deviceId )->default_value( 0 ),   "Select a specific OpenCL device id as it is reported by clinfo" )
			( "outPlace,o",    "Out of place FFT transform (default: in place)" )
			( "double",		   "Double precision transform (default: single)" )
			( "inv",			"Backward transform (default: forward)" )
			( "dumpKernels,d", "FFT engine will dump generated OpenCL FFT kernels to disk (default: dump off)" )
			( "lenX,x",        po::value< size_t >( &lengths[ 0 ] )->default_value( 1024 ),   "Specify the length of the 1st dimension of a test array" )
			( "lenY,y",        po::value< size_t >( &lengths[ 1 ] )->default_value( 1 ),      "Specify the length of the 2nd dimension of a test array" )
			( "lenZ,z",        po::value< size_t >( &lengths[ 2 ] )->default_value( 1 ),      "Specify the length of the 3rd dimension of a test array" )
			( "isX",   po::value< size_t >( &iStrides[ 0 ] )->default_value( 1 ),						"Specify the input stride of the 1st dimension of a test array" )
			( "isY",   po::value< size_t >( &iStrides[ 1 ] )->default_value( 0 ),	"Specify the input stride of the 2nd dimension of a test array" )
			( "isZ",   po::value< size_t >( &iStrides[ 2 ] )->default_value( 0 ),	"Specify the input stride of the 3rd dimension of a test array" )
			( "iD", po::value< size_t >( &iStrides[ 3 ] )->default_value( 0 ), "input distance between subsequent sets of data when batch size > 1" )
			( "osX",   po::value< size_t >( &oStrides[ 0 ] )->default_value( 1 ),						"Specify the output stride of the 1st dimension of a test array" )
			( "osY",   po::value< size_t >( &oStrides[ 1 ] )->default_value( 0 ),	"Specify the output stride of the 2nd dimension of a test array" )
			( "osZ",   po::value< size_t >( &oStrides[ 2 ] )->default_value( 0 ),	"Specify the output stride of the 3rd dimension of a test array" )
			( "oD", po::value< size_t >( &oStrides[ 3 ] )->default_value( 0 ), "output distance between subsequent sets of data when batch size > 1" )
			( "batchSize,b",   po::value< size_t >( &batchSize )->default_value( 1 ), "If this value is greater than one, arrays will be used " )
			( "profile,p",     po::value< cl_uint >( &profile_count )->default_value( 1 ), "Time and report the kernel speed of the FFT (default: profiling off)" )
			( "inLayout",      po::value< clfftLayout >( &inLayout )->default_value( CLFFT_COMPLEX_INTERLEAVED ), "Layout of input data:\n1) interleaved\n2) planar\n3) hermitian interleaved\n4) hermitian planar\n5) real" )
			( "outLayout",     po::value< clfftLayout >( &outLayout )->default_value( CLFFT_COMPLEX_INTERLEAVED ), "Layout of input data:\n1) interleaved\n2) planar\n3) hermitian interleaved\n4) hermitian planar\n5) real" )
			;

		po::variables_map vm;
		po::store( po::parse_command_line( argc, argv, desc ), vm );
		po::notify( vm );

		if( vm.count( "version" ) )
		{
			const int indent = countOf( "clFFT client API version: " );
			tout << std::left << std::setw( indent ) << _T( "clFFT client API version: " )
				<< clfftVersionMajor << _T( "." )
				<< clfftVersionMinor << _T( "." )
				<< clfftVersionPatch << std::endl;

			cl_uint libMajor, libMinor, libPatch;
			clfftGetVersion( &libMajor, &libMinor, &libPatch );

			tout << std::left << std::setw( indent ) << _T( "clFFT runtime version: " )
				<< libMajor << _T( "." )
				<< libMinor << _T( "." )
				<< libPatch << std::endl << std::endl;
		}

		if( vm.count( "help" ) )
		{
			//	This needs to be 'cout' as program-options does not support wcout yet
			std::cout << desc << std::endl;
			return 0;
		}

		size_t mutex = ((vm.count( "gpu" ) > 0) ? 1 : 0)
			| ((vm.count( "cpu" ) > 0) ? 2 : 0)
			| ((vm.count( "all" ) > 0) ? 4 : 0);
		if ((mutex & (mutex-1)) != 0) {
			terr << _T("You have selected mutually-exclusive OpenCL device options:") << std::endl;
			if (vm.count ( "gpu" )  > 0) terr << _T("    gpu,g   Force selection of OpenCL GPU devices only" ) << std::endl;
			if (vm.count ( "cpu" )  > 0) terr << _T("    cpu,c   Force selection of OpenCL CPU devices only" ) << std::endl;
			if (vm.count ( "all" )  > 0) terr << _T("    all,a   Force selection of all OpenCL devices (default)" ) << std::endl;
			return 1;
		}

		if( vm.count( "gpu" ) )
		{
			deviceType	= CL_DEVICE_TYPE_GPU;
		}

		if( vm.count( "cpu" ) )
		{
			deviceType	= CL_DEVICE_TYPE_CPU;
		}

		if( vm.count( "all" ) )
		{
			deviceType	= CL_DEVICE_TYPE_ALL;
		}

		if( vm.count( "clinfo" ) )
		{
			std::vector< cl_platform_id > platformInfos;
			std::vector< std::vector< cl_device_id > > deviceInfos;
			discoverCLPlatforms( deviceType, platformInfos, deviceInfos );
			prettyPrintCLPlatforms(platformInfos, deviceInfos);
			return 0;
		}

		bool printInfo = false;
		if( vm.count( "printChosen" ) )
		{
			printInfo = true;
		}

		if( vm.count( "outPlace" ) )
		{
			place = CLFFT_OUTOFPLACE;
		}

		if( vm.count( "double" ) )
		{
			precision = CLFFT_DOUBLE;
		}

		if( vm.count( "inv" ) )
		{
			dir = CLFFT_BACKWARD;
		}

		if( profile_count > 1 )
		{
			command_queue_flags |= CL_QUEUE_PROFILING_ENABLE;
		}

		if( vm.count( "dumpKernels" ) )
		{
			setupData->debugFlags	|= CLFFT_DUMP_PROGRAMS;
		}

		int inL = (int)inLayout;
		int otL = (int)outLayout;

		// input output layout support matrix
		int ioLayoutSupport[5][5] =		{
										{ 1, 1, 0, 0, 1 },
										{ 1, 1, 0, 0, 1 },
										{ 0, 0, 0, 0, 1 },
										{ 0, 0, 0, 0, 1 },
										{ 1, 1, 1, 1, 0 },
										};

		if((inL < 1) || (inL > 5)) throw std::runtime_error( "Invalid Input layout format" );
		if((otL < 1) || (otL > 5)) throw std::runtime_error( "Invalid Output layout format" );

		if(ioLayoutSupport[inL-1][otL-1] == 0) throw std::runtime_error( "Invalid combination of Input/Output layout formats" );

		if( ((inL == 1) || (inL == 2)) && ((otL == 1) || (otL == 2)) ) // Complex-Complex cases
		{
			iStrides[1] = iStrides[1] ? iStrides[1] : lengths[0] * iStrides[0];
			iStrides[2] = iStrides[2] ? iStrides[2] : lengths[1] * iStrides[1];
			iStrides[3] = iStrides[3] ? iStrides[3] : lengths[2] * iStrides[2];



			if(place == CLFFT_INPLACE)
			{
				oStrides[0] = iStrides[0];
				oStrides[1] = iStrides[1];
				oStrides[2] = iStrides[2];
				oStrides[3] = iStrides[3];
			}
			else
			{
				oStrides[1] = oStrides[1] ? oStrides[1] : lengths[0] * oStrides[0];
				oStrides[2] = oStrides[2] ? oStrides[2] : lengths[1] * oStrides[1];
				oStrides[3] = oStrides[3] ? oStrides[3] : lengths[2] * oStrides[2];
			}
		}
		else // Real-Complex and Complex-Real cases
		{
			size_t *rst, *cst;
			size_t N = lengths[0];
			size_t Nt = 1 + lengths[0]/2;
			bool iflag = false;
			bool rcFull = (inL == 1) || (inL == 2) || (otL == 1) || (otL == 2);

			if(inLayout == CLFFT_REAL) { iflag = true; rst = iStrides; }
			else { rst = oStrides; } // either in or out should be REAL

			// Set either in or out strides whichever is real
			if(place == CLFFT_INPLACE)
			{
				if(rcFull)	{ rst[1] = rst[1] ? rst[1] :  N * 2 * rst[0]; }
				else		{ rst[1] = rst[1] ? rst[1] : Nt * 2 * rst[0]; }

				rst[2] = rst[2] ? rst[2] : lengths[1] * rst[1];
				rst[3] = rst[3] ? rst[3] : lengths[2] * rst[2];
			}
			else
			{
				rst[1] = rst[1] ? rst[1] : lengths[0] * rst[0];
				rst[2] = rst[2] ? rst[2] : lengths[1] * rst[1];
				rst[3] = rst[3] ? rst[3] : lengths[2] * rst[2];
			}

			// Set the remaining of in or out strides that is not real
			if(iflag) { cst = oStrides; }
			else	  { cst = iStrides; }

			if(rcFull)	{ cst[1] = cst[1] ? cst[1] :  N * cst[0]; }
			else		{ cst[1] = cst[1] ? cst[1] : Nt * cst[0]; }

			cst[2] = cst[2] ? cst[2] : lengths[1] * cst[1];
			cst[3] = cst[3] ? cst[3] : lengths[2] * cst[2];
		}

		if( precision == CLFFT_SINGLE )
			transform<float>( lengths, iStrides, oStrides, batchSize, inLayout, outLayout, place, precision, dir, deviceType, deviceId, platformId, printInfo, command_queue_flags, profile_count, setupData );
		else
			transform<double>( lengths, iStrides, oStrides, batchSize, inLayout, outLayout, place, precision, dir, deviceType, deviceId, platformId, printInfo, command_queue_flags, profile_count, setupData );
	}
	catch( std::exception& e )
	{
		terr << _T( "clFFT error condition reported:" ) << std::endl << e.what() << std::endl;
		return 1;
	}
	return 0;
}
