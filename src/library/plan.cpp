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

////////////////////////////////////////////

// clfft.plan.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <math.h>
#include "private.h"
#include "repo.h"
#include "plan.h"
#include "generator.stockham.h"
#include "../include/convenienceFunctions.h"

using std::vector;

const std::string beginning_of_binary( "<[£_beginning_of_binary_£]>" );
const std::string end_of_binary( "<[£_I_may_be_a_sorry_case,_but_I_don't_write_jokes_in_base_13_£]>" );
const std::string end_of_file( "<[£_You're_off_the_edge_of_the_map,_mate._Here_there_be_monsters_£]>" );

//	This operator is used to sort FFTKernelGenKeyParams structs inside of a std::map
bool operator<( const FFTKernelGenKeyParams& lhs, const FFTKernelGenKeyParams& rhs)
{
	int ret = ::memcmp( &lhs, &rhs, sizeof( FFTKernelGenKeyParams ) );

	if( ret < 0 )
		return true;

	return false;
}

clfftStatus	clfftCreateDefaultPlan( clfftPlanHandle* plHandle, cl_context context, const clfftDim dim,
						const size_t* clLengths )
{
	if( clLengths == NULL )
		return CLFFT_INVALID_HOST_PTR;

	size_t lenX = 1, lenY = 1, lenZ = 1;

	switch( dim )
	{
		case CLFFT_1D:
		{
			//	Minimum length size is 1
			if( clLengths[ DimX ] == 0 )
				return CLFFT_INVALID_ARG_VALUE;

			if( !IsASupportedLength( clLengths[ DimX ] ) )
			{
				return CLFFT_NOTIMPLEMENTED;
			}

			lenX = clLengths[ DimX ];
		}
			break;
		case CLFFT_2D:
		{
			//	Minimum length size is 1
			if( clLengths[ DimX ] == 0 || clLengths[ DimY ] == 0 )
				return CLFFT_INVALID_ARG_VALUE;

			if( !IsASupportedLength( clLengths[ DimX ] ) || !IsASupportedLength( clLengths[ DimY ] ) )
			{
				return CLFFT_NOTIMPLEMENTED;
			}

			lenX = clLengths[ DimX ];
			lenY = clLengths[ DimY ];
		}
			break;
		case CLFFT_3D:
		{
			//	Minimum length size is 1
			if( clLengths[ DimX ] == 0 || clLengths[ DimY ] == 0 || clLengths[ DimZ ] == 0 )
				return CLFFT_INVALID_ARG_VALUE;

			if( !IsASupportedLength( clLengths[ DimX ] ) || !IsASupportedLength( clLengths[ DimY ] ) ||
				!IsASupportedLength( clLengths[ DimZ ] ))
			{
				return CLFFT_NOTIMPLEMENTED;
			}

			lenX = clLengths[ DimX ];
			lenY = clLengths[ DimY ];
			lenZ = clLengths[ DimZ ];
		}
			break;
		default:
			return CLFFT_NOTIMPLEMENTED;
			break;
	}

	FFTPlan* fftPlan	= NULL;
	FFTRepo& fftRepo	= FFTRepo::getInstance( );
	OPENCL_V( fftRepo.createPlan( plHandle, fftPlan ), _T( "fftRepo.insertPlan failed" ) );

	fftPlan->baked			= false;
	fftPlan->dim			= dim;
	fftPlan->placeness		= CLFFT_INPLACE;
	fftPlan->inputLayout	= CLFFT_COMPLEX_INTERLEAVED;
	fftPlan->outputLayout	= CLFFT_COMPLEX_INTERLEAVED;
	fftPlan->precision		= CLFFT_SINGLE;
	fftPlan->context		= context;
	fftPlan->forwardScale	= 1.0;
	fftPlan->backwardScale	= 1.0 / static_cast< double >( lenX * lenY * lenZ );
	fftPlan->batchsize		= 1;

	fftPlan->gen			= Stockham; //default setting

	OPENCL_V(fftPlan->SetEnvelope(), _T("SetEnvelope failed"));

	clRetainContext( fftPlan->context );

	/////////////////////////////////////////////////////////////////
	// Detect OpenCL devices
	/////////////////////////////////////////////////////////////////
	// First, get the size of device list data
	size_t deviceListSize;
	OPENCL_V( ::clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &deviceListSize ),
		"Getting device array size ( ::clGetContextInfo() )" );

	//	Allocate memory for the devices
	fftPlan->devices.resize( deviceListSize / sizeof( cl_device_id ) );

	/* Now, get the device list data */
	OPENCL_V( ::clGetContextInfo( context, CL_CONTEXT_DEVICES, deviceListSize, &fftPlan->devices[ 0 ], NULL ),
		"Getting device array ( ::clGetContextInfo() )" );

	//	Need to devise a way to generate better names
	tstringstream	tstream;
	tstream << _T( "plan_" ) << *plHandle;

	lockRAII* planLock	= NULL;
	OPENCL_V( fftRepo.getPlan( *plHandle, fftPlan, planLock ), _T( "fftRepo.getPlan failed" ) );
	planLock->setName( tstream.str( ) );

	//	Set the lengths and default strides/pitches depending on the dim that the user passes to us
	switch( dim )
	{
		case CLFFT_1D:
		{
			fftPlan->length.push_back( lenX );
			fftPlan->inStride.push_back( 1 );
			fftPlan->outStride.push_back( 1 );
			fftPlan->iDist		= lenX;
			fftPlan->oDist		= lenX;
		}
			break;
		case CLFFT_2D:
		{
			fftPlan->length.push_back( lenX );
			fftPlan->length.push_back( lenY );
			fftPlan->inStride.push_back( 1 );
			fftPlan->inStride.push_back( lenX );
			fftPlan->outStride.push_back( 1 );
			fftPlan->outStride.push_back( lenX );
			fftPlan->iDist		= lenX*lenY;
			fftPlan->oDist		= lenX*lenY;
		}
			break;
		case CLFFT_3D:
		{
			fftPlan->length.push_back( lenX );
			fftPlan->length.push_back( lenY );
			fftPlan->length.push_back( lenZ );
			fftPlan->inStride.push_back( 1 );
			fftPlan->inStride.push_back( lenX );
			fftPlan->inStride.push_back( lenX*lenY );
			fftPlan->outStride.push_back( 1 );
			fftPlan->outStride.push_back( lenX );
			fftPlan->outStride.push_back( lenX*lenY );
			fftPlan->iDist		= lenX*lenY*lenZ;
			fftPlan->oDist		= lenX*lenY*lenZ;
		}
			break;
	}

	return	CLFFT_SUCCESS;
}

//	Read the kernels that this plan uses from file, and store into the plan
clfftStatus WriteKernel( const clfftPlanHandle plHandle, const clfftGenerators gen, const FFTKernelGenKeyParams& fftParams, const cl_context& context )
{
	FFTRepo& fftRepo	= FFTRepo::getInstance( );

	//	Logic to define a sensible filename
	const std::string kernelPrefix( "clfft.kernel." );
	std::string generatorName;
	std::stringstream kernelPath;

	switch( gen )
	{
		case Stockham:		generatorName = "Stockham"; break;
		case Transpose:		generatorName = "Transpose"; break;
	}

	kernelPath << kernelPrefix << generatorName << plHandle << ".cl";

	//	Logic to write string contents out to file
	tofstreamRAII< std::ofstream, std::string > kernelFile( kernelPath.str( ) );
	if( !kernelFile.get( ) )
	{
		std::cerr << "Failed to open kernel file for writing: " << kernelPath.str( ) << std::endl;
		return CLFFT_FILE_CREATE_FAILURE;
	}

	std::string kernel;
	OPENCL_V( fftRepo.getProgramCode( gen, fftParams, kernel, context ), _T( "fftRepo.getProgramCode failed." ) );

	kernelFile.get( ) << kernel << std::endl;

	return	CLFFT_SUCCESS;
}

// **************** TODO TODO TODO ***********************
// Making CompileKernels function take in command queue parameter so we can build for 1 particular device only;
// this may not be desirable for persistent plans, where we may have to compile for all devices in the context;
// make changes appropriately before enabling persistent plans and then remove this comment

//	Compile the kernels that this plan uses, and store into the plan
clfftStatus CompileKernels( const cl_command_queue commQueueFFT, const clfftPlanHandle plHandle, const clfftGenerators gen, FFTPlan* fftPlan )
{
	cl_int status = 0;
	size_t deviceListSize = 0;

	FFTRepo& fftRepo	= FFTRepo::getInstance( );


	// create a cl program executable for the device associated with command queue
	// Get the device
	cl_device_id q_device;
	clGetCommandQueueInfo(commQueueFFT, CL_QUEUE_DEVICE, sizeof(cl_device_id), &q_device, NULL);

	FFTKernelGenKeyParams fftParams;
	OPENCL_V( fftPlan->GetKernelGenKey( fftParams ), _T("GetKernelGenKey() failed!") );

	cl_program program;
  if( fftRepo.getclProgram( gen, fftParams, program, fftPlan->context ) == CLFFT_INVALID_PROGRAM )
	{
		//	If the user wishes us to write the kernels out to disk, we do so
		if( fftRepo.setupData.debugFlags & CLFFT_DUMP_PROGRAMS )
		{
			OPENCL_V( WriteKernel( plHandle, gen, fftParams, fftPlan->context ), _T( "WriteKernel failed." ) );
		}

		std::string programCode;
		OPENCL_V( fftRepo.getProgramCode( gen, fftParams, programCode, fftPlan->context  ), _T( "fftRepo.getProgramCode failed." ) );

		const char* source = programCode.c_str();
		program = clCreateProgramWithSource( fftPlan->context, 1, &source, NULL, &status );
		OPENCL_V( status, _T( "clCreateProgramWithSource failed." ) );

		// create a cl program executable for the device associated with command queue

#if defined(DEBUGGING)
		status = clBuildProgram( program, 1, &q_device, "-g -cl-opt-disable", NULL, NULL); // good for debugging kernels

// if you have trouble creating smbols that GDB can pick up to set a breakpoint after kernels are loaded into memory
// this can be used to stop execution to allow you to set a breakpoint in a kernel after kernel symbols are in memory.
#ifdef DEBUG_BREAK_GDB
		__debugbreak();
#endif
#else
		status = clBuildProgram( program, 1, &q_device, NULL, NULL, NULL);
#endif
		if( status != CL_SUCCESS )
		{
			if( status == CL_BUILD_PROGRAM_FAILURE )
			{
				size_t buildLogSize = 0;
				OPENCL_V( clGetProgramBuildInfo( program, q_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize ),
						_T( "clGetProgramBuildInfo failed" ) );

				vector< char > buildLog( buildLogSize );
				::memset( &buildLog[ 0 ], 0x0, buildLogSize );

				OPENCL_V( clGetProgramBuildInfo( program, q_device, CL_PROGRAM_BUILD_LOG, buildLogSize, &buildLog[ 0 ], NULL ),
						_T( "clGetProgramBuildInfo failed" ) );

				std::cerr << "\n\t\t\tBUILD LOG\n";
				std::cerr << "************************************************\n";
				std::cerr << &buildLog[ 0 ] << std::endl;
				std::cerr << "************************************************\n";
			}

			OPENCL_V( status, _T( "clBuildProgram failed" ) );
		}

		fftRepo.setclProgram( gen, fftParams, program );

		// For real transforms we comppile either forward or backward kernel
		bool r2c_transform = (fftParams.fft_inputLayout == CLFFT_REAL);
		bool c2r_transform = (fftParams.fft_outputLayout == CLFFT_REAL);
		bool real_transform = (gen == Copy) ? true : (r2c_transform || c2r_transform);
		bool h2c = (gen == Copy) && ((fftParams.fft_inputLayout == CLFFT_HERMITIAN_PLANAR) || (fftParams.fft_inputLayout == CLFFT_HERMITIAN_INTERLEAVED));
		bool c2h = (gen == Copy) && ((fftParams.fft_outputLayout == CLFFT_HERMITIAN_PLANAR) || (fftParams.fft_outputLayout == CLFFT_HERMITIAN_INTERLEAVED));

		// get a kernel object handle for a kernel with the given name
		cl_kernel kernel;
		if( (!real_transform) || r2c_transform || c2h )
		{
			if( fftRepo.getclKernel( program, CLFFT_FORWARD, kernel ) == CLFFT_INVALID_KERNEL )
			{
				std::string entryPoint;
				OPENCL_V( fftRepo.getProgramEntryPoint( gen, fftParams, CLFFT_FORWARD, entryPoint, fftPlan->context ), _T( "fftRepo.getProgramEntryPoint failed." ) );

				kernel = clCreateKernel( program, entryPoint.c_str( ), &status );
				OPENCL_V( status, _T( "clCreateKernel failed" ) );

				fftRepo.setclKernel( program, CLFFT_FORWARD, kernel );
			}
		}

		if( (!real_transform) || c2r_transform || h2c )
		{
			if( fftRepo.getclKernel( program, CLFFT_BACKWARD, kernel ) == CLFFT_INVALID_KERNEL )
			{
				std::string entryPoint;
				OPENCL_V( fftRepo.getProgramEntryPoint( gen, fftParams, CLFFT_BACKWARD, entryPoint, fftPlan->context ), _T( "fftRepo.getProgramEntryPoint failed." ) );

				kernel = clCreateKernel( program, entryPoint.c_str( ), &status );
				OPENCL_V( status, _T( "clCreateKernel failed" ) );

				fftRepo.setclKernel( program, CLFFT_BACKWARD, kernel );
			}
		}
	}

//TODO caching kernel binaries for later reload
#if 0
	// figure out number of devices and the sizes of the binary for each device.
	OPENCL_V( clGetProgramInfo( program, CL_PROGRAM_NUM_DEVICES, sizeof(fftPlan->number_of_devices), &(fftPlan->number_of_devices), NULL ), _T("CompileKernels(): error getting number of devices") );


	// get the sizes of the different binaries
	fftPlan->ResetBinarySizes();
	OPENCL_V( clGetProgramInfo( program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * fftPlan->number_of_devices, fftPlan->binary_sizes.get(), NULL ), _T("CompileKernels(): error getting binary sizes") );

	// we need a list of naked pointers to all of the binaries for OpenCL
	std::unique_ptr<char*[]> naked_binary_pointers( new char*[fftPlan->number_of_devices] );

	// make space for all of the generated binaries
	for( int i = 0; i < fftPlan->number_of_devices; i++ )
	{
		// this is our permanent storage place for the binaries
		fftPlan->binaries.push_back( std::unique_ptr<char[]>(new char[fftPlan->binary_sizes[i]] ) );
		// and we need this second copy of it for OpenCL
		naked_binary_pointers[i] = fftPlan->binaries[i].get();
	}

	// copy all of the generated binaries over
	OPENCL_V( clGetProgramInfo( program, CL_PROGRAM_BINARIES, sizeof(char*) * fftPlan->number_of_devices, naked_binary_pointers.get(), NULL ), _T("CompileKernels(): error getting program binaries") );
#endif

	return	CLFFT_SUCCESS;
}

//TODO caching kernel binaries for later reload
#if 0
//	Compile the kernels that this plan uses, and store into the plan
clfftStatus LoadCompiledKernels( const clfftPlanHandle plHandle, const clfftGenerators gen, FFTPlan* plan )
{
	// if there are no devices, there are not any kernels to load
	if( plan->number_of_devices == 0 )
		return CLFFT_SUCCESS;

	FFTRepo& repo = FFTRepo::getInstance( );

	FFTKernelGenKeyParams fftParams;
	OPENCL_V( plan->GetKernelGenKey( fftParams ), _T("GetKernelGenKey() failed!") );

	cl_program program;
	if( repo.getclProgram( gen, fftParams, program ) == CLFFT_INVALID_PROGRAM )
	{
		//if( repo.setupData.debugFlags & CLFFT_DUMP_PROGRAMS )
		//{
		//	OPENCL_V( WriteKernel( plHandle, gen, fftParams ), _T( "WriteKernel failed." ) );
		//	//TODO there's no source to spit out, but we should consider giving the user a helpful message
		//	// such as "there's no source to output -- kernel binaries loaded from file"
		//}

		std::unique_ptr<cl_int[]> binary_status( new cl_int[plan->number_of_devices] );
		cl_int error_code;

		std::unique_ptr<const unsigned char*[]> binaries( new const unsigned char*[plan->number_of_devices] );
		for( int i = 0; i < plan->number_of_devices; i++ )
		{
			binaries[i] = reinterpret_cast<const unsigned char*>(plan->binaries[0].get());
		}

		if( plan->number_of_devices > 0 )
		{
			program = clCreateProgramWithBinary( plan->context,
				(cl_uint)plan->number_of_devices, &plan->devices[0], &plan->binary_sizes[0], &binaries[0],
				binary_status.get(), &error_code);

			cl_int status = 0;
			// create a cl program executable for all the devices specified
			status = clBuildProgram( program, 1, &plan->devices[0], NULL, NULL, NULL);

			if( status != CL_SUCCESS )
			{
				if( status == CL_BUILD_PROGRAM_FAILURE )
				{
					size_t buildLogSize = 0;
					OPENCL_V( clGetProgramBuildInfo( program, plan->devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize ),
						_T( "clGetProgramBuildInfo failed" ) );

					vector< char > buildLog( buildLogSize );
					::memset( &buildLog[ 0 ], 0x0, buildLogSize );

					OPENCL_V( clGetProgramBuildInfo( program, plan->devices[0], CL_PROGRAM_BUILD_LOG, buildLogSize, &buildLog[ 0 ], NULL ),
						_T( "clGetProgramBuildInfo failed" ) );

					std::cerr << " \n\t\t\tBUILD LOG\n";
					std::cerr << " ************************************************\n";
					std::cerr << &buildLog[ 0 ] << std::endl;
					std::cerr << " ************************************************\n";
				}

				OPENCL_V( status, _T( "clBuildProgram failed" ) );
			}

			repo.setclProgram( gen, fftParams, program );

			// get a kernel object handle for a kernel with the given name
			cl_kernel kernel;
			if( repo.getclKernel( program, CLFFT_FORWARD, kernel ) == CLFFT_INVALID_KERNEL )
			{
				kernel = clCreateKernel( program, "fft_fwd", &status );
				OPENCL_V( status, _T( "clCreateKernel failed" ) );

				repo.setclKernel( program, CLFFT_FORWARD, kernel );
			}

			if( repo.getclKernel( program, CLFFT_BACKWARD, kernel ) == CLFFT_INVALID_KERNEL )
			{
				kernel = clCreateKernel( program, "fft_back", &status );
				OPENCL_V( status, _T( "clCreateKernel failed" ) );

				repo.setclKernel( program, CLFFT_BACKWARD, kernel );
			}

			FFTKernelGenKeyParams params;
			plan->GetKernelGenKey( params );
			OPENCL_V( repo.setProgramEntryPoints( Stockham, params, "fft_fwd", "fft_back" ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );
		}
	}

	return CLFFT_SUCCESS;
}
#endif

clfftStatus	clfftBakePlan( clfftPlanHandle plHandle, cl_uint numQueues, cl_command_queue* commQueueFFT,
							void (CL_CALLBACK *pfn_notify)( clfftPlanHandle plHandle, void *user_data ), void* user_data )
{
	//	We do not currently support multi-GPU transforms
	if( numQueues > 1 )
		return CLFFT_NOTIMPLEMENTED;

	//	Notification mechanism is not set up yet; BakePlan can be called recursively to decompose higher dimension FFT's into
	//	arrays of 1d transforms, and this must be implemented to make only a single callback to the user.
	if( pfn_notify != NULL )
		return CLFFT_NOTIMPLEMENTED;

	if( user_data != NULL )
		return CLFFT_NOTIMPLEMENTED;

	FFTRepo& fftRepo	= FFTRepo::getInstance( );
	FFTPlan* fftPlan	= NULL;
	lockRAII* planLock	= NULL;

	OPENCL_V( fftRepo.getPlan( plHandle, fftPlan, planLock ), _T( "fftRepo.getPlan failed" ) );
	scopedLock sLock( *planLock, _T( "clfftBakePlan" ) );

	// if we have already baked the plan and nothing has changed since, we're done here
	if( fftPlan->baked == true )
	{
		return CLFFT_SUCCESS;
	}

	//find product of lengths
	size_t pLength = 1;
	switch(fftPlan->dim)
	{
		case CLFFT_3D: pLength *= fftPlan->length[DimZ];
		case CLFFT_2D: pLength *= fftPlan->length[DimY];
		case CLFFT_1D: pLength *= fftPlan->length[DimX];
	}

	// upper bounds on transfrom lengths - address this in the next release
	size_t SP_MAX_LEN = 1 << 24;
	size_t DP_MAX_LEN = 1 << 22;
	if((fftPlan->precision == CLFFT_SINGLE) && (pLength > SP_MAX_LEN)) return CLFFT_NOTIMPLEMENTED;
	if((fftPlan->precision == CLFFT_DOUBLE) && (pLength > DP_MAX_LEN)) return CLFFT_NOTIMPLEMENTED;


	// release buffers, as these will be created only in EnqueueTransform
	if( NULL != fftPlan->intBuffer ) { OPENCL_V( clReleaseMemObject( fftPlan->intBuffer ), _T( "Failed to release internal temporary buffer" ) ); fftPlan->intBuffer = NULL; }
	if( NULL != fftPlan->intBufferRC ) { OPENCL_V( clReleaseMemObject( fftPlan->intBufferRC ), _T( "Failed to release internal temporary buffer" ) ); fftPlan->intBufferRC = NULL; }
	if( NULL != fftPlan->intBufferC2R ) { OPENCL_V( clReleaseMemObject( fftPlan->intBufferC2R ), _T( "Failed to release internal temporary buffer" ) ); fftPlan->intBufferC2R = NULL; }


	if(fftPlan->dim == fftPlan->length.size() && fftPlan->gen != Transpose && fftPlan->gen != Copy) // confirm it is top-level plan (user plan)
	{
		if(fftPlan->placeness == CLFFT_INPLACE)
		{
			if( (fftPlan->inputLayout == CLFFT_HERMITIAN_PLANAR) || (fftPlan->outputLayout == CLFFT_HERMITIAN_PLANAR) )
				return CLFFT_INVALID_PLAN;
		}

		// Make sure strides & distance are same for C-C transforms
		if(fftPlan->placeness == CLFFT_INPLACE)
		{
			if( (fftPlan->inputLayout != CLFFT_REAL) && (fftPlan->outputLayout != CLFFT_REAL) )
			{
				// check strides
				for(size_t i=0; i<fftPlan->dim; i++)
					if(fftPlan->inStride[i] != fftPlan->outStride[i])
						return CLFFT_INVALID_PLAN;

				// check distance
				if(fftPlan->iDist != fftPlan->oDist)
					return CLFFT_INVALID_PLAN;
			}
		}
	}

	if(fftPlan->gen == Copy)
	{
		OPENCL_V( fftPlan->GenerateKernel( fftRepo, *commQueueFFT ), _T( "GenerateKernel() failed" ) );
		OPENCL_V( CompileKernels( *commQueueFFT, plHandle, fftPlan->gen, fftPlan ), _T( "CompileKernels() failed" ) );
		fftPlan->baked		= true;
		return	CLFFT_SUCCESS;
	}

//TODO caching kernel binaries for later reload
#if 0
	if( fftPlan->readFromFile == true )
	{
		OPENCL_V( LoadCompiledKernels( plHandle, fftPlan->gen, fftPlan ), _T( "LoadCompiledKernels() failed" ) );

		// all of the plan compressing and subplan making should be done already,
		// but we still need to make constant buffers
		OPENCL_V( fftPlan->AllocateBuffers(), _T("AllocateBuffers() failed"));
		fftPlan->ConstructAndEnqueueConstantBuffers( commQueueFFT );

		if( fftPlan->planX )
		{
			OPENCL_V( clfftBakePlan( fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ), "clfftBakePlan failed (planX)" );
		}

		if( fftPlan->planY )
		{
			OPENCL_V( clfftBakePlan( fftPlan->planY, numQueues, commQueueFFT, NULL, NULL ), "clfftBakePlan failed (planY)" );
		}

		if( fftPlan->planZ )
		{
			OPENCL_V( clfftBakePlan( fftPlan->planZ, numQueues, commQueueFFT, NULL, NULL ), "clfftBakePlan failed (planZ)" );
		}

		fftPlan->baked = true;
		return CLFFT_SUCCESS;
	}
#endif

	bool rc = (fftPlan->inputLayout == CLFFT_REAL) || (fftPlan->outputLayout == CLFFT_REAL);

	// Compress the plan by discarding length '1' dimensions
	// decision to pick generator
	if(fftPlan->dim == fftPlan->length.size() && fftPlan->gen != Transpose && !rc) // confirm it is top-level plan (user plan)
	{
		size_t dmnsn = fftPlan->dim;
		bool pow2flag = true;

		// switch case flows with no 'break' statements
		switch(fftPlan->dim)
		{
		case CLFFT_3D:

			if(fftPlan->length[DimZ] == 1)
			{
				dmnsn -= 1;
				fftPlan-> inStride.erase(fftPlan-> inStride.begin() + 2);
				fftPlan->outStride.erase(fftPlan->outStride.begin() + 2);
				fftPlan->   length.erase(fftPlan->   length.begin() + 2);
			}
			else
			{
				if( !IsPo2(fftPlan->length[DimZ])) pow2flag=false;
			}
		case CLFFT_2D:

			if(fftPlan->length[DimY] == 1)
			{
				dmnsn -= 1;
				fftPlan-> inStride.erase(fftPlan-> inStride.begin() + 1);
				fftPlan->outStride.erase(fftPlan->outStride.begin() + 1);
				fftPlan->   length.erase(fftPlan->   length.begin() + 1);
			}
			else
			{
				if( !IsPo2(fftPlan->length[DimY])) pow2flag=false;
			}

		case CLFFT_1D:

			if( (fftPlan->length[DimX] == 1) && (dmnsn > 1) )
			{
				dmnsn -= 1;
				fftPlan-> inStride.erase(fftPlan-> inStride.begin());
				fftPlan->outStride.erase(fftPlan->outStride.begin());
				fftPlan->   length.erase(fftPlan->   length.begin());
			}
			else
			{
				if( !IsPo2(fftPlan->length[DimX])) pow2flag=false;
			}
		}

		fftPlan->dim = (clfftDim)dmnsn;
	}

	// first time check transposed
	if (fftPlan->transposed != CLFFT_NOTRANSPOSE && fftPlan->dim != CLFFT_2D &&
		fftPlan->dim == fftPlan->length.size())
		return CLFFT_TRANSPOSED_NOTIMPLEMENTED;

	//	The largest vector we can transform in a single pass
	//	depends on the GPU caps -- especially the amount of LDS
	//	available
	//
	size_t Large1DThreshold = 0;

	//First time check or see if LDS paramters are set-up.
	if (fftPlan->uLdsFraction == 0)
	{
		switch( fftPlan->dim )
		{
		case CLFFT_1D:
			{
				if (fftPlan->length[0] < 32768 || fftPlan->length[0] > 1048576)
					fftPlan->uLdsFraction = 8;
				else
					fftPlan->uLdsFraction = 4;

				if (fftPlan->length[0] < 1024 )
					fftPlan->bLdsComplex = true;
				else
					fftPlan->bLdsComplex = false;
			}
			break;
		case CLFFT_2D:
			{
				fftPlan->uLdsFraction = 4;
				fftPlan->bLdsComplex = false;
			}
			break;
		case CLFFT_3D:
			{
				//for case 128*128*128 and 1024*128*128, fraction = 8 is faster.
				fftPlan->uLdsFraction = 4;
				fftPlan->bLdsComplex = false;
			}
			break;
		}
	}
	OPENCL_V(fftPlan->GetMax1DLength (&Large1DThreshold), _T("GetMax1DLength failed"));
	BUG_CHECK (Large1DThreshold > 1);

	//	Verify that the data passed to us is packed
	switch( fftPlan->dim )
	{
	case CLFFT_1D:
		{
			if ( fftPlan->length[0] > Large1DThreshold )
			{
				size_t clLengths[] = { 1, 1, 0 };
				size_t in_1d, in_x, count;

				BUG_CHECK (IsPo2 (Large1DThreshold))
					//ARG_CHECK (IsPo2 (fftPlan->length[0]))

					// see whether large1D_Xfactor are fixed or not
					if (fftPlan->large1D_Xfactor == 0 )
					{
						if( IsPo2(fftPlan->length[0]) )
						{
							in_1d = BitScanF (Large1DThreshold);	// this is log2(LARGE1D_THRESHOLD)
							in_x  = BitScanF (fftPlan->length[0]);	// this is log2(length)
							BUG_CHECK (in_1d > 0)
								count = in_x/in_1d;
							if (count*in_1d < in_x)
							{
								count++;
								in_1d = in_x / count;
								if (in_1d * count < in_x) in_1d++;
							}
							clLengths[1] = (size_t)1 << in_1d;

						}
						else
						{
							// This array must be kept sorted in the ascending order
							size_t supported[] = {	1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40,
													45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135,
													144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288,
													300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480, 486, 500, 512, 540,
													576, 600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864, 900, 960,
													972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440,
													1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048,
													2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916, 3000,
													3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840, 3888, 4000, 4050, 4096 };

							size_t lenSupported = sizeof(supported)/sizeof(supported[0]);
							size_t maxFactoredLength = (supported[lenSupported-1] < Large1DThreshold) ? supported[lenSupported-1] : Large1DThreshold;

							size_t halfPowerLength = (size_t)1 << ( (StockhamGenerator::CeilPo2(fftPlan->length[0]) + 1) / 2 );
							size_t factoredLengthStart =  (halfPowerLength < maxFactoredLength) ? halfPowerLength : maxFactoredLength;

							size_t indexStart = 0;
							while(supported[indexStart] < factoredLengthStart) indexStart++;

							for(size_t i = indexStart; i >= 1; i--)
							{
								if( fftPlan->length[0] % supported[i] == 0 )
								{
									clLengths[1] = supported[i];
									break;
								}
							}
						}

						clLengths[0] = fftPlan->length[0]/clLengths[1];
					}
					else
					{
						//large1D_Xfactor will not pass to the second level of call
						clLengths[0] = fftPlan->large1D_Xfactor;
						clLengths[1] = fftPlan->length[0]/clLengths[0];
						ARG_CHECK (fftPlan->length[0] == clLengths[0] * clLengths[1]);
					}

					while (1 && (fftPlan->inputLayout != CLFFT_REAL) && (fftPlan->outputLayout != CLFFT_REAL))
					{
						if (!IsPo2(fftPlan->length[0])) break;
						//if (fftPlan->precision != CLFFT_SINGLE) break;
						//TBD, only one dimension?
						if (fftPlan->length.size() > 1) break;
						if (fftPlan->inStride[0] != 1 || fftPlan->outStride[0] != 1) break;
						//This length is good for using transpose
						if (fftPlan->length[0] < 131072) break;

						//first version not support huge1D, TBD
						if (clLengths[0] > Large1DThreshold) break;
						ARG_CHECK(clLengths[0]>=32 && clLengths[1]>=32);

						if (fftPlan->tmpBufSize==0 )
						{
							fftPlan->tmpBufSize = clLengths[0] * clLengths[1] *
								fftPlan->batchsize * fftPlan->ElementSize();
						}

						//Transpose
						//Input --> tmp buffer
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planTX, fftPlan->context, CLFFT_2D, clLengths ),
							_T( "CreateDefaultPlan Large1d transpose 1 failed" ) );

						FFTPlan* trans1Plan	= NULL;
						lockRAII* trans1Lock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planTX, trans1Plan, trans1Lock ), _T( "fftRepo.getPlan failed" ) );

						trans1Plan->placeness     = CLFFT_OUTOFPLACE;
						trans1Plan->precision     = fftPlan->precision;
						trans1Plan->tmpBufSize    = 0;
						trans1Plan->batchsize     = fftPlan->batchsize;
						trans1Plan->envelope	  = fftPlan->envelope;
						trans1Plan->inputLayout   = fftPlan->inputLayout;
						trans1Plan->outputLayout  = CLFFT_COMPLEX_INTERLEAVED;
						trans1Plan->inStride[0]   = fftPlan->inStride[0];
						trans1Plan->inStride[1]   = clLengths[0];
						trans1Plan->outStride[0]  = 1;
						trans1Plan->outStride[1]  = clLengths[1];
						trans1Plan->iDist         = fftPlan->iDist;
						trans1Plan->oDist         = fftPlan->length[0];
						trans1Plan->gen           = Transpose;
						trans1Plan->transflag     = true;

						OPENCL_V(clfftBakePlan(fftPlan->planTX, numQueues, commQueueFFT, NULL, NULL ),
							_T( "BakePlan large1d trans1 plan failed" ) );

						//Row transform
						//tmp->output
						//size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planX, fftPlan->context, CLFFT_1D, &clLengths[1] ),
							_T( "CreateDefaultPlan Large1d column failed" ) );

						FFTPlan* row1Plan	= NULL;
						lockRAII* row1Lock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planX, row1Plan, row1Lock ), _T( "fftRepo.getPlan failed" ) );

						row1Plan->placeness     = CLFFT_OUTOFPLACE;
						row1Plan->precision     = fftPlan->precision;
						row1Plan->forwardScale  = 1.0f;
						row1Plan->backwardScale = 1.0f;
						row1Plan->tmpBufSize    = 0;
						row1Plan->batchsize     = fftPlan->batchsize;
						row1Plan->bLdsComplex   = fftPlan->bLdsComplex;
						row1Plan->uLdsFraction  = fftPlan->uLdsFraction;
						row1Plan->ldsPadding    = fftPlan->ldsPadding;
						row1Plan->gen			= fftPlan->gen;
						row1Plan->envelope		= fftPlan->envelope;

						//Pass large1D flag to confirm we need multiply twiddle factor
						row1Plan->large1D       = fftPlan->length[0];

						row1Plan->length.push_back(clLengths[0]);
						row1Plan->inputLayout   = CLFFT_COMPLEX_INTERLEAVED;
						row1Plan->outputLayout  = fftPlan->outputLayout;
						row1Plan->inStride[0]   = 1;
						row1Plan->outStride[0]  = fftPlan->outStride[0];
						row1Plan->iDist         = fftPlan->length[0];
						row1Plan->oDist         = fftPlan->oDist;
						row1Plan->inStride.push_back(clLengths[1]);
						row1Plan->outStride.push_back(clLengths[1]);

						OPENCL_V(clfftBakePlan(fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ),
							_T( "BakePlan large1d first row plan failed" ) );

						//Transpose 2
						//Output --> tmp buffer
						clLengths[2] = clLengths[0];
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planTY, fftPlan->context, CLFFT_2D, &clLengths[1] ),
							_T( "CreateDefaultPlan Large1d transpose 1 failed" ) );

						FFTPlan* trans2Plan	= NULL;
						lockRAII* trans2Lock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planTY, trans2Plan, trans2Lock ), _T( "fftRepo.getPlan failed" ) );

						trans2Plan->placeness     = CLFFT_OUTOFPLACE;
						trans2Plan->precision     = fftPlan->precision;
						trans2Plan->tmpBufSize    = 0;
						trans2Plan->batchsize     = fftPlan->batchsize;
						trans2Plan->envelope	  = fftPlan->envelope;
						trans2Plan->inputLayout   = fftPlan->outputLayout;
						trans2Plan->outputLayout  = CLFFT_COMPLEX_INTERLEAVED;
						trans2Plan->inStride[0]   = fftPlan->outStride[0];
						trans2Plan->inStride[1]   = clLengths[1];
						trans2Plan->outStride[0]  = 1;
						trans2Plan->outStride[1]  = clLengths[0];
						trans2Plan->iDist         = fftPlan->oDist;
						trans2Plan->oDist         = fftPlan->length[0];
						trans2Plan->gen           = Transpose;
						trans2Plan->transflag     = true;

						OPENCL_V(clfftBakePlan(fftPlan->planTY, numQueues, commQueueFFT, NULL, NULL ),
							_T( "BakePlan large1d trans2 plan failed" ) );

						//Row transform 2
						//tmp->tmp
						//size clLengths[0], batch clLengths[1]
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planY, fftPlan->context, CLFFT_1D, &clLengths[0] ),
							_T( "CreateDefaultPlan Large1d column failed" ) );

						FFTPlan* row2Plan	= NULL;
						lockRAII* row2Lock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planY, row2Plan, row2Lock ), _T( "fftRepo.getPlan failed" ) );

						row2Plan->placeness     = CLFFT_INPLACE;
						row2Plan->precision     = fftPlan->precision;
						row2Plan->forwardScale  = fftPlan->forwardScale;
						row2Plan->backwardScale = fftPlan->backwardScale;
						row2Plan->tmpBufSize    = 0;
						row2Plan->batchsize     = fftPlan->batchsize;
						row2Plan->bLdsComplex   = fftPlan->bLdsComplex;
						row2Plan->uLdsFraction  = fftPlan->uLdsFraction;
						row2Plan->ldsPadding    = fftPlan->ldsPadding;
						row2Plan->gen			= fftPlan->gen;
						row2Plan->envelope		= fftPlan->envelope;

						//No twiddle factor is needed.
						row2Plan->large1D       = 0;

						row2Plan->length.push_back(clLengths[1]);
						row2Plan->inputLayout   = CLFFT_COMPLEX_INTERLEAVED;
						row2Plan->outputLayout  = CLFFT_COMPLEX_INTERLEAVED;
						row2Plan->inStride[0]   = 1;
						row2Plan->outStride[0]  = 1;
						row2Plan->iDist         = fftPlan->length[0];
						row2Plan->oDist         = fftPlan->length[0];
						row2Plan->inStride.push_back(clLengths[0]);
						row2Plan->outStride.push_back(clLengths[0]);

						OPENCL_V(clfftBakePlan(fftPlan->planY, numQueues, commQueueFFT, NULL, NULL ),
							_T( "BakePlan large1d first row plan failed" ) );

						//Transpose 3
						//tmp --> output
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planTZ, fftPlan->context, CLFFT_2D, clLengths ),
							_T( "CreateDefaultPlan Large1d transpose 1 failed" ) );

						FFTPlan* trans3Plan	= NULL;
						lockRAII* trans3Lock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planTZ, trans3Plan, trans3Lock ), _T( "fftRepo.getPlan failed" ) );

						trans3Plan->placeness     = CLFFT_OUTOFPLACE;
						trans3Plan->precision     = fftPlan->precision;
						trans3Plan->tmpBufSize    = 0;
						trans3Plan->batchsize     = fftPlan->batchsize;
						trans3Plan->envelope	  = fftPlan->envelope;
						trans3Plan->inputLayout   = CLFFT_COMPLEX_INTERLEAVED;
						trans3Plan->outputLayout  = fftPlan->outputLayout;
						trans3Plan->inStride[0]   = 1;
						trans3Plan->inStride[1]   = clLengths[0];
						trans3Plan->outStride[0]  = fftPlan->outStride[0];
						trans3Plan->outStride[1]  = clLengths[1];
						trans3Plan->iDist         = fftPlan->length[0];
						trans3Plan->oDist         = fftPlan->oDist;
						trans3Plan->gen           = Transpose;
						trans3Plan->transflag     = true;

						OPENCL_V(clfftBakePlan(fftPlan->planTZ, numQueues, commQueueFFT, NULL, NULL ),
							_T( "BakePlan large1d trans3 plan failed" ) );

						fftPlan->transflag = true;
						fftPlan->baked = true;
						return	CLFFT_SUCCESS;
					}

					size_t length0 = clLengths[0];
					size_t length1 = clLengths[1];

					if(fftPlan->inputLayout == CLFFT_REAL)
					{
						if (fftPlan->tmpBufSizeRC==0 )
						{
							fftPlan->tmpBufSizeRC = length0 * length1 *
								fftPlan->batchsize * fftPlan->ElementSize();
							for (size_t index=1; index < fftPlan->length.size(); index++)
							{
								fftPlan->tmpBufSizeRC *= fftPlan->length[index];
							}
						}

						// column FFT, size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
						// transposed output
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planX, fftPlan->context, CLFFT_1D, &clLengths[1] ),
							_T( "CreateDefaultPlan Large1d column failed" ) );

						FFTPlan* colTPlan	= NULL;
						lockRAII* colLock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planX, colTPlan, colLock ), _T( "fftRepo.getPlan failed" ) );

						// current plan is to create intermediate buffer, packed and interleave
						// This is a column FFT, the first elements distance between each FFT is the distance of the first two
						// elements in the original buffer. Like a transpose of the matrix
						// we need to pass clLengths[0] and instride size to kernel, so kernel can tell the difference

						//this part are common for both passes
						colTPlan->placeness     = CLFFT_OUTOFPLACE;
						colTPlan->precision     = fftPlan->precision;
						colTPlan->forwardScale  = 1.0f;
						colTPlan->backwardScale = 1.0f;
						colTPlan->tmpBufSize    = 0;
						colTPlan->batchsize     = fftPlan->batchsize;
						colTPlan->bLdsComplex   = fftPlan->bLdsComplex;
						colTPlan->uLdsFraction  = fftPlan->uLdsFraction;
						colTPlan->ldsPadding    = fftPlan->ldsPadding;
						colTPlan->gen			= fftPlan->gen;
						colTPlan->envelope			= fftPlan->envelope;

						//Pass large1D flag to confirm we need multiply twiddle factor
						colTPlan->large1D       = fftPlan->length[0];
						colTPlan->RCsimple		= true;

						colTPlan->length.push_back(clLengths[0]);

						// first Pass
						colTPlan->inputLayout   = fftPlan->inputLayout;
						colTPlan->outputLayout  = CLFFT_COMPLEX_INTERLEAVED;
						colTPlan->inStride[0]   = fftPlan->inStride[0] * clLengths[0];
						colTPlan->outStride[0]  = 1;
						colTPlan->iDist         = fftPlan->iDist;
						colTPlan->oDist         = length0 * length1;//fftPlan->length[0];
						colTPlan->inStride.push_back(fftPlan->inStride[0]);
						colTPlan->outStride.push_back(length1);//clLengths[1]);

						for (size_t index=1; index < fftPlan->length.size(); index++)
						{
							colTPlan->length.push_back(fftPlan->length[index]);
							colTPlan->inStride.push_back(fftPlan->inStride[index]);
							// tmp buffer is tightly packed
							colTPlan->outStride.push_back(colTPlan->oDist);
							colTPlan->oDist        *= fftPlan->length[index];
						}

						OPENCL_V(clfftBakePlan(fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan large1d first column plan failed" ) );

						//another column FFT, size clLengths[0], batch clLengths[1], output without transpose
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planY, fftPlan->context, CLFFT_1D,  &clLengths[0] ),
							_T( "CreateDefaultPlan large1D row failed" ) );

						FFTPlan* col2Plan	= NULL;
						lockRAII* rowLock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planY, col2Plan, rowLock ), _T( "fftRepo.getPlan failed" ) );

						// This is second column fft, intermediate buffer is packed and interleaved
						// we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference

						// common part for both passes
						col2Plan->placeness     = CLFFT_INPLACE;
						col2Plan->inputLayout   = CLFFT_COMPLEX_INTERLEAVED;
						col2Plan->outputLayout  = CLFFT_COMPLEX_INTERLEAVED;

						col2Plan->precision     = fftPlan->precision;
						col2Plan->forwardScale  = fftPlan->forwardScale;
						col2Plan->backwardScale = fftPlan->backwardScale;
						col2Plan->tmpBufSize    = 0;
						col2Plan->batchsize     = fftPlan->batchsize;
						col2Plan->bLdsComplex   = fftPlan->bLdsComplex;
						col2Plan->uLdsFraction  = fftPlan->uLdsFraction;
						col2Plan->ldsPadding    = fftPlan->ldsPadding;
						col2Plan->gen			= fftPlan->gen;
						col2Plan->envelope			= fftPlan->envelope;

						col2Plan->length.push_back(length1);

						col2Plan->inStride[0]  = length1;
						col2Plan->inStride.push_back(1);
						col2Plan->iDist        = length0 * length1;

						col2Plan->outStride[0] = length1;
						col2Plan->outStride.push_back(1);
						col2Plan->oDist         = length0 * length1;

						for (size_t index=1; index < fftPlan->length.size(); index++)
						{
							col2Plan->length.push_back(fftPlan->length[index]);
							col2Plan->inStride.push_back(col2Plan->iDist);
							col2Plan->outStride.push_back(col2Plan->oDist);
							col2Plan->iDist   *= fftPlan->length[index];
							col2Plan->oDist   *= fftPlan->length[index];
						}

						OPENCL_V(clfftBakePlan(fftPlan->planY, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan large1d second column plan failed" ) );


						// copy plan to get back to hermitian
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planRCcopy, fftPlan->context, CLFFT_1D,  &fftPlan->length[0] ),
							_T( "CreateDefaultPlan RC copy failed" ) );

						FFTPlan* copyPlan	= NULL;
						lockRAII* copyLock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planRCcopy, copyPlan, copyLock ), _T( "fftRepo.getPlan failed" ) );

						// This is second column fft, intermediate buffer is packed and interleaved
						// we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference

						// common part for both passes
						copyPlan->placeness     = CLFFT_OUTOFPLACE;
						copyPlan->inputLayout   = CLFFT_COMPLEX_INTERLEAVED;
						copyPlan->outputLayout  = fftPlan->outputLayout;

						copyPlan->precision     = fftPlan->precision;
						copyPlan->forwardScale  = 1.0f;
						copyPlan->backwardScale = 1.0f;
						copyPlan->tmpBufSize    = 0;
						copyPlan->batchsize     = fftPlan->batchsize;
						copyPlan->bLdsComplex   = fftPlan->bLdsComplex;
						copyPlan->uLdsFraction  = fftPlan->uLdsFraction;
						copyPlan->ldsPadding    = fftPlan->ldsPadding;
						copyPlan->gen			= Copy;
						copyPlan->envelope		= fftPlan->envelope;


						copyPlan->inStride[0]  = 1;
						copyPlan->iDist        = fftPlan->length[0];

						copyPlan->outStride[0] = fftPlan->outStride[0];
						copyPlan->oDist         = fftPlan->oDist;

						for (size_t index=1; index < fftPlan->length.size(); index++)
						{
							copyPlan->length.push_back(fftPlan->length[index]);
							copyPlan->inStride.push_back(copyPlan->inStride[index-1] * fftPlan->length[index-1]);
							copyPlan->iDist   *= fftPlan->length[index];
							copyPlan->outStride.push_back(fftPlan->outStride[index]);
						}

						OPENCL_V(clfftBakePlan(fftPlan->planRCcopy, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan large1d RC copy plan failed" ) );

					}
					else if(fftPlan->outputLayout == CLFFT_REAL)
					{
						if (fftPlan->tmpBufSizeRC==0 )
						{
							fftPlan->tmpBufSizeRC = length0 * length1 *
								fftPlan->batchsize * fftPlan->ElementSize();
							for (size_t index=1; index < fftPlan->length.size(); index++)
							{
								fftPlan->tmpBufSizeRC *= fftPlan->length[index];
							}
						}

						// copy plan to from hermitian to full complex
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planRCcopy, fftPlan->context, CLFFT_1D,  &fftPlan->length[0] ),
							_T( "CreateDefaultPlan RC copy failed" ) );

						FFTPlan* copyPlan	= NULL;
						lockRAII* copyLock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planRCcopy, copyPlan, copyLock ), _T( "fftRepo.getPlan failed" ) );

						// This is second column fft, intermediate buffer is packed and interleaved
						// we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference

						// common part for both passes
						copyPlan->placeness     = CLFFT_OUTOFPLACE;
						copyPlan->inputLayout   = fftPlan->inputLayout;
						copyPlan->outputLayout  = CLFFT_COMPLEX_INTERLEAVED;

						copyPlan->precision     = fftPlan->precision;
						copyPlan->forwardScale  = 1.0f;
						copyPlan->backwardScale = 1.0f;
						copyPlan->tmpBufSize    = 0;
						copyPlan->batchsize     = fftPlan->batchsize;
						copyPlan->bLdsComplex   = fftPlan->bLdsComplex;
						copyPlan->uLdsFraction  = fftPlan->uLdsFraction;
						copyPlan->ldsPadding    = fftPlan->ldsPadding;
						copyPlan->gen			= Copy;
						copyPlan->envelope		= fftPlan->envelope;

						copyPlan->inStride[0]  = fftPlan->inStride[0];
						copyPlan->iDist        = fftPlan->iDist;

						copyPlan->outStride[0]  = 1;
						copyPlan->oDist        = fftPlan->length[0];

						for (size_t index=1; index < fftPlan->length.size(); index++)
						{
							copyPlan->length.push_back(fftPlan->length[index]);
							copyPlan->outStride.push_back(copyPlan->outStride[index-1] * fftPlan->length[index-1]);
							copyPlan->oDist   *= fftPlan->length[index];
							copyPlan->inStride.push_back(fftPlan->inStride[index]);
						}

						OPENCL_V(clfftBakePlan(fftPlan->planRCcopy, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan large1d RC copy plan failed" ) );

						// column FFT, size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
						// transposed output
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planX, fftPlan->context, CLFFT_1D, &clLengths[1] ),
							_T( "CreateDefaultPlan Large1d column failed" ) );

						FFTPlan* colTPlan	= NULL;
						lockRAII* colLock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planX, colTPlan, colLock ), _T( "fftRepo.getPlan failed" ) );

						// current plan is to create intermediate buffer, packed and interleave
						// This is a column FFT, the first elements distance between each FFT is the distance of the first two
						// elements in the original buffer. Like a transpose of the matrix
						// we need to pass clLengths[0] and instride size to kernel, so kernel can tell the difference

						//this part are common for both passes
						colTPlan->placeness     = CLFFT_INPLACE;
						colTPlan->precision     = fftPlan->precision;
						colTPlan->forwardScale  = 1.0f;
						colTPlan->backwardScale = 1.0f;
						colTPlan->tmpBufSize    = 0;
						colTPlan->batchsize     = fftPlan->batchsize;
						colTPlan->bLdsComplex   = fftPlan->bLdsComplex;
						colTPlan->uLdsFraction  = fftPlan->uLdsFraction;
						colTPlan->ldsPadding    = fftPlan->ldsPadding;
						colTPlan->gen			= fftPlan->gen;
						colTPlan->envelope			= fftPlan->envelope;

						//Pass large1D flag to confirm we need multiply twiddle factor
						colTPlan->large1D       = fftPlan->length[0];

						colTPlan->length.push_back(clLengths[0]);

						// first Pass
						colTPlan->inputLayout   = CLFFT_COMPLEX_INTERLEAVED;
						colTPlan->outputLayout  = CLFFT_COMPLEX_INTERLEAVED;


						colTPlan->inStride[0]  = length0;
						colTPlan->inStride.push_back(1);
						colTPlan->iDist        = length0 * length1;

						colTPlan->outStride[0] = length0;
						colTPlan->outStride.push_back(1);
						colTPlan->oDist         = length0 * length1;

						for (size_t index=1; index < fftPlan->length.size(); index++)
						{
							colTPlan->length.push_back(fftPlan->length[index]);
							colTPlan->inStride.push_back(colTPlan->iDist);
							colTPlan->outStride.push_back(colTPlan->oDist);
							colTPlan->iDist   *= fftPlan->length[index];
							colTPlan->oDist   *= fftPlan->length[index];
						}


						OPENCL_V(clfftBakePlan(fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan large1d first column plan failed" ) );

						//another column FFT, size clLengths[0], batch clLengths[1], output without transpose
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planY, fftPlan->context, CLFFT_1D,  &clLengths[0] ),
							_T( "CreateDefaultPlan large1D row failed" ) );

						FFTPlan* col2Plan	= NULL;
						lockRAII* rowLock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planY, col2Plan, rowLock ), _T( "fftRepo.getPlan failed" ) );

						// This is second column fft, intermediate buffer is packed and interleaved
						// we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference

						// common part for both passes
						col2Plan->placeness     = CLFFT_OUTOFPLACE;
						col2Plan->inputLayout   = CLFFT_COMPLEX_INTERLEAVED;
						col2Plan->outputLayout  = fftPlan->outputLayout;

						col2Plan->precision     = fftPlan->precision;
						col2Plan->forwardScale  = fftPlan->forwardScale;
						col2Plan->backwardScale = fftPlan->backwardScale;
						col2Plan->tmpBufSize    = 0;
						col2Plan->batchsize     = fftPlan->batchsize;
						col2Plan->bLdsComplex   = fftPlan->bLdsComplex;
						col2Plan->uLdsFraction  = fftPlan->uLdsFraction;
						col2Plan->ldsPadding    = fftPlan->ldsPadding;
						col2Plan->gen			= fftPlan->gen;
						col2Plan->envelope			= fftPlan->envelope;

						col2Plan->RCsimple = true;
						col2Plan->length.push_back(length1);

						col2Plan->inStride[0]  = 1;
						col2Plan->inStride.push_back(length0);
						col2Plan->iDist        = length0 * length1;

						col2Plan->outStride[0] = length1 * fftPlan->outStride[0];
						col2Plan->outStride.push_back(fftPlan->outStride[0]);
						col2Plan->oDist         = fftPlan->oDist;

						for (size_t index=1; index < fftPlan->length.size(); index++)
						{
							col2Plan->length.push_back(fftPlan->length[index]);
							col2Plan->inStride.push_back(col2Plan->iDist);
							col2Plan->iDist   *= fftPlan->length[index];
							col2Plan->outStride.push_back(fftPlan->outStride[index]);
						}

						OPENCL_V(clfftBakePlan(fftPlan->planY, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan large1d second column plan failed" ) );
					}
					else
					{
						if (fftPlan->cacheSize) {
							length0 += fftPlan->cacheSize & 0xFF;
							length1 += (fftPlan->cacheSize >> 8) & 0xFF;
							if (length0 * length1 > 2 * fftPlan->length[0])
							{
								length0 = clLengths[0];
								length1 = clLengths[1];
							}
						}
						else
						{
							if (fftPlan->length[0] == 131072) length1 += 1;     //x0=0, y0=1 good for Cayman card
							else if (fftPlan->length[0] == 65536) length1 += 8; //x0=0, y0=8 good for Cypress card
						}

						if (clLengths[0] > Large1DThreshold)
						{//make no change for Huge 1D case
							length0 = clLengths[0];
							length1 = clLengths[1];
						}

						if (fftPlan->tmpBufSize==0 )
						{
							fftPlan->tmpBufSize = length0 * length1 *
								fftPlan->batchsize * fftPlan->ElementSize();
							for (size_t index=1; index < fftPlan->length.size(); index++)
							{
								fftPlan->tmpBufSize *= fftPlan->length[index];
							}
						}
						else
						{//make no change for cases passed from higher dimension
							length0 = clLengths[0];
							length1 = clLengths[1];
						}

						// column FFT, size clLengths[1], batch clLengths[0], with length[0] twiddle factor multiplication
						// transposed output
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planX, fftPlan->context, CLFFT_1D, &clLengths[1] ),
							_T( "CreateDefaultPlan Large1d column failed" ) );

						FFTPlan* colTPlan	= NULL;
						lockRAII* colLock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planX, colTPlan, colLock ), _T( "fftRepo.getPlan failed" ) );

						// current plan is to create intermediate buffer, packed and interleave
						// This is a column FFT, the first elements distance between each FFT is the distance of the first two
						// elements in the original buffer. Like a transpose of the matrix
						// we need to pass clLengths[0] and instride size to kernel, so kernel can tell the difference

						//this part are common for both passes
						colTPlan->placeness     = CLFFT_OUTOFPLACE;
						colTPlan->precision     = fftPlan->precision;
						colTPlan->forwardScale  = 1.0f;
						colTPlan->backwardScale = 1.0f;
						colTPlan->tmpBufSize    = 0;
						colTPlan->batchsize     = fftPlan->batchsize;
						colTPlan->bLdsComplex   = fftPlan->bLdsComplex;
						colTPlan->uLdsFraction  = fftPlan->uLdsFraction;
						colTPlan->ldsPadding    = fftPlan->ldsPadding;
						colTPlan->gen			= fftPlan->gen;
						colTPlan->envelope			= fftPlan->envelope;

						//Pass large1D flag to confirm we need multiply twiddle factor
						colTPlan->large1D       = fftPlan->length[0];

						colTPlan->length.push_back(clLengths[0]);

						if (fftPlan->large1D == 0)
						{
							// first Pass
							colTPlan->inputLayout   = fftPlan->inputLayout;
							colTPlan->outputLayout  = CLFFT_COMPLEX_INTERLEAVED;
							colTPlan->inStride[0]   = fftPlan->inStride[0] * clLengths[0];
							colTPlan->outStride[0]  = 1;
							colTPlan->iDist         = fftPlan->iDist;
							colTPlan->oDist         = length0 * length1;//fftPlan->length[0];
							colTPlan->inStride.push_back(fftPlan->inStride[0]);
							colTPlan->outStride.push_back(length1);//clLengths[1]);

							for (size_t index=1; index < fftPlan->length.size(); index++)
							{
								colTPlan->length.push_back(fftPlan->length[index]);
								colTPlan->inStride.push_back(fftPlan->inStride[index]);
								// tmp buffer is tightly packed
								colTPlan->outStride.push_back(colTPlan->oDist);
								colTPlan->oDist        *= fftPlan->length[index];
							}
						}
						else
						{
							// second pass for huge 1D
							colTPlan->inputLayout   = CLFFT_COMPLEX_INTERLEAVED;
							colTPlan->outputLayout  = fftPlan->outputLayout;
							colTPlan->inStride[0]   = fftPlan->length[1]*clLengths[0];
							colTPlan->outStride[0]  = fftPlan->outStride[0];
							colTPlan->iDist         = fftPlan->length[0];
							colTPlan->oDist         = fftPlan->oDist;
							colTPlan->inStride.push_back(fftPlan->length[1]);
							colTPlan->outStride.push_back(fftPlan->outStride[0]*clLengths[1]);

							for (size_t index=1; index < fftPlan->length.size(); index++)
							{
								colTPlan->length.push_back(fftPlan->length[index]);
								colTPlan->inStride.push_back(fftPlan->inStride[index]);
								colTPlan->outStride.push_back(fftPlan->outStride[index]);
								colTPlan->iDist        *= fftPlan->length[index];
							}
						}

						OPENCL_V(clfftBakePlan(fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan large1d first column plan failed" ) );

						//another column FFT, size clLengths[0], batch clLengths[1], output without transpose
						OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planY, fftPlan->context, CLFFT_1D,  &clLengths[0] ),
							_T( "CreateDefaultPlan large1D row failed" ) );

						FFTPlan* col2Plan	= NULL;
						lockRAII* rowLock	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planY, col2Plan, rowLock ), _T( "fftRepo.getPlan failed" ) );

						// This is second column fft, intermediate buffer is packed and interleaved
						// we need to pass clLengths[1] and instride size to kernel, so kernel can tell the difference

						// common part for both passes
						col2Plan->outputLayout  = fftPlan->outputLayout;
						col2Plan->precision     = fftPlan->precision;
						col2Plan->forwardScale  = fftPlan->forwardScale;
						col2Plan->backwardScale = fftPlan->backwardScale;
						col2Plan->tmpBufSize    = 0;
						col2Plan->batchsize     = fftPlan->batchsize;
						col2Plan->oDist         = fftPlan->oDist;
						col2Plan->bLdsComplex   = fftPlan->bLdsComplex;
						col2Plan->uLdsFraction  = fftPlan->uLdsFraction;
						col2Plan->ldsPadding    = fftPlan->ldsPadding;
						col2Plan->gen			= fftPlan->gen;
						col2Plan->envelope			= fftPlan->envelope;

						if (clLengths[0] > Large1DThreshold)
							//prepare for huge 1D
							col2Plan->large1D   = fftPlan->length[0];

						col2Plan->length.push_back(clLengths[1]);
						col2Plan->outStride.push_back(fftPlan->outStride[0]);

						if (fftPlan->large1D == 0)
						{
							//first layer, large 1D from tmp buffer to output buffer
							col2Plan->placeness    = CLFFT_OUTOFPLACE;
							col2Plan->inputLayout  = CLFFT_COMPLEX_INTERLEAVED;
							col2Plan->inStride[0]  = length1;//clLengths[1];
							col2Plan->outStride[0] = fftPlan->outStride[0] * clLengths[1];
							col2Plan->iDist        = length0 * length1; //fftPlan->length[0];
							col2Plan->inStride.push_back(1);

							for (size_t index=1; index < fftPlan->length.size(); index++)
							{
								col2Plan->length.push_back(fftPlan->length[index]);
								col2Plan->inStride.push_back(col2Plan->iDist);
								col2Plan->outStride.push_back(fftPlan->outStride[index]);
								col2Plan->iDist   *= fftPlan->length[index];
							}
						}
						else
						{
							//second layer, huge 1D from output buffer to output buffer
							col2Plan->placeness    = CLFFT_INPLACE;
							col2Plan->inputLayout  = fftPlan->outputLayout;
							col2Plan->inStride[0]  = fftPlan->outStride[0] * clLengths[1];
							col2Plan->outStride[0] = col2Plan->inStride[0];
							col2Plan->iDist        = fftPlan->oDist;
							col2Plan->inStride.push_back(fftPlan->outStride[0]);

							for (size_t index=1; index < fftPlan->length.size(); index++)
							{
								col2Plan->length.push_back(fftPlan->length[index]);
								col2Plan->inStride.push_back(fftPlan->outStride[index]);
								col2Plan->outStride.push_back(fftPlan->outStride[index]);
							}
						}

						OPENCL_V(clfftBakePlan(fftPlan->planY, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan large1d second column plan failed" ) );
					}

					fftPlan->baked = true;
					return	CLFFT_SUCCESS;
			}
		}
		break;
	case CLFFT_2D:
		{
			size_t length0 = fftPlan->length[0];
			size_t length1 = fftPlan->length[1];


			if (fftPlan->cacheSize)
			{
				length0 += fftPlan->cacheSize & 0xFF;
				length1 += (fftPlan->cacheSize >> 8) & 0xFF;
				if (length0 * length1 > 2 * fftPlan->length[0] * fftPlan->length[1])
				{
					length0 = fftPlan->length[0];
					length1 = fftPlan->length[1];
				}
			}
			else
			{
				if (fftPlan->length[0]==256 && fftPlan->length[1]==256)
				{
					length0 += 8;
					length1 += 1;
				}
				else if (fftPlan->length[0]==512 && fftPlan->length[1]==512)
				{
					length0 += 1;
					length1 += 1;//length1 += 0;
				}
				else if (fftPlan->length[0]==1024 && fftPlan->length[1]==512)
				{
					length0 += 2;
					length1 += 2;//length1 += 0;
				}
				else if (fftPlan->length[0]==1024 && fftPlan->length[1]==1024)
				{
					length0 += 1;
					length1 += 1;//length1 += 0;
				}
			}

			if (fftPlan->length[0] > Large1DThreshold ||
				fftPlan->length[1] > Large1DThreshold)
				fftPlan->large2D = true;

			while (1 && (fftPlan->inputLayout != CLFFT_REAL) && (fftPlan->outputLayout != CLFFT_REAL))
			{
				//break;
				if (fftPlan->transflag) //Transpose for 2D
				{
					OPENCL_V( fftPlan->GenerateKernel( fftRepo, *commQueueFFT ), _T( "GenerateTransposeProgram() failed" ) );
					OPENCL_V( CompileKernels( *commQueueFFT, plHandle, fftPlan->gen, fftPlan ), _T( "CompileKernels() failed" ) );

					fftPlan->baked		= true;
					return	CLFFT_SUCCESS;
				}

				if (fftPlan->length.size() != 2) break;
				if (!(IsPo2(fftPlan->length[0])) || !(IsPo2(fftPlan->length[1])))
					break;
				if (fftPlan->length[1] < 32) break;
				//TBD: restrict the use large2D in x!=y case becase we will need two temp buffers
				//     (1) for 2D usage (2) for 1D large usage
				//if (fftPlan->large2D) break;
				//Performance show 512 is the good case with transpose
				//if user want the result to be transposed, then we will.
				if (fftPlan->length[0] < 512 && fftPlan->transposed == CLFFT_NOTRANSPOSE) break;
				if (fftPlan->length[0] < 32) break;
				//x!=y case, we need tmp buffer, currently temp buffer only support interleaved format
				//if (fftPlan->length[0] != fftPlan->length[1] && fftPlan->outputLayout == CLFFT_COMPLEX_PLANAR) break;
				if (fftPlan->inStride[0] != 1 || fftPlan->outStride[0] != 1 ||
					fftPlan->inStride[1] != fftPlan->length[0] || fftPlan->outStride[1] != fftPlan->length[0])
					break;
				//if (fftPlan->placeness != CLFFT_INPLACE || fftPlan->inputLayout != CLFFT_COMPLEX_PLANAR)
				//	break;
				//if (fftPlan->batchsize != 1) break;
				//if (fftPlan->precision != CLFFT_SINGLE) break;

				fftPlan->transflag = true;

				//create row plan,
				// x=y & x!=y, In->In for inplace, In->out for outofplace
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planX, fftPlan->context, CLFFT_1D, &fftPlan->length[ DimX ] ),
					_T( "CreateDefaultPlan for planX failed" ) );

				FFTPlan* rowPlan	= NULL;
				lockRAII* rowLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock ), _T( "fftRepo.getPlan failed" ) );

				rowPlan->inputLayout     = fftPlan->inputLayout;
				rowPlan->outputLayout    = fftPlan->outputLayout;
				rowPlan->placeness       = fftPlan->placeness;
				rowPlan->outStride[0]    = fftPlan->outStride[0];
				rowPlan->outStride.push_back(fftPlan->outStride[1]);
				rowPlan->oDist           = fftPlan->oDist;
				rowPlan->precision       = fftPlan->precision;
				rowPlan->forwardScale    = 1.0f;
				rowPlan->backwardScale   = 1.0f;
				rowPlan->tmpBufSize      = 0;
				rowPlan->bLdsComplex     = fftPlan->bLdsComplex;
				rowPlan->uLdsFraction    = fftPlan->uLdsFraction;
				rowPlan->ldsPadding      = fftPlan->ldsPadding;
				rowPlan->gen			 = fftPlan->gen;
				rowPlan->envelope		 = fftPlan->envelope;
				rowPlan->batchsize       = fftPlan->batchsize;
				rowPlan->inStride[0]     = fftPlan->inStride[0];
				rowPlan->length.push_back(fftPlan->length[1]);
				rowPlan->inStride.push_back(fftPlan->inStride[1]);
				rowPlan->iDist           = fftPlan->iDist;

				OPENCL_V(clfftBakePlan(fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ),
					_T( "BakePlan for planX failed" ) );

				//Create transpose plan for first transpose
				//x=y: inplace. x!=y inplace: in->tmp, outofplace out->tmp
				size_t clLengths[] = { 1, 1, 0 };
				clLengths[0] = fftPlan->length[0];
				clLengths[1] = fftPlan->length[1];

				bool xyflag = (clLengths[0]==clLengths[1]) ? false : true;
				if (xyflag && fftPlan->tmpBufSize==0 && fftPlan->length.size()<=2)
				{
					// we need tmp buffer for x!=y case
					// we assume the tmp buffer is packed interleaved
					fftPlan->tmpBufSize = length0 * length1 *
						fftPlan->batchsize * fftPlan->ElementSize();
				}

				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planTX, fftPlan->context, CLFFT_2D, clLengths ),
					_T( "CreateDefaultPlan for planT failed" ) );

				FFTPlan* transPlanX	= NULL;
				lockRAII* transLockX	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planTX, transPlanX, transLockX ), _T( "fftRepo.getPlan failed" ) );

				transPlanX->inputLayout     = fftPlan->outputLayout;
				transPlanX->precision       = fftPlan->precision;
				transPlanX->tmpBufSize      = 0;
				transPlanX->gen			    = Transpose;
				transPlanX->envelope		= fftPlan->envelope;
				transPlanX->batchsize       = fftPlan->batchsize;
				transPlanX->inStride[0]     = fftPlan->outStride[0];
				transPlanX->inStride[1]     = fftPlan->outStride[1];
				transPlanX->iDist           = fftPlan->oDist;
				transPlanX->transflag       = true;

				if (xyflag)
				{
					transPlanX->outputLayout    = CLFFT_COMPLEX_INTERLEAVED;
					transPlanX->placeness       = CLFFT_OUTOFPLACE;
					transPlanX->outStride[0]    = 1;
					transPlanX->outStride[1]    = clLengths[0];
					transPlanX->oDist           = clLengths[0] * clLengths[1];
				}
				else
				{
					transPlanX->outputLayout    = fftPlan->outputLayout;
					transPlanX->placeness       = CLFFT_INPLACE;
					transPlanX->outStride[0]    = fftPlan->outStride[0];
					transPlanX->outStride[1]    = fftPlan->outStride[1];
					transPlanX->oDist           = fftPlan->oDist;
				}

				OPENCL_V(clfftBakePlan(fftPlan->planTX, numQueues, commQueueFFT, NULL, NULL ),
					_T( "BakePlan for planTX failed" ) );

				//create second row plan
				//x!=y: tmp->tmp, x=y case: In->In or Out->Out
				//if Transposed result is a choice x!=y: tmp->In or out
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planY, fftPlan->context, CLFFT_1D, &fftPlan->length[ DimY ] ),
					_T( "CreateDefaultPlan for planY failed" ) );

				FFTPlan* colPlan	= NULL;
				lockRAII* colLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planY, colPlan, colLock ), _T( "fftRepo.getPlan failed" ) );

				if (xyflag)
				{
					colPlan->inputLayout     = CLFFT_COMPLEX_INTERLEAVED;
					colPlan->inStride[0]     = 1;
					colPlan->inStride.push_back(clLengths[1]);
					colPlan->iDist           = clLengths[0] * clLengths[1];

					if (fftPlan->transposed == CLFFT_NOTRANSPOSE)
					{
						colPlan->outputLayout    = CLFFT_COMPLEX_INTERLEAVED;
						colPlan->outStride[0]    = 1;
						colPlan->outStride.push_back(clLengths[1]);
						colPlan->oDist           = clLengths[0] * clLengths[1];
						colPlan->placeness       = CLFFT_INPLACE;
					}
					else
					{
						colPlan->outputLayout    = fftPlan->outputLayout;
						colPlan->outStride[0]    = fftPlan->outStride[0];
						colPlan->outStride.push_back(clLengths[1] * fftPlan->outStride[0]);
						colPlan->oDist           = fftPlan->oDist;
						colPlan->placeness       = CLFFT_OUTOFPLACE;
					}
				}
				else
				{
					colPlan->inputLayout     = fftPlan->outputLayout;
					colPlan->outputLayout    = fftPlan->outputLayout;
					colPlan->outStride[0]    = fftPlan->outStride[0];
					colPlan->outStride.push_back(fftPlan->outStride[1]);
					colPlan->oDist           = fftPlan->oDist;
					colPlan->inStride[0]     = fftPlan->outStride[0];
					colPlan->inStride.push_back(fftPlan->outStride[1]);
					colPlan->iDist           = fftPlan->oDist;
					colPlan->placeness       = CLFFT_INPLACE;
				}

				colPlan->precision       = fftPlan->precision;
				colPlan->forwardScale    = fftPlan->forwardScale;
				colPlan->backwardScale   = fftPlan->backwardScale;
				colPlan->tmpBufSize      = 0;
				colPlan->bLdsComplex     = fftPlan->bLdsComplex;
				colPlan->uLdsFraction    = fftPlan->uLdsFraction;
				colPlan->ldsPadding      = fftPlan->ldsPadding;
				colPlan->gen			 = fftPlan->gen;
				colPlan->envelope		 = fftPlan->envelope;
				colPlan->batchsize       = fftPlan->batchsize;
				colPlan->length.push_back(fftPlan->length[0]);

				OPENCL_V(clfftBakePlan(fftPlan->planY, numQueues, commQueueFFT, NULL, NULL ),
					_T( "BakePlan for planY failed" ) );

				if (fftPlan->transposed == CLFFT_TRANSPOSED)
				{
					fftPlan->baked = true;
					return	CLFFT_SUCCESS;
				}

				//Create transpose plan for second transpose
				//x!=y case tmp->In or Out, x=y case In->In or Out->out
				clLengths[0] = fftPlan->length[1];
				clLengths[1] = fftPlan->length[0];
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planTY, fftPlan->context, CLFFT_2D, clLengths ),
					_T( "CreateDefaultPlan for planTY failed" ) );

				FFTPlan* transPlanY	= NULL;
				lockRAII* transLockY	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planTY, transPlanY, transLockY ), _T( "fftRepo.getPlan failed" ) );

				if (xyflag)
				{
					transPlanY->inputLayout     = CLFFT_COMPLEX_INTERLEAVED;
					transPlanY->placeness       = CLFFT_OUTOFPLACE;
					transPlanY->inStride[0]     = 1;
					transPlanY->inStride[1]     = clLengths[0];
					transPlanY->iDist           = clLengths[0] * clLengths[1];
				}
				else
				{
					transPlanY->inputLayout     = fftPlan->outputLayout;
					transPlanY->placeness       = CLFFT_INPLACE;
					transPlanY->inStride[0]     = fftPlan->outStride[0];
					transPlanY->inStride[1]     = fftPlan->outStride[1];
					transPlanY->iDist           = fftPlan->oDist;
				}
				transPlanY->outputLayout    = fftPlan->outputLayout;
				transPlanY->outStride[0]    = fftPlan->outStride[0];
				transPlanY->outStride[1]    = fftPlan->outStride[1];
				transPlanY->oDist           = fftPlan->oDist;
				transPlanY->precision       = fftPlan->precision;
				transPlanY->tmpBufSize      = 0;
				transPlanY->gen			    = Transpose;
				transPlanY->envelope		= fftPlan->envelope;
				transPlanY->batchsize       = fftPlan->batchsize;
				transPlanY->transflag       = true;

				OPENCL_V(clfftBakePlan(fftPlan->planTY, numQueues, commQueueFFT, NULL, NULL ),
					_T( "BakePlan for planTY failed" ) );

				fftPlan->baked = true;
				return	CLFFT_SUCCESS;
			}

			//check transposed
			if (fftPlan->transposed != CLFFT_NOTRANSPOSE)
				return CLFFT_TRANSPOSED_NOTIMPLEMENTED;


			if(fftPlan->inputLayout == CLFFT_REAL)
			{
				length0 = fftPlan->length[0];
				length1 = fftPlan->length[1];

				size_t Nt = (1 + length0/2);
				if (fftPlan->tmpBufSize==0)
				{
					fftPlan->tmpBufSize = Nt * length1 * fftPlan->batchsize * fftPlan->ElementSize();
					if(fftPlan->length.size() > 2) fftPlan->tmpBufSize *= fftPlan->length[2];
				}

				// create row plan
				// real to hermitian

				//create row plan
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planX, fftPlan->context, CLFFT_1D, &fftPlan->length[ DimX ] ),
					_T( "CreateDefaultPlan for planX failed" ) );

				FFTPlan* rowPlan	= NULL;
				lockRAII* rowLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock ), _T( "fftRepo.getPlan failed" ) );


				rowPlan->outputLayout  = fftPlan->outputLayout;
				rowPlan->inputLayout  = fftPlan->inputLayout;
				rowPlan->placeness     = fftPlan->placeness;
				rowPlan->length.push_back(length1);

				rowPlan->inStride[0]  = fftPlan->inStride[0];
				rowPlan->inStride.push_back(fftPlan->inStride[1]);
				rowPlan->iDist         = fftPlan->iDist;

				rowPlan->precision     = fftPlan->precision;
				rowPlan->forwardScale  = 1.0f;
				rowPlan->backwardScale = 1.0f;
				rowPlan->tmpBufSize    = fftPlan->tmpBufSize;
				rowPlan->bLdsComplex   = fftPlan->bLdsComplex;
				rowPlan->uLdsFraction  = fftPlan->uLdsFraction;
				rowPlan->ldsPadding    = fftPlan->ldsPadding;
				rowPlan->gen			= fftPlan->gen;
				rowPlan->envelope			= fftPlan->envelope;

				rowPlan->batchsize    = fftPlan->batchsize;

				rowPlan->outStride[0]  = fftPlan->outStride[0];
				rowPlan->outStride.push_back(fftPlan->outStride[1]);
				rowPlan->oDist         = fftPlan->oDist;

				//this 2d is decomposed from 3d
				if (fftPlan->length.size()>2)
				{
					rowPlan->length.push_back(fftPlan->length[2]);
					rowPlan->inStride.push_back(fftPlan->inStride[2]);
					rowPlan->outStride.push_back(fftPlan->outStride[2]);
				}


				OPENCL_V(clfftBakePlan(fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan for planX failed" ) );

				// create col plan
				// complex to complex

				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planY, fftPlan->context, CLFFT_1D, &fftPlan->length[ DimY ] ),
					_T( "CreateDefaultPlan for planY failed" ) );

				FFTPlan* colPlan	= NULL;
				lockRAII* colLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planY, colPlan, colLock ), _T( "fftRepo.getPlan failed" ) );

				switch(fftPlan->outputLayout)
				{
				case CLFFT_HERMITIAN_INTERLEAVED:
					{
						colPlan->outputLayout = CLFFT_COMPLEX_INTERLEAVED;
						colPlan->inputLayout  = CLFFT_COMPLEX_INTERLEAVED;
					}
					break;
				case CLFFT_HERMITIAN_PLANAR:
					{
						colPlan->outputLayout = CLFFT_COMPLEX_PLANAR;
						colPlan->inputLayout  = CLFFT_COMPLEX_PLANAR;
					}
					break;
				default: assert(false);
				}

				colPlan->placeness     = CLFFT_INPLACE;
				colPlan->length.push_back(Nt);

				colPlan->outStride[0]  = fftPlan->outStride[1];
				colPlan->outStride.push_back(fftPlan->outStride[0]);
				colPlan->oDist         = fftPlan->oDist;


				colPlan->precision     = fftPlan->precision;
				colPlan->forwardScale  = fftPlan->forwardScale;
				colPlan->backwardScale = fftPlan->backwardScale;
				colPlan->tmpBufSize    = fftPlan->tmpBufSize;
				colPlan->bLdsComplex   = fftPlan->bLdsComplex;
				colPlan->uLdsFraction  = fftPlan->uLdsFraction;
				colPlan->ldsPadding    = fftPlan->ldsPadding;
				colPlan->gen			= fftPlan->gen;
				colPlan->envelope			= fftPlan->envelope;

				colPlan->batchsize = fftPlan->batchsize;

				colPlan->inStride[0]  = rowPlan->outStride[1];
				colPlan->inStride.push_back(rowPlan->outStride[0]);
				colPlan->iDist         = rowPlan->oDist;

				//this 2d is decomposed from 3d
				if (fftPlan->length.size()>2)
				{
					colPlan->length.push_back(fftPlan->length[2]);
					colPlan->outStride.push_back(fftPlan->outStride[2]);
					colPlan->inStride.push_back(rowPlan->outStride[2]);
				}

				OPENCL_V(clfftBakePlan(fftPlan->planY, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan for planY failed" ) );

			}
			else if(fftPlan->outputLayout == CLFFT_REAL)
			{
				length0 = fftPlan->length[0];
				length1 = fftPlan->length[1];

				size_t Nt = (1 + length0/2);
				if (fftPlan->tmpBufSize==0)
				{
					fftPlan->tmpBufSize = Nt * length1 * fftPlan->batchsize * fftPlan->ElementSize();
					if(fftPlan->length.size() > 2) fftPlan->tmpBufSize *= fftPlan->length[2];
				}

				// create col plan
				// complex to complex

				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planY, fftPlan->context, CLFFT_1D, &fftPlan->length[ DimY ] ),
					_T( "CreateDefaultPlan for planY failed" ) );

				FFTPlan* colPlan	= NULL;
				lockRAII* colLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planY, colPlan, colLock ), _T( "fftRepo.getPlan failed" ) );

				colPlan->length.push_back(Nt);

				colPlan->inStride[0]  = fftPlan->inStride[1];
				colPlan->inStride.push_back(fftPlan->inStride[0]);
				colPlan->iDist         = fftPlan->iDist;


				//this 2d is decomposed from 3d
				if (fftPlan->length.size()>2)
				{
					colPlan->placeness = CLFFT_INPLACE;

					colPlan->length.push_back(fftPlan->length[2]);
					colPlan->inStride.push_back(fftPlan->inStride[2]);
					colPlan->outStride[0]  = colPlan->inStride[0];
					colPlan->outStride.push_back(colPlan->inStride[1]);
					colPlan->outStride.push_back(colPlan->inStride[2]);
					colPlan->oDist         = fftPlan->iDist;
				}
				else
				{
					colPlan->placeness = CLFFT_OUTOFPLACE;

					colPlan->outStride[0]  = Nt;
					colPlan->outStride.push_back(1);
					colPlan->oDist         = Nt*length1;
				}


				switch(fftPlan->inputLayout)
				{
				case CLFFT_HERMITIAN_INTERLEAVED:
					{
						colPlan->outputLayout = CLFFT_COMPLEX_INTERLEAVED;
						colPlan->inputLayout  = CLFFT_COMPLEX_INTERLEAVED;
					}
					break;
				case CLFFT_HERMITIAN_PLANAR:
					{
						colPlan->outputLayout = CLFFT_COMPLEX_INTERLEAVED;
						colPlan->inputLayout  = CLFFT_COMPLEX_PLANAR;
					}
					break;
				default: assert(false);
				}


				colPlan->precision     = fftPlan->precision;
				colPlan->forwardScale  = 1.0f;
				colPlan->backwardScale = 1.0f;
				colPlan->tmpBufSize    = fftPlan->tmpBufSize;
				colPlan->bLdsComplex   = fftPlan->bLdsComplex;
				colPlan->uLdsFraction  = fftPlan->uLdsFraction;
				colPlan->ldsPadding    = fftPlan->ldsPadding;
				colPlan->gen			= fftPlan->gen;
				colPlan->envelope			= fftPlan->envelope;

				colPlan->batchsize = fftPlan->batchsize;

				if ((fftPlan->tmpBufSizeC2R==0) && (length1 > Large1DThreshold) && (fftPlan->length.size()<=2))
				{
					fftPlan->tmpBufSizeC2R = Nt * length1 * fftPlan->batchsize * fftPlan->ElementSize();
					if(fftPlan->length.size() > 2) fftPlan->tmpBufSizeC2R *= fftPlan->length[2];
				}

				OPENCL_V(clfftBakePlan(fftPlan->planY, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan for planY failed" ) );

				// create row plan
				// hermitian to real

				//create row plan
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planX, fftPlan->context, CLFFT_1D, &fftPlan->length[ DimX ] ),
					_T( "CreateDefaultPlan for planX failed" ) );

				FFTPlan* rowPlan	= NULL;
				lockRAII* rowLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock ), _T( "fftRepo.getPlan failed" ) );


				rowPlan->outputLayout  = fftPlan->outputLayout;
				rowPlan->inputLayout   = CLFFT_HERMITIAN_INTERLEAVED;
				rowPlan->placeness     = CLFFT_OUTOFPLACE;
				rowPlan->length.push_back(length1);

				rowPlan->inStride[0]   = 1;
				rowPlan->inStride.push_back(Nt);
				rowPlan->iDist         = colPlan->oDist;

				rowPlan->precision     = fftPlan->precision;
				rowPlan->forwardScale  = fftPlan->forwardScale;
				rowPlan->backwardScale = fftPlan->backwardScale;
				rowPlan->tmpBufSize    = fftPlan->tmpBufSize;
				rowPlan->bLdsComplex   = fftPlan->bLdsComplex;
				rowPlan->uLdsFraction  = fftPlan->uLdsFraction;
				rowPlan->ldsPadding    = fftPlan->ldsPadding;
				rowPlan->gen			= fftPlan->gen;
				rowPlan->envelope			= fftPlan->envelope;

				rowPlan->batchsize    = fftPlan->batchsize;

				rowPlan->outStride[0]  = fftPlan->outStride[0];
				rowPlan->outStride.push_back(fftPlan->outStride[1]);
				rowPlan->oDist         = fftPlan->oDist;

				//this 2d is decomposed from 3d
				if (fftPlan->length.size()>2)
				{
					rowPlan->length.push_back(fftPlan->length[2]);
					rowPlan->inStride.push_back(Nt*length1);
					rowPlan->outStride.push_back(fftPlan->outStride[2]);
				}


				OPENCL_V(clfftBakePlan(fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan for planX failed" ) );
			}
			else
			{
				if (fftPlan->tmpBufSize==0 && fftPlan->length.size()<=2)
				{
					fftPlan->tmpBufSize = length0 * length1 *
						fftPlan->batchsize * fftPlan->ElementSize();
				}

				//create row plan
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planX, fftPlan->context, CLFFT_1D, &fftPlan->length[ DimX ] ),
					_T( "CreateDefaultPlan for planX failed" ) );

				FFTPlan* rowPlan	= NULL;
				lockRAII* rowLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planX, rowPlan, rowLock ), _T( "fftRepo.getPlan failed" ) );

				rowPlan->inputLayout   = fftPlan->inputLayout;
				if (fftPlan->large2D || fftPlan->length.size()>2)
				{
					rowPlan->outputLayout  = fftPlan->outputLayout;
					rowPlan->placeness     = fftPlan->placeness;
					rowPlan->outStride[0]  = fftPlan->outStride[0];
					rowPlan->outStride.push_back(fftPlan->outStride[1]);
					rowPlan->oDist         = fftPlan->oDist;
				}
				else
				{
					rowPlan->outputLayout  = CLFFT_COMPLEX_INTERLEAVED;
					rowPlan->placeness     = CLFFT_OUTOFPLACE;
					rowPlan->outStride[0]  = length1;//1;
					rowPlan->outStride.push_back(1);//length0);
					rowPlan->oDist         = length0 * length1;
				}
				rowPlan->precision     = fftPlan->precision;
				rowPlan->forwardScale  = 1.0f;
				rowPlan->backwardScale = 1.0f;
				rowPlan->tmpBufSize    = fftPlan->tmpBufSize;
				rowPlan->bLdsComplex   = fftPlan->bLdsComplex;
				rowPlan->uLdsFraction  = fftPlan->uLdsFraction;
				rowPlan->ldsPadding    = fftPlan->ldsPadding;
				rowPlan->gen			= fftPlan->gen;
				rowPlan->envelope			= fftPlan->envelope;

				// This is the row fft, the first elements distance between the first two FFTs is the distance of the first elements
				// of the first two rows in the original buffer.
				rowPlan->batchsize    = fftPlan->batchsize;
				rowPlan->inStride[0]  = fftPlan->inStride[0];

				//pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
				rowPlan->length.push_back(fftPlan->length[1]);
				rowPlan->inStride.push_back(fftPlan->inStride[1]);

				//this 2d is decomposed from 3d
				if (fftPlan->length.size()>2)
				{
					rowPlan->length.push_back(fftPlan->length[2]);
					rowPlan->inStride.push_back(fftPlan->inStride[2]);
					rowPlan->outStride.push_back(fftPlan->outStride[2]);
				}

				rowPlan->iDist    = fftPlan->iDist;


				OPENCL_V(clfftBakePlan(fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan for planX failed" ) );

				//create col plan
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planY, fftPlan->context, CLFFT_1D, &fftPlan->length[ DimY ] ),
					_T( "CreateDefaultPlan for planY failed" ) );

				FFTPlan* colPlan	= NULL;
				lockRAII* colLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planY, colPlan, colLock ), _T( "fftRepo.getPlan failed" ) );

				if (fftPlan->large2D || fftPlan->length.size()>2)
				{
					colPlan->inputLayout   = fftPlan->outputLayout;
					colPlan->placeness     = CLFFT_INPLACE;
					colPlan->inStride[0]   = fftPlan->outStride[1];
					colPlan->inStride.push_back(fftPlan->outStride[0]);
					colPlan->iDist         = fftPlan->oDist;
				}
				else
				{
					colPlan->inputLayout   = CLFFT_COMPLEX_INTERLEAVED;
					colPlan->placeness     = CLFFT_OUTOFPLACE;
					colPlan->inStride[0]   = 1;//length0;
					colPlan->inStride.push_back(length1);//1);
					colPlan->iDist         = length0 * length1;
				}

				colPlan->outputLayout  = fftPlan->outputLayout;
				colPlan->precision     = fftPlan->precision;
				colPlan->forwardScale  = fftPlan->forwardScale;
				colPlan->backwardScale = fftPlan->backwardScale;
				colPlan->tmpBufSize    = fftPlan->tmpBufSize;
				colPlan->bLdsComplex   = fftPlan->bLdsComplex;
				colPlan->uLdsFraction  = fftPlan->uLdsFraction;
				colPlan->ldsPadding    = fftPlan->ldsPadding;
				colPlan->gen			= fftPlan->gen;
				colPlan->envelope			= fftPlan->envelope;

				// This is a column FFT, the first elements distance between each FFT is the distance of the first two
				// elements in the original buffer. Like a transpose of the matrix
				colPlan->batchsize = fftPlan->batchsize;
				colPlan->outStride[0] = fftPlan->outStride[1];

				//pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
				colPlan->length.push_back(fftPlan->length[0]);
				colPlan->outStride.push_back(fftPlan->outStride[0]);
				colPlan->oDist    = fftPlan->oDist;

				//this 2d is decomposed from 3d
				if (fftPlan->length.size()>2)
				{
					//assert(fftPlan->large2D);
					colPlan->length.push_back(fftPlan->length[2]);
					colPlan->inStride.push_back(fftPlan->outStride[2]);
					colPlan->outStride.push_back(fftPlan->outStride[2]);
				}

				OPENCL_V(clfftBakePlan(fftPlan->planY, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan for planY failed" ) );
			}

			fftPlan->baked = true;
			return	CLFFT_SUCCESS;
		}
	case CLFFT_3D:
		{
			if(fftPlan->inputLayout == CLFFT_REAL)
			{
				size_t clLengths[] = { 1, 1, 0 };
				clLengths[0] = fftPlan->length[ DimX ];
				clLengths[1] = fftPlan->length[ DimY ];

				//create 2D xy plan
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planX, fftPlan->context, CLFFT_2D, clLengths ),
					_T( "CreateDefaultPlan 2D planX failed" ) );

				FFTPlan* xyPlan	= NULL;
				lockRAII* rowLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planX, xyPlan, rowLock ), _T( "fftRepo.getPlan failed" ) );

				xyPlan->inputLayout   = fftPlan->inputLayout;
				xyPlan->outputLayout  = fftPlan->outputLayout;
				xyPlan->placeness     = fftPlan->placeness;
				xyPlan->precision     = fftPlan->precision;
				xyPlan->forwardScale  = 1.0f;
				xyPlan->backwardScale = 1.0f;
				xyPlan->tmpBufSize    = fftPlan->tmpBufSize;
				xyPlan->bLdsComplex   = fftPlan->bLdsComplex;
				xyPlan->uLdsFraction  = fftPlan->uLdsFraction;
				xyPlan->ldsPadding    = fftPlan->ldsPadding;
				xyPlan->gen			 = fftPlan->gen;
				xyPlan->envelope			 = fftPlan->envelope;

				// This is the xy fft, the first elements distance between the first two FFTs is the distance of the first elements
				// of the first two rows in the original buffer.
				xyPlan->batchsize    = fftPlan->batchsize;
				xyPlan->inStride[0]  = fftPlan->inStride[0];
				xyPlan->inStride[1]  = fftPlan->inStride[1];
				xyPlan->outStride[0] = fftPlan->outStride[0];
				xyPlan->outStride[1] = fftPlan->outStride[1];

				//pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
				xyPlan->length.push_back(fftPlan->length[2]);
				xyPlan->inStride.push_back(fftPlan->inStride[2]);
				xyPlan->outStride.push_back(fftPlan->outStride[2]);
				xyPlan->iDist    = fftPlan->iDist;
				xyPlan->oDist    = fftPlan->oDist;

				OPENCL_V(clfftBakePlan(fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan 3D->2D planX failed" ) );

				clLengths[0] = fftPlan->length[ DimZ ];
				clLengths[1] = clLengths[2] = 0;
				//create 1D col plan
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planZ, fftPlan->context, CLFFT_1D, clLengths ),
					_T( "CreateDefaultPlan for planZ failed" ) );

				FFTPlan* colPlan	= NULL;
				lockRAII* colLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planZ, colPlan, colLock ), _T( "fftRepo.getPlan failed" ) );

				switch(fftPlan->outputLayout)
				{
				case CLFFT_HERMITIAN_INTERLEAVED:
					{
						colPlan->outputLayout = CLFFT_COMPLEX_INTERLEAVED;
						colPlan->inputLayout  = CLFFT_COMPLEX_INTERLEAVED;
					}
					break;
				case CLFFT_HERMITIAN_PLANAR:
					{
						colPlan->outputLayout = CLFFT_COMPLEX_PLANAR;
						colPlan->inputLayout  = CLFFT_COMPLEX_PLANAR;
					}
					break;
				default: assert(false);
				}

				colPlan->placeness     = CLFFT_INPLACE;
				colPlan->precision     = fftPlan->precision;
				colPlan->forwardScale  = fftPlan->forwardScale;
				colPlan->backwardScale = fftPlan->backwardScale;
				colPlan->tmpBufSize    = fftPlan->tmpBufSize;
				colPlan->bLdsComplex   = fftPlan->bLdsComplex;
				colPlan->uLdsFraction  = fftPlan->uLdsFraction;
				colPlan->ldsPadding    = fftPlan->ldsPadding;
				colPlan->gen			 = fftPlan->gen;
				colPlan->envelope			 = fftPlan->envelope;

				// This is a column FFT, the first elements distance between each FFT is the distance of the first two
				// elements in the original buffer. Like a transpose of the matrix
				colPlan->batchsize = fftPlan->batchsize;
				colPlan->inStride[0] = fftPlan->outStride[2];
				colPlan->outStride[0] = fftPlan->outStride[2];

				//pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
				colPlan->length.push_back(1 + fftPlan->length[0]/2);
				colPlan->length.push_back(fftPlan->length[1]);
				colPlan->inStride.push_back(fftPlan->outStride[0]);
				colPlan->inStride.push_back(fftPlan->outStride[1]);
				colPlan->outStride.push_back(fftPlan->outStride[0]);
				colPlan->outStride.push_back(fftPlan->outStride[1]);
				colPlan->iDist    = fftPlan->oDist;
				colPlan->oDist    = fftPlan->oDist;

				OPENCL_V(clfftBakePlan(fftPlan->planZ, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan 3D->1D planZ failed" ) );
			}
			else if(fftPlan->outputLayout == CLFFT_REAL)
			{
				if (fftPlan->tmpBufSize == 0)
				{
					fftPlan->tmpBufSize = fftPlan->length[2] * fftPlan->length[1] * (1 + fftPlan->length[0]/2);
					fftPlan->tmpBufSize *= fftPlan->batchsize * fftPlan->ElementSize();
				}

				size_t clLengths[] = { 1, 1, 0 };

				clLengths[0] = fftPlan->length[ DimZ ];
				clLengths[1] = clLengths[2] = 0;

				//create 1D col plan
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planZ, fftPlan->context, CLFFT_1D, clLengths ),
					_T( "CreateDefaultPlan for planZ failed" ) );

				FFTPlan* colPlan	= NULL;
				lockRAII* colLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planZ, colPlan, colLock ), _T( "fftRepo.getPlan failed" ) );

				switch(fftPlan->inputLayout)
				{
				case CLFFT_HERMITIAN_INTERLEAVED:
					{
						colPlan->outputLayout = CLFFT_COMPLEX_INTERLEAVED;
						colPlan->inputLayout  = CLFFT_COMPLEX_INTERLEAVED;
					}
					break;
				case CLFFT_HERMITIAN_PLANAR:
					{
						colPlan->outputLayout = CLFFT_COMPLEX_INTERLEAVED;
						colPlan->inputLayout  = CLFFT_COMPLEX_PLANAR;
					}
					break;
				default: assert(false);
				}

				colPlan->placeness     = CLFFT_OUTOFPLACE;

				colPlan->precision     = fftPlan->precision;
				colPlan->forwardScale  = 1.0f;
				colPlan->backwardScale = 1.0f;
				colPlan->tmpBufSize    = fftPlan->tmpBufSize;
				colPlan->bLdsComplex   = fftPlan->bLdsComplex;
				colPlan->uLdsFraction  = fftPlan->uLdsFraction;
				colPlan->ldsPadding    = fftPlan->ldsPadding;
				colPlan->gen			 = fftPlan->gen;
				colPlan->envelope			 = fftPlan->envelope;

				// This is a column FFT, the first elements distance between each FFT is the distance of the first two
				// elements in the original buffer. Like a transpose of the matrix
				colPlan->batchsize = fftPlan->batchsize;
				colPlan->inStride[0] = fftPlan->inStride[2];
				colPlan->outStride[0] = fftPlan->length[1] * (1 + fftPlan->length[0]/2);

				//pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
				colPlan->length.push_back(1 + fftPlan->length[0]/2);
				colPlan->length.push_back(fftPlan->length[1]);
				colPlan->inStride.push_back(fftPlan->inStride[0]);
				colPlan->inStride.push_back(fftPlan->inStride[1]);
				colPlan->outStride.push_back(1);
				colPlan->outStride.push_back(1 + fftPlan->length[0]/2);
				colPlan->iDist    = fftPlan->iDist;
				colPlan->oDist    = fftPlan->length[2] * fftPlan->length[1] * (1 + fftPlan->length[0]/2);

				if ((fftPlan->tmpBufSizeC2R==0) && ((fftPlan->length[2] > Large1DThreshold) || (fftPlan->length[1] > Large1DThreshold)))
				{
					fftPlan->tmpBufSizeC2R = (1 + fftPlan->length[0]/2) * (fftPlan->length[1]) * (fftPlan->length[2]) *
						fftPlan->batchsize * fftPlan->ElementSize();
				}

				OPENCL_V(clfftBakePlan(fftPlan->planZ, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan 3D->1D planZ failed" ) );

				clLengths[0] = fftPlan->length[ DimX ];
				clLengths[1] = fftPlan->length[ DimY ];

				//create 2D xy plan
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planX, fftPlan->context, CLFFT_2D, clLengths ),
					_T( "CreateDefaultPlan 2D planX failed" ) );

				FFTPlan* xyPlan	= NULL;
				lockRAII* rowLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planX, xyPlan, rowLock ), _T( "fftRepo.getPlan failed" ) );

				xyPlan->inputLayout   = CLFFT_HERMITIAN_INTERLEAVED;
				xyPlan->outputLayout  = fftPlan->outputLayout;

				xyPlan->placeness     = CLFFT_OUTOFPLACE;

				xyPlan->precision     = fftPlan->precision;
				xyPlan->forwardScale  = fftPlan->forwardScale;
				xyPlan->backwardScale = fftPlan->backwardScale;
				xyPlan->tmpBufSize    = fftPlan->tmpBufSize;
				xyPlan->bLdsComplex   = fftPlan->bLdsComplex;
				xyPlan->uLdsFraction  = fftPlan->uLdsFraction;
				xyPlan->ldsPadding    = fftPlan->ldsPadding;
				xyPlan->gen			 = fftPlan->gen;
				xyPlan->envelope			 = fftPlan->envelope;

				// This is the xy fft, the first elements distance between the first two FFTs is the distance of the first elements
				// of the first two rows in the original buffer.
				xyPlan->batchsize    = fftPlan->batchsize;
				xyPlan->inStride[0]  = 1;
				xyPlan->inStride[1]  = (1 + fftPlan->length[0]/2);
				xyPlan->outStride[0] = fftPlan->outStride[0];
				xyPlan->outStride[1] = fftPlan->outStride[1];

				//pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
				xyPlan->length.push_back(fftPlan->length[2]);
				xyPlan->inStride.push_back(fftPlan->length[1] * (1 + fftPlan->length[0]/2));
				xyPlan->outStride.push_back(fftPlan->outStride[2]);
				xyPlan->iDist    = colPlan->oDist;
				xyPlan->oDist    = fftPlan->oDist;

				OPENCL_V(clfftBakePlan(fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan 3D->2D planX failed" ) );
			}
			else
			{
				if (fftPlan->tmpBufSize==0 && (
					fftPlan->length[0] > Large1DThreshold ||
					fftPlan->length[1] > Large1DThreshold ||
					fftPlan->length[2] > Large1DThreshold
					))
				{
					fftPlan->tmpBufSize = fftPlan->length[0] * fftPlan->length[1] * fftPlan->length[2] *
						fftPlan->batchsize * fftPlan->ElementSize();
				}

				size_t clLengths[] = { 1, 1, 0 };
				clLengths[0] = fftPlan->length[ DimX ];
				clLengths[1] = fftPlan->length[ DimY ];

				//create 2D xy plan
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planX, fftPlan->context, CLFFT_2D, clLengths ),
					_T( "CreateDefaultPlan 2D planX failed" ) );

				FFTPlan* xyPlan	= NULL;
				lockRAII* rowLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planX, xyPlan, rowLock ), _T( "fftRepo.getPlan failed" ) );

				xyPlan->inputLayout   = fftPlan->inputLayout;
				xyPlan->outputLayout  = fftPlan->outputLayout;
				xyPlan->placeness     = fftPlan->placeness;
				xyPlan->precision     = fftPlan->precision;
				xyPlan->forwardScale  = 1.0f;
				xyPlan->backwardScale = 1.0f;
				xyPlan->tmpBufSize    = fftPlan->tmpBufSize;
				xyPlan->bLdsComplex   = fftPlan->bLdsComplex;
				xyPlan->uLdsFraction  = fftPlan->uLdsFraction;
				xyPlan->ldsPadding    = fftPlan->ldsPadding;
				xyPlan->gen			 = fftPlan->gen;
				xyPlan->envelope			 = fftPlan->envelope;

				// This is the xy fft, the first elements distance between the first two FFTs is the distance of the first elements
				// of the first two rows in the original buffer.
				xyPlan->batchsize    = fftPlan->batchsize;
				xyPlan->inStride[0]  = fftPlan->inStride[0];
				xyPlan->inStride[1]  = fftPlan->inStride[1];
				xyPlan->outStride[0] = fftPlan->outStride[0];
				xyPlan->outStride[1] = fftPlan->outStride[1];

				//pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
				xyPlan->length.push_back(fftPlan->length[2]);
				xyPlan->inStride.push_back(fftPlan->inStride[2]);
				xyPlan->outStride.push_back(fftPlan->outStride[2]);
				xyPlan->iDist    = fftPlan->iDist;
				xyPlan->oDist    = fftPlan->oDist;

				OPENCL_V(clfftBakePlan(fftPlan->planX, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan 3D->2D planX failed" ) );

				clLengths[0] = fftPlan->length[ DimZ ];
				clLengths[1] = clLengths[2] = 0;
				//create 1D col plan
				OPENCL_V(clfftCreateDefaultPlan( &fftPlan->planZ, fftPlan->context, CLFFT_1D, clLengths ),
					_T( "CreateDefaultPlan for planZ failed" ) );

				FFTPlan* colPlan	= NULL;
				lockRAII* colLock	= NULL;
				OPENCL_V( fftRepo.getPlan( fftPlan->planZ, colPlan, colLock ), _T( "fftRepo.getPlan failed" ) );

				colPlan->inputLayout   = fftPlan->outputLayout;
				colPlan->outputLayout  = fftPlan->outputLayout;
				colPlan->placeness     = CLFFT_INPLACE;
				colPlan->precision     = fftPlan->precision;
				colPlan->forwardScale  = fftPlan->forwardScale;
				colPlan->backwardScale = fftPlan->backwardScale;
				colPlan->tmpBufSize    = fftPlan->tmpBufSize;
				colPlan->bLdsComplex   = fftPlan->bLdsComplex;
				colPlan->uLdsFraction  = fftPlan->uLdsFraction;
				colPlan->ldsPadding    = fftPlan->ldsPadding;
				colPlan->gen			 = fftPlan->gen;
				colPlan->envelope			 = fftPlan->envelope;

				// This is a column FFT, the first elements distance between each FFT is the distance of the first two
				// elements in the original buffer. Like a transpose of the matrix
				colPlan->batchsize = fftPlan->batchsize;
				colPlan->inStride[0] = fftPlan->outStride[2];
				colPlan->outStride[0] = fftPlan->outStride[2];

				//pass length and other info to kernel, so the kernel knows this is decomposed from higher dimension
				colPlan->length.push_back(fftPlan->length[0]);
				colPlan->length.push_back(fftPlan->length[1]);
				colPlan->inStride.push_back(fftPlan->outStride[0]);
				colPlan->inStride.push_back(fftPlan->outStride[1]);
				colPlan->outStride.push_back(fftPlan->outStride[0]);
				colPlan->outStride.push_back(fftPlan->outStride[1]);
				colPlan->iDist    = fftPlan->oDist;
				colPlan->oDist    = fftPlan->oDist;

				OPENCL_V(clfftBakePlan(fftPlan->planZ, numQueues, commQueueFFT, NULL, NULL ), _T( "BakePlan 3D->1D planZ failed" ) );
			}

			fftPlan->baked = true;
			return	CLFFT_SUCCESS;
		}
	}

	//	For the radices that we have factored, we need to load/compile and build the appropriate OpenCL kernels
	OPENCL_V( fftPlan->GenerateKernel( fftRepo, *commQueueFFT ), _T( "GenerateKernel() failed" ) );

	//	For the radices that we have factored, we need to load/compile and build the appropriate OpenCL kernels
	OPENCL_V( CompileKernels( *commQueueFFT, plHandle, fftPlan->gen, fftPlan ), _T( "CompileKernels() failed" ) );

	//	Allocate resources
	OPENCL_V( fftPlan->AllocateBuffers (), _T("AllocateBuffers() failed"));

	fftPlan->ConstructAndEnqueueConstantBuffers( commQueueFFT );

	//	Record that we baked the plan
	fftPlan->baked		= true;

	return	CLFFT_SUCCESS;
}

clfftStatus clfftCopyPlan( clfftPlanHandle* out_plHandle, cl_context new_context, clfftPlanHandle in_plHandle )
{
	FFTRepo& fftRepo	= FFTRepo::getInstance( );
	FFTPlan* in_fftPlan	= NULL, *out_fftPlan = NULL;
	lockRAII* in_planLock = NULL, *out_planLock = NULL;

	OPENCL_V( fftRepo.getPlan( in_plHandle, in_fftPlan, in_planLock ), _T( "fftRepo.getPlan failed" ) );

	OPENCL_V( clfftCreateDefaultPlan( out_plHandle, new_context, in_fftPlan->dim, &in_fftPlan->length[ 0 ] ),
		_T( "clfftCreateDefaultPlan failed" ) );

	OPENCL_V( fftRepo.getPlan( *out_plHandle, out_fftPlan, out_planLock ), _T( "fftRepo.getPlan failed" ) );

	//	Let other operations complete before attempting to copy the plan
	scopedLock sLock( *in_planLock, _T( "clfftCopyPlan" ) );

	out_fftPlan->baked = false;
	out_fftPlan->gen = in_fftPlan->gen;
	out_fftPlan->envelope = in_fftPlan->envelope;
	out_fftPlan->dim = in_fftPlan->dim;
	out_fftPlan->inputLayout = in_fftPlan->inputLayout;
	out_fftPlan->outputLayout = in_fftPlan->outputLayout;
	out_fftPlan->placeness = in_fftPlan->placeness;
	out_fftPlan->precision = in_fftPlan->precision;
	out_fftPlan->forwardScale = in_fftPlan->forwardScale;
	out_fftPlan->backwardScale = in_fftPlan->backwardScale;
	out_fftPlan->iDist = in_fftPlan->iDist;
	out_fftPlan->oDist = in_fftPlan->oDist;
	out_fftPlan->length = in_fftPlan->length;
	out_fftPlan->inStride = in_fftPlan->inStride;
	out_fftPlan->outStride = in_fftPlan->outStride;
	out_fftPlan->batchsize = in_fftPlan->batchsize;

	return	CLFFT_SUCCESS;
}

clfftStatus FFTPlan::ConstructAndEnqueueConstantBuffers( cl_command_queue* commQueueFFT )
{
	//	Construct the constant buffer and call clEnqueueWriteBuffer
	//
	cb_t ConstantBufferParams [CLFFT_CB_SIZE];
	memset (& ConstantBufferParams, 0, sizeof (ConstantBufferParams));

	cl_uint nY = 1;
	cl_uint nZ = 0;
	cl_uint nW = 0;
	cl_uint n5 = 0;

	switch( /*fftPlan->*/length.size() )
	{
	case 1:
		nY = std::max<cl_uint> (1, cl_uint (/*fftPlan->*/batchsize));
		break;

	case 2:
		nY = (cl_uint)/*fftPlan->*/length[DimY];
		nZ = std::max<cl_uint> (1, cl_uint (/*fftPlan->*/batchsize));
		break;

	case 3:
		nY = (cl_uint)/*fftPlan->*/length[DimY];
		nZ = (cl_uint)/*fftPlan->*/length[DimZ];
		nW = std::max<cl_uint> (1, cl_uint (/*fftPlan->*/batchsize));
		break;

	case 4:
		nY = (cl_uint)/*fftPlan->*/length[DimY];
		nZ = (cl_uint)/*fftPlan->*/length[DimZ];
		nW = (cl_uint)/*fftPlan->*/length[DimW];
		n5 = std::max<cl_uint> (1, cl_uint (/*fftPlan->*/batchsize));
		break;
	}
	ConstantBufferParams[CLFFT_CB_NY ].u = nY;
	ConstantBufferParams[CLFFT_CB_NZ ].u = nZ;
	ConstantBufferParams[CLFFT_CB_NW ].u = nW;
	ConstantBufferParams[CLFFT_CB_N5 ].u = n5;

	assert (/*fftPlan->*/inStride.size() == /*fftPlan->*/outStride.size());

	switch (/*fftPlan->*/inStride.size()) {
	case 1:
		ConstantBufferParams[CLFFT_CB_ISX].u = cl_uint (/*fftPlan->*/inStride[0]);
		ConstantBufferParams[CLFFT_CB_ISY].u = cl_uint (/*fftPlan->*/iDist);
		break;

	case 2:
		ConstantBufferParams[CLFFT_CB_ISX].u = cl_uint (/*fftPlan->*/inStride[0]);
		ConstantBufferParams[CLFFT_CB_ISY].u = cl_uint (/*fftPlan->*/inStride[1]);
		ConstantBufferParams[CLFFT_CB_ISZ].u = cl_uint (/*fftPlan->*/iDist);
		break;

	case 3:
		ConstantBufferParams[CLFFT_CB_ISX].u = cl_uint (/*fftPlan->*/inStride[0]);
		ConstantBufferParams[CLFFT_CB_ISY].u = cl_uint (/*fftPlan->*/inStride[1]);
		ConstantBufferParams[CLFFT_CB_ISZ].u = cl_uint (/*fftPlan->*/inStride[2]);
		ConstantBufferParams[CLFFT_CB_ISW].u = cl_uint (/*fftPlan->*/iDist);
		break;

	case 4:
		ConstantBufferParams[CLFFT_CB_ISX].u = cl_uint (/*fftPlan->*/inStride[0]);
		ConstantBufferParams[CLFFT_CB_ISY].u = cl_uint (/*fftPlan->*/inStride[1]);
		ConstantBufferParams[CLFFT_CB_ISZ].u = cl_uint (/*fftPlan->*/inStride[2]);
		ConstantBufferParams[CLFFT_CB_ISW].u = cl_uint (/*fftPlan->*/inStride[3]);
		ConstantBufferParams[CLFFT_CB_IS5].u = cl_uint (/*fftPlan->*/iDist);
		break;
	}

	switch (/*fftPlan->*/outStride.size()) {
	case 1:
		ConstantBufferParams[CLFFT_CB_OSX].u = cl_uint (/*fftPlan->*/outStride[0]);
		ConstantBufferParams[CLFFT_CB_OSY].u = cl_uint (/*fftPlan->*/oDist);
		break;

	case 2:
		ConstantBufferParams[CLFFT_CB_OSX].u = cl_uint (/*fftPlan->*/outStride[0]);
		ConstantBufferParams[CLFFT_CB_OSY].u = cl_uint (/*fftPlan->*/outStride[1]);
		ConstantBufferParams[CLFFT_CB_OSZ].u = cl_uint (/*fftPlan->*/oDist);
		break;

	case 3:
		ConstantBufferParams[CLFFT_CB_OSX].u = cl_uint (/*fftPlan->*/outStride[0]);
		ConstantBufferParams[CLFFT_CB_OSY].u = cl_uint (/*fftPlan->*/outStride[1]);
		ConstantBufferParams[CLFFT_CB_OSZ].u = cl_uint (/*fftPlan->*/outStride[2]);
		ConstantBufferParams[CLFFT_CB_OSW].u = cl_uint (/*fftPlan->*/oDist);
		break;

	case 4:
		ConstantBufferParams[CLFFT_CB_OSX].u = cl_uint (/*fftPlan->*/outStride[0]);
		ConstantBufferParams[CLFFT_CB_OSY].u = cl_uint (/*fftPlan->*/outStride[1]);
		ConstantBufferParams[CLFFT_CB_OSZ].u = cl_uint (/*fftPlan->*/outStride[2]);
		ConstantBufferParams[CLFFT_CB_OSW].u = cl_uint (/*fftPlan->*/outStride[3]);
		ConstantBufferParams[CLFFT_CB_OS5].u = cl_uint (/*fftPlan->*/oDist);
		break;
	}

	OPENCL_V(clEnqueueWriteBuffer( *commQueueFFT,
		/*fftPlan->*/const_buffer,
		1,		// TODO? non-blocking write?
		0,
		sizeof(ConstantBufferParams),
		&ConstantBufferParams,
		0,
		NULL,
		NULL), _T("clEnqueueWriteBuffer failed") );

	return CLFFT_SUCCESS;
}

//TODO caching kernel binaries for later reload
#if 0
typedef std::pair<std::string, clfftPlanHandle> plan_tree_node_t;
typedef std::vector< std::pair<std::string, clfftPlanHandle> > plan_tree_t;

void make_plan_tree( plan_tree_t & tree, std::string name, clfftPlanHandle handle )
{
	tree.push_back( plan_tree_node_t(name, handle) );

	FFTPlan* plan = NULL;
	FFTRepo& repo = FFTRepo::getInstance();
	lockRAII* lock = NULL;
	clfftStatus status = repo.getPlan( handle, plan, lock );
	if( status != CLFFT_SUCCESS )
	{
		throw( "make_plan_tree failure: repo.getPlan" );
	}

	if( plan->planX )
	{
		std::string subplan(name);
		subplan += "X";
		make_plan_tree(tree, subplan, plan->planX );
	}

	if( plan->planY )
	{
		std::string subplan(name);
		subplan += "Y";
		make_plan_tree(tree, subplan, plan->planY );
	}

	if( plan->planZ )
	{
		std::string subplan(name);
		subplan += "Z";
		make_plan_tree(tree, subplan, plan->planZ );
	}
}

clfftStatus clfftWritePlanToDisk( clfftPlanHandle plan_handle, const char* filename )
{
	plan_tree_t plan_tree;
	make_plan_tree( plan_tree, "plan", plan_handle );

	std::ofstream planfile;
	planfile.open(filename, std::ios::binary);

	while( !plan_tree.empty() )
	{
		plan_tree_node_t node( *plan_tree.begin() );
		plan_tree.erase( plan_tree.begin() );

		FFTPlan* plan = NULL;
		FFTRepo& repo = FFTRepo::getInstance();
		lockRAII* lock = NULL;
		OPENCL_V(repo.getPlan( node.second, plan, lock ), _T("getPlan failure"));

		// print the name of the node (plan, planX, planXX, planY, plan XY, etc)
		planfile << node.first << " ";

		planfile << "dimensions " << plan->dim << " " << plan->length.size();
		// dimensions must be listed first because clfftReadPlanFromDisk
		// will need to use dimensions for reading in strides and such

		for( int i = 0; i < plan->length.size(); ++i )
		{
			planfile << " " << plan->length[i];
		}

		planfile << " batch " << plan->batchsize;

		planfile << " instride " << plan->inStride.size();
		for( int i = 0; i < plan->inStride.size(); ++i )
		{
			planfile << " " << plan->inStride[i];
		}

		planfile << " outstride " << plan->outStride.size();
		for( int i = 0; i < plan->outStride.size(); ++i )
		{
			planfile << " " << plan->outStride[i];
		}

		planfile << " in-out-distances " << plan->iDist << " " << plan->oDist;
		planfile << " in-out-layouts " << plan->inputLayout << " " << plan->outputLayout;
		planfile << " resultlocation " << plan->placeness;
		planfile << " precision " << plan->precision;
		planfile << " forwardscale " << float_as_hex<double>(plan->forwardScale);
		planfile << " backwardscale " << float_as_hex<double>(plan->backwardScale);
		// we need to stash scales as hex so that we don't have any roundoff error
		// clfftReadPlanFromDisk will read the hex back in as float

		planfile << " gen " << plan->gen;
		planfile << " bLdsComplex " << plan->bLdsComplex;
		planfile << " ldsPadding " << plan->ldsPadding;
		planfile << " uLdsFraction " << plan->uLdsFraction;
		planfile << " large1D_Xfactor " << plan->large1D_Xfactor;
		planfile << " cacheSize " << plan->cacheSize;
		planfile << " tmpBufSize " << plan->tmpBufSize;
		planfile << " large1D " << plan->large1D;
		planfile << " large2D " << plan->large2D;

		if( plan->baked == true )
		{
			planfile << " number-of-devices " << plan->number_of_devices;

			if( plan->number_of_devices > 0 )
			{
				planfile << " binary-sizes";
				for( int i = 0; i < plan->number_of_devices; i++ )
				{
					planfile << " " << *(plan->binary_sizes.get() + i);
				}

				planfile << " binaries ";
				for( int i = 0; i < plan->number_of_devices; i++ )
				{
					planfile << beginning_of_binary;
					planfile.write( plan->binaries[i].get(), plan->binary_sizes[i] );
					planfile << end_of_binary;
				}
			}
			else
			{
				planfile << " ";
			}
		}
	}

	planfile << " " << end_of_file;
	planfile.close();
	return CLFFT_SUCCESS;
}

void FFTPlan::ResetBinarySizes()
{
	binary_sizes.reset(new size_t[number_of_devices]);
}

void FFTPlan::ResetBinaries()
{
	binaries.clear();
	for( int i = 0; i < number_of_devices; i++ )
	{
		binaries.push_back( std::unique_ptr<char[]>(new char[binary_sizes[i]] ) );
	}
}

std::string pop_next_word( std::string & str )
{
	size_t next_space = str.find_first_of(' ');

	std::string next_word( str.substr( 0, next_space ) );
	str.erase( 0, next_space+1 ); // we need the extra +1 to munch off the space

	return next_word;
}

int my_string_to_int( std::string str )
{
	int i;
	std::stringstream string_to_int( str );
	string_to_int >> i;
	return i;
}

bool start_of_a_plan( std::string word )
{
	if( word.substr(0,4) == "plan" )
		return true;
	else
		return false;
}

clfftStatus clfftReadPlanFromDisk( clfftPlanHandle plan_handle, const char* filename )
{
	plan_tree_t tree;

	FFTPlan* plan = NULL;
	FFTRepo& repo = FFTRepo::getInstance();
	lockRAII* lock = NULL;
	OPENCL_V(repo.getPlan( plan_handle, plan, lock ), _T("getPlan failure"));

	std::ifstream planfile;
	planfile.open(filename, std::ios::in | std::ios::binary);

	unsigned int dimensions = 0;
	std::string next_word;

	while( planfile >> next_word )
	{
		if( start_of_a_plan( next_word ) )
		{
			if( next_word.length() > 4 )
			// if true, this is not a base plan
			{
				clfftDim temp_dimension = CLFFT_1D;
				size_t temp_lengths[3] = {1,1,1};

				// let's create the plan to represent the child plan
				clfftPlanHandle child_plan;
				OPENCL_V(clfftCreateDefaultPlan( &child_plan, plan->context, temp_dimension, temp_lengths ),
					"clfftReadPlanFromDisk(): error calling clfftCreateDefaultPlan()");

				tree.push_back( plan_tree_node_t( next_word, child_plan ) );

				// we need to update the planX, Y, or Z pointer to point at the child plan
				char child_plan_name = next_word.rbegin()[0]; // this tells us if this is planX, Y, or Z
				next_word.erase( next_word.end()-1 ); // this tells us the parent plan
				std::string parent_plan_name = next_word;

				clfftPlanHandle parent_plan = 0;

				for( int i = 0; i < tree.size(); i++ )
				{
					if( tree[i].first == parent_plan_name )
					{
						parent_plan = tree[i].second;
					}
				}

				plan = NULL;
				OPENCL_V(repo.getPlan( parent_plan, plan, lock ), _T("getPlan failure"));

				if( child_plan_name == 'X' )
					plan->planX = child_plan;
				else if( child_plan_name == 'Y' )
					plan->planY = child_plan;
				else if( child_plan_name == 'Z' )
					plan->planZ = child_plan;
				else
					OPENCL_V(CLFFT_INVALID_PLAN, "clfftReadPlanFromDisk(): could not identify child plan" );

				// our child plan is now the active plan
				plan = NULL;
				OPENCL_V(repo.getPlan( child_plan, plan, lock ), _T("getPlan failure"));
				plan_handle = child_plan;
			}
			else
			// if this is a base plan, we don't need to do anything fancy.
			// just add the node to the tree
			{
				tree.push_back( plan_tree_node_t( next_word, plan_handle ) );
			}

			plan->readFromFile = true;
		}
		else if( next_word == "dimensions" )
		{
			size_t lengths[3];

			// read number of dimensions
			planfile >> dimensions;

			// number of length values that follow (subplans have some really strange things going on,
			// so this might not always match the dimension of the transform)
			size_t number_of_lengths = 0;
			planfile >> number_of_lengths;

			OPENCL_V( clfftSetPlanDim(plan_handle, static_cast<clfftDim>(dimensions)), _T("clfftReadPlanFromDisk: clfftSetPlanDim") );

			for( unsigned int i = 0; i < number_of_lengths; ++i )
			{
				planfile >> lengths[i]; // read one dimension

				// We have to explicitly set the lengths instead of using clfftSetPlanLength here.
				// Because the number of values to add might be greater than the number of dimensions in plan->dimension,
				// we don't want to miss out on any super awesome numbers getting added to plan->length with clfftSetPlanLength
				if( i >= plan->length.size() ) plan->length.push_back(1);
				plan->length[i] = lengths[i];
			}
		}
		else if( next_word == "batch" )
		{
			unsigned int batch;
			planfile >> batch;

			OPENCL_V( clfftSetPlanBatchSize(plan_handle, batch), _T("clfftReadPlanFromDisk: clfftSetPlanBatchSize") );
		}
		else if( next_word == "instride" )
		{
			size_t strides[3];

			// number of stride values that follow (subplans have some really strange things going on,
			// so this might not always match the dimension of the transform)
			size_t number_of_strides = 0;
			planfile >> number_of_strides;

			for( unsigned int i = 0; i < number_of_strides; ++i )
			{
				planfile >> strides[i]; // read one dimension

				// We have to explicitly set inStride instead of using clfftSetPlanInStride here.
				// Because the number of values to add might be greater than the number of dimensions in plan->dimension,
				// we don't want to miss out on any super awesome numbers getting added to plan->inStride with clfftSetPlanInStride
				if( i >= plan->inStride.size() ) plan->inStride.push_back(1);
				plan->inStride[i] = strides[i];
			}
		}
		else if( next_word == "outstride" )
		{
			size_t strides[3];

			// number of stride values that follow (subplans have some really strange things going on,
			// so this might not always match the dimension of the transform)
			size_t number_of_strides = 0;
			planfile >> number_of_strides;

			for( unsigned int i = 0; i < number_of_strides; ++i )
			{
				planfile >> strides[i]; // read one dimension

				// We have to explicitly set outStride instead of using clfftSetPlanOutStride here.
				// Because the number of values to add might be greater than the number of dimensions in plan->dimension,
				// we don't want to miss out on any super awesome numbers getting added to plan->outStride with clfftSetPlanOutStride
				if( i >= plan->outStride.size() ) plan->outStride.push_back(1);
				plan->outStride[i] = strides[i];
			}
		}
		else if( next_word == "in-out-distances" )
		{
			size_t indistance, outdistance;
			planfile >> indistance >> outdistance;

			OPENCL_V( clfftSetPlanDistance( plan_handle, indistance, outdistance ), _T("clfftReadPlanFromDisk: clfftSetPlanDistance" ) );
		}
		else if( next_word == "in-out-layouts" )
		{
			size_t inlayout, outlayout;
			planfile >> inlayout >> outlayout;

			OPENCL_V( clfftSetLayout( plan_handle, static_cast<clfftLayout>(inlayout), static_cast<clfftLayout>(outlayout) ), _T("clfftReadPlanFromDisk: clfftSetLayout") );
		}
		else if( next_word == "resultlocation" )
		{
			size_t location;
			planfile >> location;

			OPENCL_V( clfftSetResultLocation( plan_handle, static_cast<clfftResultLocation>(location) ), _T("clfftReadPlanFromDisk: clfftSetResultLocation") );
		}
		else if( next_word == "precision" )
		{
			size_t precision;
			planfile >> precision;

			OPENCL_V( clfftSetPlanPrecision( plan_handle, static_cast<clfftPrecision>(precision) ), _T("clfftReadPlanFromDisk: clfftSetPlanPrecision") );
		}
		else if( next_word == "forwardscale" || next_word == "backwardscale" )
		{
			size_t scale;
			planfile >> scale;

			if( next_word == "forwardscale" )
			{
				OPENCL_V( clfftSetPlanScale( plan_handle, CLFFT_FORWARD, hex_as_float<float>((unsigned int)scale) ), _T("clfftReadPlanFromDisk: clfftSetPlanScale") );
			}
			else
			{
				OPENCL_V( clfftSetPlanScale( plan_handle, CLFFT_BACKWARD, hex_as_float<float>((unsigned int)scale) ), _T("clfftReadPlanFromDisk: clfftSetPlanScale") );
			}
		}
		else if( next_word == "gen" )
		{
			int gen_read;
			planfile >> gen_read;
			plan->gen = static_cast<clfftGenerators>(gen_read);
		}
		else if( next_word == "bLdsComplex" )
		{
			planfile >> plan->bLdsComplex;
		}
		else if( next_word == "ldsPadding" )
		{
			planfile >> plan->ldsPadding;
		}
		else if( next_word == "uLdsFraction" )
		{
			planfile >> plan->uLdsFraction;
		}
		else if( next_word == "large1D_Xfactor" )
		{
			planfile >> plan->large1D_Xfactor;
		}
		else if( next_word == "cacheSize" )
		{
			planfile >> plan->cacheSize;
		}
		else if( next_word == "tmpBufSize" )
		{
			planfile >> plan->tmpBufSize;
		}
		else if( next_word == "large1D" )
		{
			planfile >> plan->large1D;
		}
		else if( next_word == "large2D" )
		{
			planfile >> plan->large2D;
		}
		else if( next_word == "number-of-devices" )
		{
			planfile >> plan->number_of_devices;
		}
		else if( next_word == "binary-sizes" )
		{
			plan->ResetBinarySizes();
			for( int i = 0; i < plan->number_of_devices; i++ )
			{
				planfile >> plan->binary_sizes[i];
			}
		}
		else if( next_word == "binaries" )
		{
			plan->ResetBinaries();

			size_t number_of_devices = plan->number_of_devices;

			while( static_cast<char>(planfile.peek()) == ' ' )
				planfile.ignore();

			// consume the beginning of binary message. the binary will begin with the character immediately following
			std::unique_ptr<char[]> beginning_message( new char[beginning_of_binary.size()] );
			planfile.read( beginning_message.get(), beginning_of_binary.size() );

			for( int i = 0; i < plan->number_of_devices; i++ )
			{
				planfile.read( plan->binaries[i].get(), plan->binary_sizes[i] );
			}

			std::unique_ptr<char[]> end_message( new char[end_of_binary.size()] );
			planfile.read( end_message.get(), end_of_binary.size() );
		}
		else if( next_word == end_of_file )
		{
			// we're at the end of the file
		}
		else
		{
			std::cout << next_word << std::endl;
			OPENCL_V( CLFFT_INVALID_ARG_VALUE, _T("clfftReadPlanFromDisk: unrecognized parameter") );
		}
	}

	return CLFFT_SUCCESS;
}
#endif

clfftStatus	clfftDestroyPlan( clfftPlanHandle* plHandle )
{
	FFTRepo& fftRepo	= FFTRepo::getInstance( );
	FFTPlan* fftPlan	= NULL;
	lockRAII* planLock	= NULL;

	OPENCL_V( fftRepo.getPlan( *plHandle, fftPlan, planLock ), _T( "fftRepo.getPlan failed" ) );

	//	Recursively destroy subplans, that are used for higher dimensional FFT's
	if( fftPlan->planX )
		clfftDestroyPlan( &fftPlan->planX );
	if( fftPlan->planY )
		clfftDestroyPlan( &fftPlan->planY );
	if( fftPlan->planZ )
		clfftDestroyPlan( &fftPlan->planZ );
	if( fftPlan->planTX )
		clfftDestroyPlan( &fftPlan->planTX );
	if( fftPlan->planTY )
		clfftDestroyPlan( &fftPlan->planTY );
	if( fftPlan->planTZ )
		clfftDestroyPlan( &fftPlan->planTZ );
	if( fftPlan->planRCcopy )
		clfftDestroyPlan( &fftPlan->planRCcopy );

	fftRepo.deletePlan( plHandle );

	return	CLFFT_SUCCESS;
}

//	This routine will query the OpenCL context for it's devices
//	and their hardware limitations, which we synthesize into a
//	hardware "envelope".
//	We only query the devices the first time we're called after
//	the object's context is set.  On 2nd and subsequent calls,
//	we just return the pointer.
//
clfftStatus FFTPlan::SetEnvelope ()
{

	// TODO  The caller has already acquired the lock on *this
	//	However, we shouldn't depend on it.

	if (0 == envelope.limit_LocalMemSize) do {
		//	First time, query OpenCL for the device info
		//
		memset (&envelope, 0, sizeof(envelope));

		//	Get the size needed for the device list
		//
		size_t deviceListSize = 0;
		OPENCL_V( ::clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &deviceListSize ),
			_T("Getting device array size ( ::clGetContextInfo() )" ));
		cl_uint n = cl_uint (deviceListSize / sizeof(cl_device_id));
		if (n == 0) break;

		std::vector< cl_device_id > devices( n+1 );
		//	Get the device list
		//
		OPENCL_V( ::clGetContextInfo( context, CL_CONTEXT_DEVICES, deviceListSize, &devices[ 0 ], NULL ),
			_T("Getting device array ( ::clGetContextInfo() )") );

		//	Get the # of devices
		//
		cl_uint cContextDevices	= 0;

		size_t deviceVersionSize	= 0;
		OPENCL_V( ::clGetDeviceInfo( devices[0], CL_DEVICE_VERSION, 0, NULL, &deviceVersionSize ),
			_T("Getting CL_DEVICE_VERSION Info string size ( ::clGetDeviceInfo() )" ));

		std::vector< char > szDeviceVersion( deviceVersionSize );
		OPENCL_V( ::clGetDeviceInfo( devices[0], CL_DEVICE_VERSION, deviceVersionSize, &szDeviceVersion[ 0 ], NULL ),
			_T("Getting CL_DEVICE_VERSION Platform Info string ( ::clGetDeviceInfo() )" ));

		char openclstr[11]="OpenCL 1.0";

		if (!strncmp((const char*)&szDeviceVersion[ 0 ], openclstr, 10))
		{
			cContextDevices	= 1;
		}
		else
		{
			OPENCL_V( ::clGetContextInfo( context, CL_CONTEXT_NUM_DEVICES, sizeof( cContextDevices ), &cContextDevices, NULL ),
				_T("Getting number of context devices ( ::clGetContextInfo() )" ));
		}

		cContextDevices = std::min<cl_uint> (cContextDevices, n);
		if (0 == cContextDevices)
			break;

		envelope.limit_LocalMemSize  = ~0;
		envelope.limit_WorkGroupSize = ~0;
		envelope.limit_Dimensions    = countOf (envelope.limit_Size);
		for (size_t u = 0; u < countOf (envelope.limit_Size); ++u) {
			envelope.limit_Size[u] = ~0;
		}

		for( cl_uint i = 0; i < cContextDevices; ++i )
		{
			cl_device_id devId = devices[i];

			cl_ulong memsize = 0;
			unsigned int maxdim = 0;
			size_t temp[countOf (envelope.limit_Size)];
			memset (&temp, 0, sizeof(temp));

			OPENCL_V( ::clGetDeviceInfo( devId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof( cl_ulong ), &memsize, NULL ),
				_T("Getting CL_DEVICE_LOCAL_MEM_SIZE device info ( ::clGetDeviceInfo() )") );
			envelope.limit_LocalMemSize = std::min<size_t> (envelope.limit_LocalMemSize, memsize);

			OPENCL_V( ::clGetDeviceInfo( devId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof( unsigned int ), &maxdim, NULL ),
				_T("Getting CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS device info ( ::clGetDeviceInfo() )") );
			BUG_CHECK (countOf (envelope.limit_Size) >= maxdim);
			envelope.limit_Dimensions = std::min<size_t> (envelope.limit_Dimensions, maxdim);

			OPENCL_V( ::clGetDeviceInfo( devId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof( size_t ), &temp[0], NULL ),
				_T("Getting CL_DEVICE_MAX_WORK_GROUP_SIZE device info ( ::clGetDeviceInfo() )") );
			envelope.limit_WorkGroupSize = std::min<size_t> (envelope.limit_WorkGroupSize, temp[0]);

			OPENCL_V( ::clGetDeviceInfo( devId, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof( temp ), &temp[0], NULL ),
				_T("Getting CL_DEVICE_MAX_WORK_ITEM_SIZES device info ( ::clGetDeviceInfo() )") );
			for (size_t u = 0; u < envelope.limit_Dimensions; ++u) {
				BUG_CHECK (temp[u] > 0)
				envelope.limit_Size[u] = std::min<size_t> (envelope.limit_Size[u], temp[u]);
			}
		}

		BUG_CHECK (envelope.limit_LocalMemSize >= 1024)
	} while (0);

	return CLFFT_SUCCESS;
}

clfftStatus FFTPlan::AllocateBuffers ()
{
	cl_int status = CL_SUCCESS;

	assert (NULL == const_buffer);
	ReleaseBuffers ();

	assert(4 == sizeof(int));

	do {
		const_buffer = clCreateBuffer (context,
										CL_MEM_READ_ONLY,
										CLFFT_CB_SIZE * sizeof (int),
										0,
										&status);
		if (CL_SUCCESS != status)
			break;
	} while (0);

	return	(clfftStatus) status;
}

clfftStatus FFTPlan::ReleaseBuffers ()
{
	clfftStatus result = CLFFT_SUCCESS;
	clfftStatus tmp;

	if( NULL != const_buffer )
	{
		tmp = static_cast< clfftStatus >( clReleaseMemObject( const_buffer ) );
		const_buffer = NULL;
		if( CLFFT_SUCCESS == result )
			result = tmp;
	}

	if( NULL != intBuffer )
	{
		tmp = static_cast< clfftStatus >( clReleaseMemObject( intBuffer ) );
		intBuffer = NULL;
		if( CLFFT_SUCCESS == result )
			result = tmp;
	}

	if( NULL != intBufferRC )
	{
		tmp = static_cast< clfftStatus >( clReleaseMemObject( intBufferRC ) );
		intBufferRC = NULL;
		if( CLFFT_SUCCESS == result )
			result = tmp;
	}

	return	result;
}

clfftStatus  FFTPlan::GetWorkSizes (std::vector<size_t> & globalws, std::vector<size_t> & localws) const
{
	switch(gen)
	{
	case Stockham:		return GetWorkSizesPvt<Stockham>(globalws, localws);
	case Transpose:		return GetWorkSizesPvt<Transpose>(globalws, localws);
	case Copy:			return GetWorkSizesPvt<Copy>(globalws, localws);
	default:			assert(false); return CLFFT_NOTIMPLEMENTED;
	}
}

clfftStatus  FFTPlan::GetKernelGenKey (FFTKernelGenKeyParams & params) const
{
	switch(gen)
	{
	case Stockham:		return GetKernelGenKeyPvt<Stockham>(params);
	case Transpose:		return GetKernelGenKeyPvt<Transpose>(params);
	case Copy:			return GetKernelGenKeyPvt<Copy>(params);
	default:			assert(false); return CLFFT_NOTIMPLEMENTED;
	}
}

clfftStatus  FFTPlan::GenerateKernel (FFTRepo & fftRepo, const cl_command_queue commQueueFFT) const
{
	switch(gen)
	{
	case Stockham:		return GenerateKernelPvt<Stockham>(fftRepo, commQueueFFT);
	case Transpose:		return GenerateKernelPvt<Transpose>(fftRepo, commQueueFFT);
	case Copy:			return GenerateKernelPvt<Copy>(fftRepo, commQueueFFT);
	default:			assert(false); return CLFFT_NOTIMPLEMENTED;
	}
}

clfftStatus FFTPlan::GetMax1DLength (size_t *longest ) const
{
	switch(gen)
	{
	case Stockham:		return GetMax1DLengthPvt<Stockham>(longest);
	//No restriction for transpose kernel
	case Transpose:     *longest = 4096; return CLFFT_SUCCESS;
	case Copy:			*longest = 4096; return CLFFT_SUCCESS;
	default:			assert(false); return CLFFT_NOTIMPLEMENTED;
	}
}

clfftStatus FFTPlan::GetEnvelope (const FFTEnvelope ** ppEnvelope) const
{
	if(&envelope == NULL) assert(false);
	*ppEnvelope = &envelope;
	return CLFFT_SUCCESS;
}

size_t FFTPlan::ElementSize() const
{
	return ( ((precision == CLFFT_DOUBLE) || (precision == CLFFT_DOUBLE_FAST)) ? sizeof( std::complex<double> ) : sizeof( std::complex<float> ) );
}

