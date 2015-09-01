#include "stdafx.h"
#include <functional>
#include <cmath>

#include "client.h"
#include "../library/private.h"
#include "openCL.misc.h"
#include "../statTimer/statisticalTimer.extern.h"
#include "../include/sharedLibrary.h"
#include "../include/unicode.compatibility.h"

#include <fftw3.h>

namespace po = boost::program_options;

enum FFTType
{
	FFT_C2C,
	FFT_R2C,
	FFT_C2R,
};

#define ZERO_PAD_C2C __attribute__((always_inline)) \n float2 zeroPad (__global void *input, \n \
								uint inoffset, \n \
							__global void *userdata) \n \
				 { \n \
					 float2 scalar = 0.0f; \n \
					 if (inoffset < 512) \n \
					 { \n \
						scalar = *((__global float2*)userdata + inoffset); \n \
					 } \n \
					 return scalar; \n \
				} \n

#define ZERO_PAD_C2C_KERNEL __kernel void zeroPad (__global void *input, \n \
								__global void *userdata, uint batchLength) \n \
				 { \n \
					uint inoffset = get_global_id(0); \n \
					 float2 scalar = 0.0f; \n \
					 if (inoffset < 512) \n \
					 { \n \
					   scalar = *((__global float2*)userdata + inoffset); \n \
					 } \n \
					 *((__global float2*)input + inoffset) = scalar; \n \
				} \n

//forward declarations

template < typename T >
void C2C_transform(std::auto_ptr< clfftSetupData > setupData, size_t* inlengths, size_t batchSize, 
				   clfftDim dim, clfftPrecision precision, cl_uint profile_count);

template < typename T >
void R2C_transform();

template < typename T >
void C2R_transform();

fftwf_complex* get_fftwf_output(size_t* lengths, size_t fftBatchSize, int batch_size, clfftLayout in_layout,
								clfftDim dim, clfftDirection dir);
template < typename T1, typename T2>
bool compare(T1 *refData, std::vector< std::complex< T2 > > data,
             size_t length, const float epsilon = 1e-6f);

template < typename T >
void runPrecallbackFFT(std::auto_ptr< clfftSetupData > setupData, cl_context context, cl_command_queue commandQueue, size_t* inlengths, clfftDim dim, clfftPrecision precision, size_t batchSize, size_t vectorLength, size_t fftLength, cl_uint profile_count);

template < typename T >
void runPreprocessKernelFFT(std::auto_ptr< clfftSetupData > setupData, cl_context context, cl_command_queue commandQueue, cl_device_id device_id, size_t* inlengths, clfftDim dim, clfftPrecision precision, size_t batchSize, size_t vectorLength, size_t fftLength, cl_uint profile_count);

int main(int argc, char **argv)
{
	size_t lengths[ 3 ] = {1,1,1};
	cl_uint profile_count = 0;
	clfftPrecision precision = CLFFT_SINGLE;

	size_t batchSize = 1; //For simplicity using fixed batch size

	int fftType = 1;
	FFTType clFFTType = FFT_C2C;

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
			( "dumpKernels,d", "FFT engine will dump generated OpenCL FFT kernels to disk (default: dump off)" )
			( "lenX,x",        po::value< size_t >( &lengths[ 0 ] )->default_value( 1024 ),   "Specify the length of the 1st dimension of a test array" )
			( "lenY,y",        po::value< size_t >( &lengths[ 1 ] )->default_value( 1 ),      "Specify the length of the 2nd dimension of a test array" )
			( "lenZ,z",        po::value< size_t >( &lengths[ 2 ] )->default_value( 1 ),      "Specify the length of the 3rd dimension of a test array" )
			( "profile,p",     po::value< cl_uint >( &profile_count )->default_value( 10 ), "Time and report the kernel speed of the FFT (default: profiling off)" )
			( "type,t",		   po::value< int	>( &fftType )->default_value( 1 ), "Type of FFT:\n1) Complex-Complex\n2) Real-Complex\n3) Complex-Real\n" )
			;

		po::variables_map vm;
		po::store( po::parse_command_line( argc, argv, desc ), vm );
		po::notify( vm );

		if( vm.count( "help" ) )
		{
			std::cout << desc << std::endl;
			return 0;
		}
		
		if( vm.count( "dumpKernels" ) )
		{
			setupData->debugFlags	|= CLFFT_DUMP_PROGRAMS;
		}

		switch (fftType)
		{
		case 1: clFFTType = FFT_C2C; break;
		case 2: clFFTType = FFT_R2C; break;
		case 3: clFFTType = FFT_C2R; break;
		default:
			throw std::runtime_error( "Invalid FFT type" );
			break;
		}
	
		clfftDim dim = CLFFT_1D;
		if( lengths[ 1 ] > 1 )
		{
			dim	= CLFFT_2D;
		}
		if( lengths[ 2 ] > 1 )
		{
			dim	= CLFFT_3D;
		}

		if( clFFTType == FFT_C2C) // Complex-Complex cases
		{
			C2C_transform<float>(setupData, lengths, batchSize, dim, precision, profile_count);
		}
		else if (clFFTType == FFT_R2C) // Complex-Complex cases
		{
			R2C_transform<float>();
		}
		else                          // Complex-Complex cases
		{
			C2R_transform<float>();
		}
	}
	catch( std::exception& e )
	{
		terr << _T( "clFFT error condition reported:" ) << std::endl << e.what() << std::endl;
		return 1;
	}
	return 0;
}

template < typename T >
void C2C_transform(std::auto_ptr< clfftSetupData > setupData, size_t* inlengths, size_t batchSize, 
				   clfftDim dim, clfftPrecision precision,  cl_uint profile_count)
{
	//	OpenCL state 
	cl_device_type		deviceType	= CL_DEVICE_TYPE_ALL;
	cl_int			deviceId = 0;
	std::vector< cl_device_id > device_id;
	cl_int				platformId = 0;
	cl_context			context;
	cl_uint command_queue_flags = 0;
	command_queue_flags |= CL_QUEUE_PROFILING_ENABLE;
	
	size_t vectorLength = inlengths[0] * inlengths[1] * inlengths[2];
	size_t fftLength = vectorLength * batchSize;

	//OpenCL initializations
	device_id = initializeCL( deviceType, deviceId, platformId, context, false);

	cl_int status = 0;
    
	cl_command_queue commandQueue = ::clCreateCommandQueue( context, device_id[0], command_queue_flags, &status );
    OPENCL_V_THROW( status, "Creating Command Queue ( ::clCreateCommandQueue() )" );

	//Run clFFT with seaparate Pre-process Kernel
	if (precision == CLFFT_SINGLE)
	{
	runPreprocessKernelFFT<float>(setupData, context, commandQueue, device_id[0], inlengths, dim, precision, batchSize, vectorLength, fftLength, profile_count);

	//Run clFFT using pre-callback 
	runPrecallbackFFT<float>(setupData, context, commandQueue, inlengths, dim, precision, batchSize, vectorLength, fftLength, profile_count);
	}

	OPENCL_V_THROW( clReleaseCommandQueue( commandQueue ), "Error: In clReleaseCommandQueue\n" );
    OPENCL_V_THROW( clReleaseContext( context ), "Error: In clReleaseContext\n" );
}

template < typename T >
void R2C_transform()
{
}

template < typename T >
void C2R_transform()
{
}

// Compute reference output using fftw for float type
fftwf_complex* get_fftwf_output(size_t* lengths, size_t fftbatchLength, int batch_size, clfftLayout in_layout,
								clfftDim dim, clfftDirection dir)
{
	//In FFTW last dimension has the fastest changing index
	int fftwLengths[3] = {(int)lengths[2], (int)lengths[1], (int)lengths[0]};

	fftwf_plan refPlan;

	fftwf_complex *refin = (fftwf_complex*) fftw_malloc(sizeof(fftwf_complex)*fftbatchLength);
	fftwf_complex *refout = (fftwf_complex*) fftw_malloc(sizeof(fftwf_complex)*fftbatchLength);

	size_t fftVectorLength = fftbatchLength/batch_size;

	refPlan = fftwf_plan_many_dft(dim, &fftwLengths[3 - dim], batch_size, 
									refin, &fftwLengths[3 - dim], 1, fftVectorLength, 
									refout, &fftwLengths[3 - dim], 1, fftVectorLength, 
									dir, FFTW_ESTIMATE);
	
	float scalar; 
	
	for( size_t i = 0; i < fftbatchLength; i++)
	{
		scalar = 0.0f;
		switch (in_layout)
		{
		case CLFFT_COMPLEX_INTERLEAVED:
			if ( (i % fftVectorLength)  < 512)
			{
				scalar = 1.0f;
			}
			break;
		default:
			break;
		}

		refin[i][0] = scalar;
		refin[i][1] = 0;
	}

	fftwf_execute(refPlan);

	fftw_free(refin);

	fftwf_destroy_plan(refPlan);

	return refout;
}

template < typename T >
void runPrecallbackFFT(std::auto_ptr< clfftSetupData > setupData, cl_context context, cl_command_queue commandQueue,
						size_t* inlengths, clfftDim dim, clfftPrecision precision,
						size_t batchSize, size_t vectorLength, size_t fftLength, cl_uint profile_count)
{
	cl_int status = 0;

	size_t userdataLengths[ 3 ] = {512,1,1};
	size_t vectorLength_userdata = userdataLengths[0] * userdataLengths[1] * userdataLengths[2];
	size_t userdataLength = vectorLength_userdata * batchSize;

	//input/output allocation sizes
	size_t size_of_buffers = fftLength * sizeof( std::complex< T > );
	size_t size_of_buffers_userdata = userdataLength * sizeof( std::complex< T > );

	//in-place transform. Same buffer for input and output
	cl_mem fftbuffer = ::clCreateBuffer( context, CL_MEM_READ_WRITE, size_of_buffers, NULL, &status);
    OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer(buffer) )" );

	//Initialize Data
	std::vector< std::complex< T > > userdata( userdataLength );

	// impulse test case
	std::complex< T > impulsedata(1,0);
	for (size_t idx = 0; idx < userdataLength; ++idx)
	{
		userdata[idx] = impulsedata;
	}

	//user data buffer
	cl_mem userDatabuffer = ::clCreateBuffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_of_buffers_userdata, &userdata[0], &status);
    OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer(userDatabuffer) )" );

	//clFFT initializations
	
	//	FFT state
	clfftResultLocation	place = CLFFT_INPLACE;
	clfftLayout	inLayout  = CLFFT_COMPLEX_INTERLEAVED;
	clfftLayout	outLayout = CLFFT_COMPLEX_INTERLEAVED;
	clfftDirection dir = CLFFT_FORWARD;

	clfftPlanHandle plan_handle;
	OPENCL_V_THROW( clfftSetup( setupData.get( ) ), "clfftSetup failed" );
	OPENCL_V_THROW( clfftCreateDefaultPlan( &plan_handle, context, dim, inlengths ), "clfftCreateDefaultPlan failed" );

	//Precallback setup
	char* precallbackstr = STRINGIFY(ZERO_PAD_C2C);

	//Register the callback
	OPENCL_V_THROW (clFFTSetPlanCallback(plan_handle, "zeroPad", precallbackstr, NULL, 0, PRECALLBACK, userDatabuffer), "clFFTSetPlanCallback failed");

	//	Default plan creates a plan that expects an inPlace transform with interleaved complex numbers
	OPENCL_V_THROW( clfftSetResultLocation( plan_handle, place ), "clfftSetResultLocation failed" );
	OPENCL_V_THROW( clfftSetLayout( plan_handle, inLayout, outLayout ), "clfftSetLayout failed" );
	OPENCL_V_THROW( clfftSetPlanBatchSize( plan_handle, batchSize ), "clfftSetPlanBatchSize failed" );
	OPENCL_V_THROW( clfftSetPlanPrecision( plan_handle, precision ), "clfftSetPlanPrecision failed" );

	//Bake Plan
	OPENCL_V_THROW( clfftBakePlan( plan_handle, 1, &commandQueue, NULL, NULL ), "clfftBakePlan failed" );

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

	cl_mem * buffersOut = NULL; //NULL for in-place

	Timer tr;
	double wtime_acc = 0.0;

	//	Loop as many times as the user specifies to average out the timings
	for( cl_uint i = 0; i < profile_count; ++i )
	{
		tr.Start();
		OPENCL_V_THROW( clfftEnqueueTransform( plan_handle, dir, 1, &commandQueue, 0, NULL, NULL,
			&fftbuffer, buffersOut, clMedBuffer ),
			"clfftEnqueueTransform failed" );
		
		OPENCL_V_THROW( clFinish( commandQueue ), "clFinish failed" );

		wtime_acc += tr.Sample();

		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_acc = 0.0;
	}
	double wtime = wtime_acc/((double)profile_count);
	size_t totalLen = 1;
	for(int i=0; i<dim; i++) totalLen *= inlengths[i];
	double opsconst = 5.0 * (double)totalLen * log((double)totalLen) / log(2.0);

	tout << "\nExecution wall time (with clFFT Pre-callback): " << 1000.0*wtime << " ms" << std::endl;
	tout << "Execution gflops (with clFFT Pre-callback): " << ((double)batchSize * opsconst)/(1000000000.0*wtime) << std::endl;
	
	if(clMedBuffer) clReleaseMemObject(clMedBuffer);
	
	if (profile_count == 1)
	{
		std::vector< std::complex< T > > output( fftLength );

		OPENCL_V_THROW( clEnqueueReadBuffer( commandQueue, fftbuffer, CL_TRUE, 0, size_of_buffers, &output[ 0 ],
			0, NULL, NULL ), "Reading the result buffer failed" );

		//Reference fftw output
		fftwf_complex *refout;

		refout = get_fftwf_output(inlengths, fftLength, batchSize, inLayout, dim, dir);

		/*for( cl_uint i = 0; i < fftLength; i++)
		{
			std::cout << "i " << i << " refreal " << refout[i][0] << " refimag " << refout[i][1] << " clreal " << output[i].real() << " climag " << output[i].imag() << std::endl;
		}*/
		if (!compare<fftwf_complex, T>(refout, output, fftLength))
		{
			std::cout << "\n\n\t\tInternal Client Test *****FAIL*****" << std::endl;
		}
		else
		{
			std::cout << "\n\n\t\tInternal Client Test *****PASS*****" << std::endl;
		}

		fftwf_free(refout);
	}

	OPENCL_V_THROW( clfftDestroyPlan( &plan_handle ), "clfftDestroyPlan failed" );
	OPENCL_V_THROW( clfftTeardown( ), "clfftTeardown failed" );

	//cleanup
	OPENCL_V_THROW( clReleaseMemObject( fftbuffer ), "Error: In clReleaseMemObject\n" );
	OPENCL_V_THROW( clReleaseMemObject( userDatabuffer ), "Error: In clReleaseMemObject\n" );
}

template < typename T >
void runPreprocessKernelFFT(std::auto_ptr< clfftSetupData > setupData, cl_context context, 
							cl_command_queue commandQueue, cl_device_id device_id,
							size_t* inlengths, clfftDim dim, clfftPrecision precision,
							size_t batchSize, size_t vectorLength, size_t fftLength, cl_uint profile_count)
{
	cl_int status = 0;

	size_t userdataLengths[ 3 ] = {512,1,1};
	size_t vectorLength_userdata = userdataLengths[0] * userdataLengths[1] * userdataLengths[2];
	size_t userdataLength = vectorLength_userdata * batchSize;

	//input/output allocation sizes
	size_t size_of_buffers = fftLength * sizeof( std::complex< T > );
	size_t size_of_buffers_userdata = userdataLength * sizeof( std::complex< T > );

	//in-place transform. Same buffer for input and output
	cl_mem fftbuffer = ::clCreateBuffer( context, CL_MEM_READ_WRITE, size_of_buffers, NULL, &status);
    OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer(buffer) )" );

	//Initialize Data
	std::vector< std::complex< T > > userdata( userdataLength );

	// impulse test case
	std::complex< T > impulsedata(1,0);
	for (size_t idx = 0; idx < userdataLength; ++idx)
	{
		userdata[idx] = impulsedata;
	}

	//user data buffer
	cl_mem userdatabuffer = ::clCreateBuffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_of_buffers_userdata, &userdata[0], &status);
    OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer(userdatabuffer) )" );

	//clFFT initializations

	//	FFT state
	clfftResultLocation	place = CLFFT_INPLACE;
	clfftLayout	inLayout  = CLFFT_COMPLEX_INTERLEAVED;
	clfftLayout	outLayout = CLFFT_COMPLEX_INTERLEAVED;
	clfftDirection dir = CLFFT_FORWARD;

	clfftPlanHandle plan_handle;
	OPENCL_V_THROW( clfftSetup( setupData.get( ) ), "clfftSetup failed" );
	OPENCL_V_THROW( clfftCreateDefaultPlan( &plan_handle, context, dim, inlengths ), "clfftCreateDefaultPlan failed" );

	//	Default plan creates a plan that expects an inPlace transform with interleaved complex numbers
	OPENCL_V_THROW( clfftSetResultLocation( plan_handle, place ), "clfftSetResultLocation failed" );
	OPENCL_V_THROW( clfftSetLayout( plan_handle, inLayout, outLayout ), "clfftSetLayout failed" );
	OPENCL_V_THROW( clfftSetPlanBatchSize( plan_handle, batchSize ), "clfftSetPlanBatchSize failed" );
	OPENCL_V_THROW( clfftSetPlanPrecision( plan_handle, precision ), "clfftSetPlanPrecision failed" );

		//Bake Plan
	OPENCL_V_THROW( clfftBakePlan( plan_handle, 1, &commandQueue, NULL, NULL ), "clfftBakePlan failed" );

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

	cl_mem * buffersOut = NULL; //NULL for in-place

	//Pre-process kernel string
	const char* source = STRINGIFY(ZERO_PAD_C2C_KERNEL);
	
	double wtime_acc = 0.0;
	Timer tr;
		
	//	Loop as many times as the user specifies to average out the timings
	for( cl_uint i = 0; i < profile_count; ++i )
	{
		tr.Start();
		cl_program program = clCreateProgramWithSource( context, 1, &source, NULL, &status );
		OPENCL_V_THROW( status, "clCreateProgramWithSource failed." );

		status = clBuildProgram( program, 1, &device_id, NULL, NULL, NULL);
		OPENCL_V_THROW( status, "clBuildProgram failed" );

#if defined( _DEBUG )
		if( status != CL_SUCCESS )
		{
			if( status == CL_BUILD_PROGRAM_FAILURE )
			{
				size_t buildLogSize = 0;
				OPENCL_V_THROW( clGetProgramBuildInfo( program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize ),
								"clGetProgramBuildInfo failed"  );

				std::vector< char > buildLog( buildLogSize );
				::memset( &buildLog[ 0 ], 0x0, buildLogSize );

				OPENCL_V_THROW( clGetProgramBuildInfo( program, device_id, CL_PROGRAM_BUILD_LOG, buildLogSize, &buildLog[ 0 ], NULL ),
							"clGetProgramBuildInfo failed"  );

				std::cerr << "\n\t\t\tBUILD LOG\n";
				std::cerr << "************************************************\n";
				std::cerr << &buildLog[ 0 ] << std::endl;
				std::cerr << "************************************************\n";
			}

			OPENCL_V_THROW( status, "clBuildProgram failed" );
		}
#endif

		cl_kernel kernel = clCreateKernel( program, "zeroPad", &status );
		OPENCL_V_THROW( status, "clCreateKernel failed" );

		cl_uint uarg = 0;

		//Buffer to be zero-padded
		OPENCL_V_THROW( clSetKernelArg( kernel, uarg++, sizeof( cl_mem ), (void*)&fftbuffer ), "clSetKernelArg failed" );

		//originial data
		OPENCL_V_THROW( clSetKernelArg( kernel, uarg++, sizeof( cl_mem ), (void*)&userdatabuffer ), "clSetKernelArg failed" );

		//single batch length 
		OPENCL_V_THROW( clSetKernelArg( kernel, uarg++, sizeof( cl_uint ), (const void *)&vectorLength ), "clSetKernelArg failed" );

		//Launch pre-process kernel
		size_t gSize = fftLength;
		size_t lSize = 64;
		status = clEnqueueNDRangeKernel( commandQueue, kernel, 1,
												NULL, &gSize, &lSize, 0, NULL, NULL );
		OPENCL_V_THROW( status, "clEnqueueNDRangeKernel failed" );
	
		OPENCL_V_THROW( clFinish( commandQueue ), "clFinish failed" );

		//Now invoke the clfft execute
		OPENCL_V_THROW( clfftEnqueueTransform( plan_handle, dir, 1, &commandQueue, 0, NULL, NULL,
			&fftbuffer, buffersOut, clMedBuffer ),
			"clfftEnqueueTransform failed" );
		
		OPENCL_V_THROW( clFinish( commandQueue ), "clFinish failed" );

		wtime_acc += tr.Sample();

		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_acc = 0.0;

		//cleanup preprocess kernel opencl objects
		OPENCL_V_THROW( clReleaseProgram( program ), "Error: In clReleaseProgram\n" );
		OPENCL_V_THROW( clReleaseKernel( kernel ), "Error: In clReleaseKernel\n" );
	}

	double wtime = wtime_acc/((double)profile_count);
	size_t totalLen = 1;
	for(int i=0; i<dim; i++) totalLen *= inlengths[i];
	double opsconst = 5.0 * (double)totalLen * log((double)totalLen) / log(2.0);

	tout << "\nExecution wall time (Separate Pre-process Kernel): " << 1000.0*wtime << " ms" << std::endl;
	tout << "Execution gflops (Separate Pre-process Kernel): " << ((double)batchSize * opsconst)/(1000000000.0*wtime) << std::endl;
	
	if(clMedBuffer) clReleaseMemObject(clMedBuffer);

	if (profile_count == 1)
	{
		std::vector< std::complex< T > > output( fftLength );

		OPENCL_V_THROW( clEnqueueReadBuffer( commandQueue, fftbuffer, CL_TRUE, 0, size_of_buffers, &output[ 0 ],
			0, NULL, NULL ), "Reading the result buffer failed" );

		//Reference fftw output
		fftwf_complex *refout;

		refout = get_fftwf_output(inlengths, fftLength, batchSize, inLayout, dim, dir);

		/*for( cl_uint i = 0; i < fftLength; i++)
		{
			std::cout << "i " << i << " refreal " << refout[i][0] << " refimag " << refout[i][1] << " clreal " << output[i].real() << " climag " << output[i].imag() << std::endl;
		}*/
		if (!compare<fftwf_complex, T>(refout, output, fftLength))
		{
			std::cout << "\n\n\t\tInternal Client Test *****FAIL*****" << std::endl;
		}
		else
		{
			std::cout << "\n\n\t\tInternal Client Test *****PASS*****" << std::endl;
		}

		fftwf_free(refout);
	}

	OPENCL_V_THROW( clfftDestroyPlan( &plan_handle ), "clfftDestroyPlan failed" );
	OPENCL_V_THROW( clfftTeardown( ), "clfftTeardown failed" );

	//cleanup
	OPENCL_V_THROW( clReleaseMemObject( fftbuffer ), "Error: In clReleaseMemObject\n" );
	OPENCL_V_THROW( clReleaseMemObject( userdatabuffer ), "Error: In clReleaseMemObject\n" );
}

//Compare reference and opencl output 
template < typename T1, typename T2>
bool compare(T1 *refData, std::vector< std::complex< T2 > > data,
             size_t length, const float epsilon)
{
    float error = 0.0f;
    T1 ref;
	T1 diff;
	float normRef = 0.0f;
	float normError = 0.0f;

    for(size_t i = 0; i < length; ++i)
    {
        diff[0] = refData[i][0] - data[i].real();
        error += (float)(diff[0] * diff[0]);
        ref[0] += refData[i][0] * refData[i][0];
    }
	if (error != 0)
	{
		normRef =::sqrtf((float) ref[0]);
		if (::fabs((float) ref[0]) < 1e-7f)
		{
			return false;
		}
		normError = ::sqrtf((float) error);
		error = normError / normRef;
    
		if (error > epsilon)
			return false;
	}

	//imag
	error = 0.0f;
	ref[1] = 0.0;
	for(size_t i = 0; i < length; ++i)
    {
        diff[1] = refData[i][1] - data[i].imag();
        error += (float)(diff[1] * diff[1]);
        ref[1] += refData[i][1] * refData[i][1];
    }
	
	if (error == 0)
		return true;

	normRef =::sqrtf((float) ref[1]);
    if (::fabs((float) ref[1]) < 1e-7f)
    {
        return false;
    }
	normError = ::sqrtf((float) error);
    error = normError / normRef;
    
	if (error > epsilon)
		return false;

	return true;
}
