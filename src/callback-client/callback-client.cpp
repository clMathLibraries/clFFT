#include <functional>
#include <cmath>

#include "client.h"
#include "../library/private.h"
#include "openCL.misc.h"
#include "../include/sharedLibrary.h"

namespace po = boost::program_options;

int main(int argc, char **argv)
{
	size_t lengths[ 3 ] = {BATCH_LENGTH,1,1}; //For simplicity, assuming 1D fft with fixed batch length of BATCH_LENGTH
	cl_uint profile_count = 0;
	clfftPrecision precision = CLFFT_SINGLE; //Testing for single precision. Easily extendable for double

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
			( "dumpKernels,d", "FFT engine will dump generated OpenCL FFT kernels to disk (default: dump off)" )
			( "batchSize,b",   po::value< size_t >( &batchSize )->default_value( 1 ), "If this value is greater than one, arrays will be used " )
			( "profile,p",     po::value< cl_uint >( &profile_count )->default_value( 10 ), "Time and report the kernel speed of the FFT (default: profiling off)" )
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
			
		clfftDim dim = CLFFT_1D;
		if( lengths[ 1 ] > 1 )
		{
			dim	= CLFFT_2D;
		}
		if( lengths[ 2 ] > 1 )
		{
			dim	= CLFFT_3D;
		}

		 // Real-Complex cases, SP
		
		R2C_transform<float>(setupData, lengths, batchSize, dim, precision, profile_count);
		
	}
	catch( std::exception& e )
	{
		terr << _T( "clFFT error condition reported:" ) << std::endl << e.what() << std::endl;
		return 1;
	}
	return 0;
}

template < typename T >
void R2C_transform(std::auto_ptr< clfftSetupData > setupData, size_t* inlengths, size_t batchSize, 
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
	
	// Test for in-place Hermitian Interleaved output 
	// Hence output size is N/2 + 1 complex. So allocate N + 2 real input
	size_t Nt = inlengths[0] + 2;
	size_t vectorLength = Nt * inlengths[1] * inlengths[2];
	size_t fftLength = vectorLength * batchSize;

	//OpenCL initializations
	device_id = initializeCL( deviceType, deviceId, platformId, context, false);

	cl_int status = 0;
    
	cl_command_queue commandQueue = ::clCreateCommandQueue( context, device_id[0], command_queue_flags, &status );
    OPENCL_V_THROW( status, "Creating Command Queue ( ::clCreateCommandQueue() )" );

	if (precision == CLFFT_SINGLE)
	{
		//Run clFFT with seaparate Pre-process Kernel
		runR2CPreprocessKernelFFT<float>(setupData, context, commandQueue, device_id[0], inlengths, dim, precision, 
										batchSize, vectorLength, fftLength, profile_count);

		//Run clFFT using pre-callback 
		runR2CPrecallbackFFT<float>(setupData, context, commandQueue, inlengths, dim, precision, 
									batchSize, vectorLength, fftLength, profile_count);
	}

	OPENCL_V_THROW( clReleaseCommandQueue( commandQueue ), "Error: In clReleaseCommandQueue\n" );
    OPENCL_V_THROW( clReleaseContext( context ), "Error: In clReleaseContext\n" );
}

template < typename T >
void runR2CPrecallbackFFT(std::auto_ptr< clfftSetupData > setupData, cl_context context, cl_command_queue commandQueue,
						size_t* inlengths, clfftDim dim, clfftPrecision precision,
						size_t batchSize, size_t vectorLength, size_t fftLength, cl_uint profile_count)
{
	cl_int status = 0;
	
	//input/output allocation sizes
	size_t in_size_of_buffers = fftLength * sizeof(uint24_t);
	size_t out_size_of_buffers = fftLength * sizeof( T  );

	uint24_t *input24bitData = (uint24_t*)malloc(in_size_of_buffers);

	//Initialize Data
	srand(1);
	for (size_t idx = 0; idx < fftLength; ++idx)
	{
		int randomVal = (int)rand();

		input24bitData[idx][0] = (randomVal >> 16) & 0xFF;
        input24bitData[idx][1] = (randomVal >> 8) & 0xFF;
        input24bitData[idx][2] = randomVal & 0xFF; 
	}
	
	//input data buffer
	cl_mem infftbuffer = ::clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in_size_of_buffers, (void*)input24bitData, &status);
    OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer(infftbuffer) )" );

	//out-place transform. 	
	cl_mem outfftbuffer = ::clCreateBuffer( context, CL_MEM_READ_WRITE, out_size_of_buffers, NULL, &status);
    OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer(oufftbuffer) )" );

	//clFFT initializations
	
	//	FFT state
	clfftResultLocation	place = CLFFT_OUTOFPLACE;
	clfftLayout	inLayout  = CLFFT_REAL;
	clfftLayout	outLayout = CLFFT_HERMITIAN_INTERLEAVED;

	clfftPlanHandle plan_handle;
	OPENCL_V_THROW( clfftSetup( setupData.get( ) ), "clfftSetup failed" );
	OPENCL_V_THROW( clfftCreateDefaultPlan( &plan_handle, context, dim, inlengths ), "clfftCreateDefaultPlan failed" );

	//Precallback setup
	char* precallbackstr = STRINGIFY(ConvertToFloat);

	//Register the callback
	OPENCL_V_THROW (clfftSetPlanCallback(plan_handle, "convert24To32bit", precallbackstr, 0, PRECALLBACK, NULL, 0), "clFFTSetPlanCallback failed");

	//	Default plan creates a plan that expects an inPlace transform with interleaved complex numbers
	OPENCL_V_THROW( clfftSetResultLocation( plan_handle, place ), "clfftSetResultLocation failed" );
	OPENCL_V_THROW( clfftSetLayout( plan_handle, inLayout, outLayout ), "clfftSetLayout failed" );
	OPENCL_V_THROW( clfftSetPlanBatchSize( plan_handle, batchSize ), "clfftSetPlanBatchSize failed" );
	OPENCL_V_THROW( clfftSetPlanPrecision( plan_handle, precision ), "clfftSetPlanPrecision failed" );
	OPENCL_V_THROW( clfftSetPlanDistance( plan_handle, BATCH_LENGTH + 2, (BATCH_LENGTH/2 + 1)), "clfftSetPlanDistance failed" );

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

	//for functional test
	OPENCL_V_THROW( clfftEnqueueTransform( plan_handle, CLFFT_FORWARD, 1, &commandQueue, 0, NULL, NULL,
		&infftbuffer, &outfftbuffer, clMedBuffer ),
		"clfftEnqueueTransform failed" );
		
	OPENCL_V_THROW( clFinish( commandQueue ), "clFinish failed" );

	if (profile_count > 1)
	{
		Timer tr;
		tr.Start();

		//	Loop as many times as the user specifies to average out the timings
		for( cl_uint i = 0; i < profile_count; ++i )
		{
			OPENCL_V_THROW( clfftEnqueueTransform( plan_handle, CLFFT_FORWARD, 1, &commandQueue, 0, NULL, NULL,
				&infftbuffer, &outfftbuffer, clMedBuffer ),
				"clfftEnqueueTransform failed" );
		
			OPENCL_V_THROW( clFinish( commandQueue ), "clFinish failed" );
		}
		double wtimesample = tr.Sample();
		double wtime = wtimesample/((double)profile_count);
	
		tout << "\nExecution wall time (with clFFT Pre-callback): " << 1000.0*wtime << " ms" << std::endl;
	}

	if(clMedBuffer) clReleaseMemObject(clMedBuffer);
	
	if (profile_count == 1)
	{
		std::vector< std::complex< T > > output( fftLength/2 );

		OPENCL_V_THROW( clEnqueueReadBuffer( commandQueue, outfftbuffer, CL_TRUE, 0, out_size_of_buffers, &output[ 0 ],
			0, NULL, NULL ), "Reading the result buffer failed" );

		//Reference fftw output
		fftwf_complex *refout;

		refout = get_R2C_fftwf_output(inlengths, fftLength, (int)batchSize, inLayout, dim);

		if (!compare<fftwf_complex, T>(refout, output, fftLength/2))
		{
			std::cout << "\n\n\t\tInternal Client Test (with clFFT Pre-callback) *****FAIL*****" << std::endl;
		}
		else
		{
			std::cout << "\n\n\t\tInternal Client Test (with clFFT Pre-callback) *****PASS*****" << std::endl;
		}

		fftwf_free(refout);
	}

	OPENCL_V_THROW( clfftDestroyPlan( &plan_handle ), "clfftDestroyPlan failed" );
	OPENCL_V_THROW( clfftTeardown( ), "clfftTeardown failed" );

	//cleanup
	OPENCL_V_THROW( clReleaseMemObject( infftbuffer ), "Error: In clReleaseMemObject\n" );
	OPENCL_V_THROW( clReleaseMemObject( outfftbuffer ), "Error: In clReleaseMemObject\n" );
}

template < typename T >
void runR2CPreprocessKernelFFT(std::auto_ptr< clfftSetupData > setupData, cl_context context, 
							cl_command_queue commandQueue, cl_device_id device_id,
							size_t* inlengths, clfftDim dim, clfftPrecision precision,
							size_t batchSize, size_t vectorLength, size_t fftLength, cl_uint profile_count)
{
	cl_int status = 0;

	//input/output allocation sizes
	size_t in_size_of_buffers = fftLength * sizeof(uint24_t);
	size_t out_size_of_buffers = fftLength * sizeof( T  );

	uint24_t *input24bitData = (uint24_t*)malloc(in_size_of_buffers);

	//Initialize Data
	srand(1);
	for (size_t idx = 0; idx < fftLength; ++idx)
	{
		int randomVal = (int)rand();

		input24bitData[idx][0] = (randomVal >> 16) & 0xFF;
        input24bitData[idx][1] = (randomVal >> 8) & 0xFF;
        input24bitData[idx][2] = randomVal & 0xFF; 
	}

	//input data buffer
	cl_mem in24bitfftbuffer = ::clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in_size_of_buffers, (void*)input24bitData, &status);
    OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer(in24bitfftbuffer) )" );

	cl_mem in32bitfftbuffer = ::clCreateBuffer( context, CL_MEM_READ_WRITE, out_size_of_buffers, NULL, &status);
    OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer(in32bitfftbuffer) )" );

	//out-place transform. 	
	cl_mem outfftbuffer = ::clCreateBuffer( context, CL_MEM_READ_WRITE, out_size_of_buffers, NULL, &status);
    OPENCL_V_THROW( status, "Creating Buffer ( ::clCreateBuffer(oufftbuffer) )" );

	//clFFT initializations
	
	//	FFT state
	clfftResultLocation	place = CLFFT_OUTOFPLACE;
	clfftLayout	inLayout  = CLFFT_REAL;
	clfftLayout	outLayout = CLFFT_HERMITIAN_INTERLEAVED;

	clfftPlanHandle plan_handle;
	OPENCL_V_THROW( clfftSetup( setupData.get( ) ), "clfftSetup failed" );
	OPENCL_V_THROW( clfftCreateDefaultPlan( &plan_handle, context, dim, inlengths ), "clfftCreateDefaultPlan failed" );

	//	Default plan creates a plan that expects an inPlace transform with interleaved complex numbers
	OPENCL_V_THROW( clfftSetResultLocation( plan_handle, place ), "clfftSetResultLocation failed" );
	OPENCL_V_THROW( clfftSetLayout( plan_handle, inLayout, outLayout ), "clfftSetLayout failed" );
	OPENCL_V_THROW( clfftSetPlanBatchSize( plan_handle, batchSize ), "clfftSetPlanBatchSize failed" );
	OPENCL_V_THROW( clfftSetPlanPrecision( plan_handle, precision ), "clfftSetPlanPrecision failed" );
	OPENCL_V_THROW( clfftSetPlanDistance( plan_handle, BATCH_LENGTH + 2, (BATCH_LENGTH/2 + 1)), "clfftSetPlanDistance failed" );

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

	//Pre-process kernel string
	const char* source = STRINGIFY(ConvertToFloat_KERNEL);
	
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

	cl_kernel kernel = clCreateKernel( program, "convert24To32bit", &status );
	OPENCL_V_THROW( status, "clCreateKernel failed" );

	//for functional test
	cl_uint uarg = 0;

	//Input 24bit Buffer 
	OPENCL_V_THROW( clSetKernelArg( kernel, uarg++, sizeof( cl_mem ), (void*)&in24bitfftbuffer ), "clSetKernelArg failed" );
	
	//output 32bit Buffer 
	OPENCL_V_THROW( clSetKernelArg( kernel, uarg++, sizeof( cl_mem ), (void*)&in32bitfftbuffer ), "clSetKernelArg failed" );

	//Launch pre-process kernel
	size_t gSize = fftLength;
	status = clEnqueueNDRangeKernel( commandQueue, kernel, 1,
											NULL, &gSize, NULL, 0, NULL, NULL );
	OPENCL_V_THROW( status, "clEnqueueNDRangeKernel failed" );
	
	OPENCL_V_THROW( clFinish( commandQueue ), "clFinish failed" );

	//Now invoke the clfft execute
	OPENCL_V_THROW( clfftEnqueueTransform( plan_handle, CLFFT_FORWARD, 1, &commandQueue, 0, NULL, NULL,
		&in32bitfftbuffer, &outfftbuffer, clMedBuffer ),
		"clfftEnqueueTransform failed" );
		
	OPENCL_V_THROW( clFinish( commandQueue ), "clFinish failed" );
	
	if (profile_count > 1)
	{
		Timer tr;
		tr.Start();

		//	Loop as many times as the user specifies to average out the timings
		for( cl_uint i = 0; i < profile_count; ++i )
		{
			uarg = 0;

			//Input 24bit Buffer 
			OPENCL_V_THROW( clSetKernelArg( kernel, uarg++, sizeof( cl_mem ), (void*)&in24bitfftbuffer ), "clSetKernelArg failed" );
	
			//output 32bit Buffer 
			OPENCL_V_THROW( clSetKernelArg( kernel, uarg++, sizeof( cl_mem ), (void*)&in32bitfftbuffer ), "clSetKernelArg failed" );

			//Launch pre-process kernel
			status = clEnqueueNDRangeKernel( commandQueue, kernel, 1,
													NULL, &gSize, NULL, 0, NULL, NULL );
			OPENCL_V_THROW( status, "clEnqueueNDRangeKernel failed" );
	
			OPENCL_V_THROW( clFinish( commandQueue ), "clFinish failed" );

			//Now invoke the clfft execute
			OPENCL_V_THROW( clfftEnqueueTransform( plan_handle, CLFFT_FORWARD, 1, &commandQueue, 0, NULL, NULL,
				&in32bitfftbuffer,  &outfftbuffer, clMedBuffer ),
				"clfftEnqueueTransform failed" );
		
			OPENCL_V_THROW( clFinish( commandQueue ), "clFinish failed" );
		}
		double wtimesample = tr.Sample();
		double wtime = wtimesample/((double)profile_count);
	
		tout << "\nExecution wall time (Separate Pre-process Kernel): " << 1000.0*wtime << " ms" << std::endl;
	}

	//cleanup preprocess kernel opencl objects
	OPENCL_V_THROW( clReleaseProgram( program ), "Error: In clReleaseProgram\n" );
	OPENCL_V_THROW( clReleaseKernel( kernel ), "Error: In clReleaseKernel\n" );

	if(clMedBuffer) clReleaseMemObject(clMedBuffer);

	if (profile_count == 1)
	{
		std::vector< std::complex< T > > output( fftLength/2 );

		OPENCL_V_THROW( clEnqueueReadBuffer( commandQueue, outfftbuffer, CL_TRUE, 0, out_size_of_buffers, &output[ 0 ],
			0, NULL, NULL ), "Reading the result buffer failed" );

		//Reference fftw output
		fftwf_complex *refout;

		refout = get_R2C_fftwf_output(inlengths, fftLength, (int)batchSize, inLayout, dim);

		/*for( cl_uint i = 0; i < fftLength/2; i++)
		{
			std::cout << "i " << i << " refreal " << refout[i][0] << " refimag " << refout[i][1] << " clreal " << output[i].real() << " climag " << output[i].imag() << std::endl;
		}*/
		if (!compare<fftwf_complex, T>(refout, output, fftLength/2))
		{
			std::cout << "\n\n\t\tInternal Client Test (Separate Pre-process Kernel) *****FAIL*****" << std::endl;
		}
		else
		{
			std::cout << "\n\n\t\tInternal Client Test (Separate Pre-process Kernel) *****PASS*****" << std::endl;
		}

		fftwf_free(refout);
	}

	OPENCL_V_THROW( clfftDestroyPlan( &plan_handle ), "clfftDestroyPlan failed" );
	OPENCL_V_THROW( clfftTeardown( ), "clfftTeardown failed" );

	//cleanup
	OPENCL_V_THROW( clReleaseMemObject( in24bitfftbuffer ), "Error: In clReleaseMemObject\n" );
	OPENCL_V_THROW( clReleaseMemObject( in32bitfftbuffer ), "Error: In clReleaseMemObject\n" );
	OPENCL_V_THROW( clReleaseMemObject( outfftbuffer ), "Error: In clReleaseMemObject\n" );
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


// Compute reference output using fftw for float type
fftwf_complex* get_R2C_fftwf_output(size_t* lengths, size_t fftbatchLength, int batch_size,
									clfftLayout in_layout, clfftDim dim)
{
	//In FFTW last dimension has the fastest changing index
	int fftwLengths[3] = {(int)lengths[2], (int)lengths[1], (int)lengths[0]};
	int inembed[3] = {(int)lengths[2], (int)lengths[1], (int)(lengths[0] + 2)};
	int outembed[3] = {(int)lengths[2], (int)lengths[1], (int)(lengths[0]/2 + 1)};

	fftwf_plan refPlan;
		
	int infftVectorLength = inembed[0] * inembed[1] * inembed[2];
	int outfftVectorLength = outembed[0] * outembed[1] * outembed[2];

	float *refin = (float*) malloc(sizeof(float)*fftbatchLength);
	fftwf_complex *refout = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*outfftVectorLength*batch_size);

	refPlan = fftwf_plan_many_dft_r2c(dim, &fftwLengths[3 - dim], batch_size, 
									refin, &inembed[3 - dim], 1, infftVectorLength,
									refout, &outembed[3 - dim], 1, outfftVectorLength, FFTW_ESTIMATE);
	
	uint24_t* in24bitData = (uint24_t*)malloc(sizeof(uint24_t) * fftbatchLength);

	//Initialize Data
	srand(1);
	for (size_t idx = 0; idx < fftbatchLength; ++idx)
	{
		int randomVal = (int)rand();

		in24bitData[idx][0] = (randomVal >> 16) & 0xFF;
        in24bitData[idx][1] = (randomVal >> 8) & 0xFF;
        in24bitData[idx][2] = randomVal & 0xFF; 
	}

	float val; 
	
	for( size_t i = 0; i < fftbatchLength; i++)
	{
		val = (float)(in24bitData[i][0] << 16 | in24bitData[i][1] << 8 | in24bitData[i][2]) ;
		
		refin[i] = val;
	}

	fftwf_execute(refPlan);

	free(refin);

	fftwf_destroy_plan(refPlan);

	return refout;
}
