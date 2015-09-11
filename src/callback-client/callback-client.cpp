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
		/*runR2CPreprocessKernelFFT<float>(setupData, context, commandQueue, device_id[0], inlengths, dim, precision, 
										batchSize, vectorLength, fftLength, profile_count);*/

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
	size_t in_size_of_buffers = fftLength * sizeof(char) * 3 ;
	size_t out_size_of_buffers = fftLength * sizeof( T  );

	char* in24bitData = (char*)malloc(in_size_of_buffers);

	//Initialize Data
	for (size_t idx = 0; idx < fftLength; ++idx)
	{
		in24bitData[3 * idx + 2] = (char)(rand() % 256);
		in24bitData[3 * idx + 1] = (char)(rand() % 256);
		in24bitData[3 * idx] = (char)(rand() % 256);
	}

	//input data buffer
	cl_mem infftbuffer = ::clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in_size_of_buffers, (void*)in24bitData, &status);
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
	OPENCL_V_THROW (clfftSetPlanCallback(plan_handle, "convert24To32bit", precallbackstr, NULL, 0, PRECALLBACK, NULL, 0), "clFFTSetPlanCallback failed");

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

		for( cl_uint i = 0; i < fftLength/2; i++)
		{
			std::cout << "i " << i << " clreal " << output[i].real() << " climag " << output[i].imag() << std::endl;
		}

		////Reference fftw output
		//fftwf_complex *refout;

		//refout = get_R2C_fftwf_output(inlengths, fftLength, batchSize, inLayout, dim);

		///*for( cl_uint i = 0; i < fftLength; i++)
		//{
		//	std::cout << "i " << i << " refreal " << refout[i][0] << " refimag " << refout[i][1] << " clreal " << output[i].real() << " climag " << output[i].imag() << std::endl;
		//}*/
		//if (!compare<fftwf_complex, T>(refout, output, fftLength))
		//{
		//	std::cout << "\n\n\t\tInternal Client Test (with clFFT Pre-callback) *****FAIL*****" << std::endl;
		//}
		//else
		//{
		//	std::cout << "\n\n\t\tInternal Client Test (with clFFT Pre-callback) *****PASS*****" << std::endl;
		//}

		//fftwf_free(refout);
	}

	OPENCL_V_THROW( clfftDestroyPlan( &plan_handle ), "clfftDestroyPlan failed" );
	OPENCL_V_THROW( clfftTeardown( ), "clfftTeardown failed" );

	//cleanup
	OPENCL_V_THROW( clReleaseMemObject( infftbuffer ), "Error: In clReleaseMemObject\n" );
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
fftwf_complex* get_C2C_fftwf_output(size_t* lengths, size_t fftbatchLength, int batch_size, clfftLayout in_layout,
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
			if ( (i % fftVectorLength)  < USERDATA_LENGTH)
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
