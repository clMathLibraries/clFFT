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

#define SCALAR 10

#define MULVAL float2 mulval(__global void* in, uint offset, __global void* userdata)\n \
				{ \n \
				int scalar = *((__global int*)userdata + offset); \n \
				float2 ret = *((__global float2*)in + offset) * scalar; \n \
				return ret; \n \
				}

#define MULVAL_C2R float2 mulval(__global void* in, uint offset, __global void* userdata)\n \
				{ \n \
				int scalar = *((__global int*)userdata + offset); \n \
				float2 ret = *((__global float2*)in + offset) * scalar; \n \
				return ret; \n \
				}

#define MULVAL_DP double2 mulval(__global void* in, uint offset, __global void* userdata)\n \
				{ \n \
				int scalar = *((__global int*)userdata + offset); \n \
				double2 ret = *((__global double2*)in + offset) * scalar; \n \
				return ret; \n \
				}

#define MULVAL_PLANAR float2 mulval(__global void* inRe, __global void* inIm, uint offset, __global void* userdata)\n \
				{ \n \
				__global USER_DATA *data = ((__global USER_DATA *)userdata + offset); \n \
				int scalar = (int)data->scalar1 + (int)data->scalar2; \n \
				float2 ret; \n \
				ret.x = *((__global float*)inRe + offset) * scalar; \n \
				ret.y = *((__global float*)inIm + offset) * scalar; \n \
				return ret; \n \
				}

#define MULVAL_PLANAR_DP double2 mulval(__global void* inRe, __global void* inIm, uint offset, __global void* userdata)\n \
				{ \n \
				__global USER_DATA *data = ((__global USER_DATA *)userdata + offset); \n \
				int scalar = (int)data->scalar1 + (int)data->scalar2; \n \
				double2 ret; \n \
				ret.x = *((__global double*)inRe + offset) * scalar; \n \
				ret.y = *((__global double*)inIm + offset) * scalar; \n \
				return ret; \n \
				}

#define STRUCT_USERDATA typedef struct USER_DATA  \
					   {  \
						int scalar1;  \
						int scalar2;  \
						} USER_DATA; 
STRUCT_USERDATA

//Compare reference and opencl output 
template < typename T1, typename T2>
bool compare(T1 *refData, std::vector< std::complex< T2 > > data,
             size_t length, const float epsilon = 1e-6f)
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

//Compare reference and opencl output
template < typename T1, typename T2 >
bool compare(T1 *refData, std::valarray< T2 > real, std::valarray< T2 > imag,
             size_t length, const float epsilon = 1e-6f)
{
    float error = 0.0f;
    T1 ref;
	T1 diff;
	float normRef = 0.0f;
	float normError = 0.0f;

	//real compare
    for(size_t i = 0; i < length; ++i)
    {
        diff[0] = refData[i][0] - real[i];
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

	//imag compare
	error = 0.0f;
    ref[1] = 0.0;

	for(size_t i = 0; i < length; ++i)
    {
        diff[1] = refData[i][1] - imag[i];
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

//Compare reference and opencl output
template < typename T1 , typename T2 >
bool compare(T1 *refData, std::valarray< T2 > real, 
             size_t length, int batchsize, const float epsilon = 1e-6f)
{
    float error = 0.0f;
    T1 ref = 0.0;
	T1 diff;
	float normRef = 0.0f;
	float normError = 0.0f;
	size_t scale = length;

	//real compare
	for (int b = 0; b < batchsize; b++)
	{	
		int idx = b * (length + 2);
		for(size_t i = idx; i < (idx + length); ++i)
		{
			diff = refData[i] - (real[i] * scale);
			error += (float)(diff * diff);
			ref += refData[i] * refData[i];
		}
	}
	if (error != 0)
	{
		normRef =::sqrtf((float) ref);
		if (::fabs((float) ref) < 1e-7f)
		{
			return false;
		}
		normError = ::sqrtf((float) error);
		error = normError / normRef;
    
		if (error > epsilon)
			return false;
	}

	return true;
}

// Compute reference output using fftw for float type
fftwf_complex* get_fftwf_output(size_t* lengths, const size_t *inStrides, const size_t *outStrides, size_t batch_size,
								size_t fftBatchSize, size_t outfftBatchSize, size_t fftVectorSizePadded, clfftLayout in_layout,
								size_t outfftVectorSizePadded, size_t fftVectorSize, clfftDim dim, clfftDirection dir)
{
	//In FFTW last dimension has the fastest changing index
	int fftwLengths[3] = {(int)lengths[2], (int)lengths[1], (int)lengths[0]};

	fftwf_plan refPlan;

	fftwf_complex *refin = (fftwf_complex*) fftw_malloc(sizeof(fftwf_complex)*fftBatchSize);
	fftwf_complex *refout = (fftwf_complex*) fftw_malloc(sizeof(fftwf_complex)*outfftBatchSize);

	refPlan = fftwf_plan_many_dft(dim, &fftwLengths[3 - dim], (int)batch_size, 
									refin, &fftwLengths[3 - dim], (int)inStrides[0], (int)fftVectorSizePadded, 
									refout, &fftwLengths[3 - dim], (int)outStrides[0], (int)outfftVectorSizePadded, 
									dir, FFTW_ESTIMATE);

	int scalar;
	for( size_t i = 0; i < fftBatchSize; i = i + inStrides[0])
	{
		switch (in_layout)
		{
		case CLFFT_COMPLEX_INTERLEAVED:
			scalar = SCALAR + (int)(i % fftVectorSize);
			break;
		case CLFFT_COMPLEX_PLANAR:
			scalar = (int)((SCALAR + (i % fftVectorSize)) + (SCALAR + (i % fftVectorSize) + 1));
			break;
		default:
			break;
		}

		refin[i][0] = (float)(1 * scalar);
		refin[i][1] = (float)(0 * scalar);
	}

	fftwf_execute(refPlan);

	fftw_free(refin);

	fftwf_destroy_plan(refPlan);

	return refout;
}

// Compute reference output using fftw for double type
fftw_complex* get_fftw_output(size_t* lengths, const size_t *inStrides, const size_t *outStrides, size_t batch_size,
								size_t fftBatchSize, size_t outfftBatchSize, size_t fftVectorSizePadded, clfftLayout in_layout,
								size_t outfftVectorSizePadded, size_t fftVectorSize, clfftDim dim, clfftDirection dir)
{
	fftw_plan refPlan;

	fftw_complex *refin = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*fftBatchSize);
	fftw_complex *refout = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*outfftBatchSize);
	
	//In FFTW last dimension has the fastest changing index
	int fftwLengths[3] = {(int)lengths[2], (int)lengths[1], (int)lengths[0]};

	refPlan = fftw_plan_many_dft(dim, &fftwLengths[3 - dim], (int)batch_size, 
									refin, &fftwLengths[3 - dim], (int)inStrides[0], (int)fftVectorSizePadded, 
									refout, &fftwLengths[3 - dim], (int)outStrides[0], (int)outfftVectorSizePadded, 
									dir, FFTW_ESTIMATE);
							
	int scalar;
	for( size_t i = 0; i < fftBatchSize; i = i + inStrides[0])
	{
		switch (in_layout)
		{
		case CLFFT_COMPLEX_INTERLEAVED:
			scalar = SCALAR + (int)(i % fftVectorSize);
			break;
		case CLFFT_COMPLEX_PLANAR:
			scalar = (int)((SCALAR + (i % fftVectorSize)) + (SCALAR + (i % fftVectorSize) + 1));
			break;
		default:
			break;
		}

		refin[i][0] = 1 * scalar;
		refin[i][1] = 0 * scalar;
	}

	fftw_execute(refPlan);

	fftw_free(refin);

	fftw_destroy_plan(refPlan);

	return refout;
}

// Compute C2R reference output using fftw for float type
float* get_fftwf_output_c2r(size_t* lengths, size_t *strides, const size_t *inStrides, const size_t *outStrides, size_t batch_size,
								size_t fftBatchSize, size_t outfftBatchSize, size_t fftVectorSizePadded, clfftLayout in_layout,
								size_t outfftVectorSizePadded, size_t outfftVectorSize, clfftDim dim, clfftDirection dir)
{
	//In FFTW last dimension has the fastest changing index
	int fftwLengths[3] = {(int)lengths[2], (int)lengths[1], (int)lengths[0]};

	fftwf_plan refPlan;

	fftwf_complex *refin = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*fftBatchSize);
	float *refout = (float*) malloc(sizeof(float)*outfftBatchSize);

	refPlan = fftwf_plan_many_dft_c2r(dim, &fftwLengths[3 - dim], (int)batch_size, 
									refin, &fftwLengths[3 - dim], (int)inStrides[0], (int)fftVectorSizePadded, 
									refout, &fftwLengths[3 - dim], (int)outStrides[0], (int)outfftVectorSizePadded,
									FFTW_ESTIMATE);

	// set zero
	for( cl_uint i = 0; i < fftBatchSize; i = i + inStrides[0] )
	{
		refin[ i ][0] = 0; refin[ i ][1] = 0;
	}

	// impulse test case
	for(size_t b = 0; b < batch_size; b++)
	{
		size_t p3 = b * strides[3];
		refin[ p3 ][0] = static_cast<float>(outfftVectorSize);
	}

	int scalar;
	for(size_t b = 0; b < batch_size; b++)
	{
		size_t p3 = b * strides[3];
	
		for( size_t i = 0; i < fftVectorSizePadded; i = i + inStrides[0])
		{
			switch (in_layout)
			{
			case CLFFT_HERMITIAN_INTERLEAVED:
				scalar = SCALAR + i;
				break;
			case CLFFT_HERMITIAN_PLANAR:
				scalar = (int)(SCALAR + i + (SCALAR + i + 1));
				break;
			default:
				break;
			}

			refin[p3 + i][0] *= (float)(scalar);
			refin[p3 + i][1] *= (float)(scalar);
		}
	}

	fftwf_execute(refPlan);

	fftw_free(refin);

	fftwf_destroy_plan(refPlan);

	return refout;
}

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

//Validate the input and output data layout
void validateDataLayout(clfftLayout in_layout, clfftLayout out_layout, clfftResultLocation place)
{
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
		{
			throw std::runtime_error( "Un-recognized data layout" );
		}
		break;
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
		{
			throw std::runtime_error( "Un-recognized data layout" );
		}
		break;
	}

	if (( place == CLFFT_INPLACE ) &&  ( in_layout != out_layout )) 
	{
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
		default:
			{
				throw std::runtime_error( "Input layout format not yet supported" );
			}
			break;
		}
	}
}

//FFT data initializations
template < typename T >
cl_int dataInitialize(size_t* lengths, clfftDim *dim, size_t batch_size,
				  const size_t *inStrides, size_t *strides, const size_t *outStrides, size_t *o_strides,
				  size_t *fftBatchSize, size_t *outfftBatchSize, size_t *fftVectorSizePadded, clfftLayout in_layout, clfftLayout out_layout,
				  size_t *outfftVectorSizePadded, size_t *fftVectorSize, size_t *outfftVectorSize,
				  clfftResultLocation place, size_t *size_of_output_buffers_in_bytes,
				  size_t *size_of_input_buffers_in_bytes, cl_mem *input_cl_mem_buffers, cl_mem *output_cl_mem_buffers,
				  cl_context *context, cl_command_queue *queue, 
				  cl_device_type deviceType, cl_int deviceId, cl_int platformId, cl_uint command_queue_flags)
{
	cl_event outEvent = NULL;
	cl_uint number_of_output_buffers = 0;
	const size_t max_dimensions = 3;
	std::vector< cl_device_id > device_id;
	
	for (unsigned u = 0; u < max_dimensions; ++u) {
		if (0 != lengths[u])
			continue;
		lengths[u] = 1;
	}

	if( lengths[ 1 ] > 1 )
	{
		*dim	= CLFFT_2D;
	}
	if( lengths[ 2 ] > 1 )
	{
		*dim	= CLFFT_3D;
	}

	strides[ 0 ] = inStrides[0];
	strides[ 1 ] = inStrides[1];
	strides[ 2 ] = inStrides[2];
	strides[ 3 ] = inStrides[3];

	o_strides[ 0 ] = outStrides[0];
	o_strides[ 1 ] = outStrides[1];
	o_strides[ 2 ] = outStrides[2];
	o_strides[ 3 ] = outStrides[3];

	*fftVectorSize = lengths[0] * lengths[1] * lengths[2];
	*fftVectorSizePadded = strides[3];
	*fftBatchSize = *fftVectorSizePadded * batch_size;

	if(place == CLFFT_INPLACE)
	{
		*outfftVectorSize = *fftVectorSize;
		*outfftVectorSizePadded = *fftVectorSizePadded;
		*outfftBatchSize = *fftBatchSize;
	}
	else
	{
		*outfftVectorSize = lengths[0] * lengths[1] * lengths[2];
		*outfftVectorSizePadded = o_strides[3];
		*outfftBatchSize = *outfftVectorSizePadded * batch_size;
	}

	// Real to complex case
	if( (in_layout == CLFFT_REAL) || (out_layout == CLFFT_REAL) )
	{
		*fftVectorSizePadded = strides[3];
		*fftBatchSize = *fftVectorSizePadded * batch_size;

		*outfftVectorSizePadded = o_strides[3];
		*outfftBatchSize = *outfftVectorSizePadded * batch_size;

		*fftVectorSize = lengths[0] * lengths[1] * lengths[2];
		*outfftVectorSize = *fftVectorSize;
	}

	switch( out_layout )
	{
	case CLFFT_COMPLEX_INTERLEAVED:
		number_of_output_buffers = 1;
		*size_of_output_buffers_in_bytes = *outfftBatchSize * sizeof( std::complex< T > );
		break;
	case CLFFT_COMPLEX_PLANAR:
		number_of_output_buffers = 2;
		*size_of_output_buffers_in_bytes = *outfftBatchSize * sizeof(T);
		break;
	case CLFFT_HERMITIAN_INTERLEAVED:
		number_of_output_buffers = 1;
		*size_of_output_buffers_in_bytes = *outfftBatchSize * sizeof( std::complex< T > );
		break;
	case CLFFT_HERMITIAN_PLANAR:
		number_of_output_buffers = 2;
		*size_of_output_buffers_in_bytes = *outfftBatchSize * sizeof(T);
		break;
	case CLFFT_REAL:
		number_of_output_buffers = 1;
		*size_of_output_buffers_in_bytes = *outfftBatchSize * sizeof(T);
		break;
	}


	// Fill the input buffers
	switch( in_layout )
	{
	case CLFFT_COMPLEX_INTERLEAVED:
		{
			//	This call creates our openCL context and sets up our devices; expected to throw on error
			*size_of_input_buffers_in_bytes = *fftBatchSize * sizeof( std::complex< T > );

			device_id = initializeCL( deviceType, deviceId, platformId, *context, false );
			createOpenCLCommandQueue( *context,
				command_queue_flags, *queue,
				device_id,
				*size_of_input_buffers_in_bytes, 1, input_cl_mem_buffers,
				*size_of_output_buffers_in_bytes, number_of_output_buffers, output_cl_mem_buffers);

			std::vector< std::complex< T > > input( *fftBatchSize );

			// set zero
			for( cl_uint i = 0; i < *fftBatchSize; ++i )
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

			OPENCL_V_THROW( clEnqueueWriteBuffer( *queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, *size_of_input_buffers_in_bytes, &input[ 0 ],
				0, NULL, &outEvent ),
				"clEnqueueWriteBuffer failed" );

		}
		break;
	case CLFFT_COMPLEX_PLANAR:
		{
			//	This call creates our openCL context and sets up our devices; expected to throw on error
			*size_of_input_buffers_in_bytes = *fftBatchSize * sizeof( T );

			device_id = initializeCL( deviceType, deviceId, platformId, *context, false );
			createOpenCLCommandQueue( *context,
				command_queue_flags, *queue,
				device_id,
				*size_of_input_buffers_in_bytes, 2, input_cl_mem_buffers,
				*size_of_output_buffers_in_bytes, number_of_output_buffers, output_cl_mem_buffers);

			std::vector< T > real( *fftBatchSize );
			std::vector< T > imag( *fftBatchSize );

			// set zero
			for( cl_uint i = 0; i < *fftBatchSize; ++i )
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

			OPENCL_V_THROW( clEnqueueWriteBuffer( *queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, *size_of_input_buffers_in_bytes, &real[ 0 ],
				0, NULL, &outEvent ),
				"clEnqueueWriteBuffer failed" );
			OPENCL_V_THROW( clEnqueueWriteBuffer( *queue, input_cl_mem_buffers[ 1 ], CL_TRUE, 0, *size_of_input_buffers_in_bytes, &imag[ 0 ],
				0, NULL, &outEvent ),
				"clEnqueueWriteBuffer failed" );
		}
		break;
	case CLFFT_HERMITIAN_INTERLEAVED:
		{
			//	This call creates our openCL context and sets up our devices; expected to throw on error
			*size_of_input_buffers_in_bytes = *fftBatchSize * sizeof( std::complex< T > );

			device_id = initializeCL( deviceType, deviceId, platformId, *context, false );
			createOpenCLCommandQueue( *context,
				command_queue_flags, *queue,
				device_id,
				*size_of_input_buffers_in_bytes, 1, input_cl_mem_buffers,
				*size_of_output_buffers_in_bytes, number_of_output_buffers, output_cl_mem_buffers);

			std::vector< std::complex< T > > input( *fftBatchSize );

			// set zero
			for( cl_uint i = 0; i < *fftBatchSize; ++i )
			{
				input[ i ] = 0;
			}

			// impulse test case
			for(size_t b = 0; b < batch_size; b++)
			{
				size_t p3 = b * strides[3];
				input[p3] = static_cast<T>(*outfftVectorSize);

			}

			OPENCL_V_THROW( clEnqueueWriteBuffer( *queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, *size_of_input_buffers_in_bytes, &input[ 0 ],
				0, NULL, &outEvent ),
				"clEnqueueWriteBuffer failed" );
		}
		break;
	case CLFFT_HERMITIAN_PLANAR:
		{
			//	This call creates our openCL context and sets up our devices; expected to throw on error
			*size_of_input_buffers_in_bytes = *fftBatchSize * sizeof( T );

			device_id = initializeCL( deviceType, deviceId, platformId, *context, false );
			createOpenCLCommandQueue( *context,
				command_queue_flags, *queue,
				device_id,
				*size_of_input_buffers_in_bytes, 2, input_cl_mem_buffers,
				*size_of_output_buffers_in_bytes, number_of_output_buffers, output_cl_mem_buffers);

			std::vector< T > real( *fftBatchSize );
			std::vector< T > imag( *fftBatchSize );

			// set zero
			for( cl_uint i = 0; i < *fftBatchSize; ++i )
			{
				real[ i ] = 0;
				imag[ i ] = 0;
			}

			// impulse test case
			for(size_t b = 0; b < batch_size; b++)
			{
				size_t p3 = b * strides[3];
				real[p3] = static_cast<T>(*outfftVectorSize);
			}

			OPENCL_V_THROW( clEnqueueWriteBuffer( *queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, *size_of_input_buffers_in_bytes, &real[ 0 ],
				0, NULL, &outEvent ),
				"clEnqueueWriteBuffer failed" );
			OPENCL_V_THROW( clEnqueueWriteBuffer( *queue, input_cl_mem_buffers[ 1 ], CL_TRUE, 0, *size_of_input_buffers_in_bytes, &imag[ 0 ],
				0, NULL, &outEvent ),
				"clEnqueueWriteBuffer failed" );
		}
		break;
	case CLFFT_REAL:
		{
			//	This call creates our openCL context and sets up our devices; expected to throw on error
			*size_of_input_buffers_in_bytes = *fftBatchSize * sizeof( T );

			device_id = initializeCL( deviceType, deviceId, platformId, *context, false );
			createOpenCLCommandQueue( *context,
				command_queue_flags, *queue,
				device_id,
				*size_of_input_buffers_in_bytes, 1, input_cl_mem_buffers,
				*size_of_output_buffers_in_bytes, number_of_output_buffers, output_cl_mem_buffers);

			std::vector< T > real( *fftBatchSize );

			// set zero
			for( cl_uint i = 0; i < *fftBatchSize; ++i )
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

			OPENCL_V_THROW( clEnqueueWriteBuffer( *queue, input_cl_mem_buffers[ 0 ], CL_TRUE, 0, *size_of_input_buffers_in_bytes, &real[ 0 ],
				0, NULL, &outEvent ),
				"clEnqueueWriteBuffer failed" );
		}
		break;
	default:
		{
			throw std::runtime_error( "Input layout format not yet supported" );
		}
		break;
	}

	return 0;
}

//Compare output with reference C/FFTW code
template < typename T >
void compareWithReference(clfftLayout in_layout, clfftLayout out_layout, size_t outfftBatchSize, clfftResultLocation place, clfftPrecision precision,
						  cl_command_queue queue, cl_mem *input_cl_mem_buffers, size_t size_of_input_buffers_in_bytes, size_t size_of_output_buffers_in_bytes,
						  cl_mem *BuffersOut, size_t* lengths, size_t * strides, const size_t *inStrides, const size_t *outStrides, size_t *o_strides,
						  size_t batch_size, size_t fftBatchSize, size_t fftVectorSizePadded, size_t outfftVectorSize,
						  size_t outfftVectorSizePadded, size_t fftVectorSize, clfftDim dim, clfftDirection dir, bool hasPrecallback)
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
			if (hasPrecallback)
			{
				switch(in_layout)
				{
				case CLFFT_HERMITIAN_INTERLEAVED:
				case CLFFT_COMPLEX_INTERLEAVED:
					{
						if (precision == CLFFT_SINGLE)
						{
							fftwf_complex *refout;

							refout = get_fftwf_output(lengths, inStrides, outStrides, batch_size, fftBatchSize, outfftBatchSize, fftVectorSizePadded,
														in_layout, outfftVectorSizePadded, fftVectorSize, dim, dir);

							if (!compare<fftwf_complex, T>(refout, output, outfftBatchSize))
								checkflag = true;

							//for( cl_uint i = 0; i < outfftBatchSize; i = i + outStrides[0])
							//{
							//	std::cout << "i " << i << " refreal " << refout[i][0] << " refimag " << refout[i][1] << " clreal " << output[i].real() << " climag " << output[i].imag() << std::endl;
							//}
							
							/*for( cl_uint i = 0; i < outfftBatchSize; i = i + outStrides[0])
							{
								std::cout << "i " << i << " refreal " << refout[i][0] << " refimag " << refout[i][1] << " clreal " << output[i].real() << " climag " << output[i].imag() << std::endl;
							}*/

							fftwf_free(refout);
						}
						else if (precision == CLFFT_DOUBLE)
						{
							fftw_complex *refout;
							
							refout = get_fftw_output(lengths, inStrides, outStrides, batch_size, fftBatchSize, outfftBatchSize, fftVectorSizePadded,
														in_layout, outfftVectorSizePadded, fftVectorSize, dim, dir);

							if (!compare<fftw_complex, T>(refout, output, outfftBatchSize))
								checkflag = true;

							/*for( cl_uint i = 0; i < outfftBatchSize; i = i + outStrides[0])
							{
								std::cout << "i " << i << " refreal " << refout[i][0] << " refimag " << refout[i][1] << " clreal " << output[i].real() << " climag " << output[i].imag() << std::endl;
							}*/
							
							fftw_free(refout);
						}
					}
					break;
				}
			}
			else
			{
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
			if (hasPrecallback)
			{
				switch(in_layout)
				{
				case CLFFT_COMPLEX_PLANAR:
					{
						if (precision == CLFFT_SINGLE)
						{
							fftwf_complex *refout;

							refout = get_fftwf_output(lengths, inStrides, outStrides, batch_size, fftBatchSize, outfftBatchSize, fftVectorSizePadded,
														in_layout, outfftVectorSizePadded, fftVectorSize, dim, dir);

							if (!compare<fftwf_complex, T>(refout, real, imag, outfftBatchSize))
								checkflag = true;

							/*for( cl_uint i = 0; i < outfftBatchSize; i = i + outStrides[0])
							{
								std::cout << "i " << i << " refreal " << refout[i][0] << " refimag " << refout[i][1] << " clreal " << real[i] << " climag " << imag[i] << std::endl;
							}*/
							
							fftwf_free(refout);
						}
						else if (precision == CLFFT_DOUBLE)
						{
							fftw_complex *refout;

							refout = get_fftw_output(lengths, inStrides, outStrides, batch_size, fftBatchSize, outfftBatchSize, fftVectorSizePadded,
														in_layout, outfftVectorSizePadded, fftVectorSize, dim, dir);

							if (!compare<fftw_complex, T>(refout, real, imag, outfftBatchSize))
								checkflag = true;

							/*for( cl_uint i = 0; i < outfftBatchSize; i = i + outStrides[0])
							{
								std::cout << "i " << i << " refreal " << refout[i][0] << " refimag " << refout[i][1] << " clreal " << real[i] << " climag " << imag[i] << std::endl;
							}*/
							
							fftw_free(refout);
						}
					}
					break;
				}
			}
			else
			{
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

			//  Check output data
			if (hasPrecallback)
			{
				if (precision == CLFFT_SINGLE)
				{
					float *refout;

					refout = get_fftwf_output_c2r(lengths, strides,  inStrides, outStrides, batch_size, fftBatchSize, outfftBatchSize, fftVectorSizePadded,
												in_layout, outfftVectorSizePadded, outfftVectorSize, dim, dir);

					if (!compare<float, T>(refout, real, outfftVectorSize, batch_size))
						checkflag = true;

					/*for( cl_uint i = 0; i < outfftBatchSize; i = i + outStrides[0])
					{
						std::cout << "i " << i << " refreal " << refout[i] << " clreal " << (real[i] * outfftVectorSize) << std::endl;
					}*/
					
					if (refout)
						free(refout);
				}
			}
			else
			{
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

template < typename T >
int transform( size_t* lengths, const size_t *inStrides, const size_t *outStrides, size_t batch_size,
				clfftLayout in_layout, clfftLayout out_layout,
				clfftResultLocation place, clfftPrecision precision, clfftDirection dir,
				cl_device_type deviceType, cl_int deviceId, cl_int platformId, bool printInfo,
				cl_uint command_queue_flags, cl_uint profile_count,
				std::auto_ptr< clfftSetupData > setupData,
				bool hasPrecallback)
{
	//	Our command line does not specify what dimension FFT we wish to transform; we decode
	//	this from the lengths that the user specifies for X, Y, Z.  A length of one means that
	//	The user does not want that dimension.

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
	
	clfftDim	dim = CLFFT_1D;
	cl_mem input_cl_mem_buffers [2] = { NULL, NULL };
	cl_mem output_cl_mem_buffers[2] = { NULL, NULL };
	cl_context context;
	cl_command_queue queue;
	cl_event outEvent = NULL;
	clfftPlanHandle plan_handle;

	//Valudate input and output data layout
	validateDataLayout(in_layout, out_layout, place);
	
	if (hasPrecallback && !(in_layout == CLFFT_COMPLEX_INTERLEAVED || in_layout == CLFFT_COMPLEX_PLANAR || in_layout == CLFFT_HERMITIAN_INTERLEAVED))
	{
		terr << _T("Pre-callback feature is currently supported only for Complex-Complex and Complex-Real Interleaved FFT " ) << std::endl;
		return 1;
	}

	//Initializations
	OPENCL_V_THROW( dataInitialize<T>(lengths, &dim, batch_size, inStrides, strides, outStrides, o_strides, &fftBatchSize, &outfftBatchSize, 
						&fftVectorSizePadded, in_layout, out_layout, &outfftVectorSizePadded, &fftVectorSize, &outfftVectorSize, place, 
						&size_of_output_buffers_in_bytes, &size_of_input_buffers_in_bytes, input_cl_mem_buffers, output_cl_mem_buffers, &context, &queue,
						deviceType, deviceId, platformId, command_queue_flags), "Data Initialization failed");

	//	Discover and load the timer module if present
	void* timerLibHandle = LoadSharedLibrary( "lib", "StatTimer", false );
	if( timerLibHandle == NULL )
	{
		terr << _T( "Could not find the external timing library; timings disabled" ) << std::endl;
	}


	//	Timer module discovered and loaded successfully
	//	Initialize function pointers to call into the shared module
	PFGETSTATTIMER get_timer = reinterpret_cast< PFGETSTATTIMER > ( LoadFunctionAddr( timerLibHandle, "getStatTimer" ) );

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

	//Check for Precallback
	//Currently test includes only for 1D
	if (hasPrecallback)
	{
		cl_mem userdata;

		//C2C 1D Interleaved 
		if (in_layout == CLFFT_COMPLEX_INTERLEAVED )
		{
			char* precallbackstr = (precision == CLFFT_SINGLE) ? STRINGIFY(MULVAL) : STRINGIFY(MULVAL_DP);

			int *h_userdata = (int*)malloc(sizeof(int)*fftBatchSize);
			for( cl_uint i = 0; i < fftBatchSize; i = i + inStrides[0])
			{
				h_userdata[ i ] = SCALAR + (i % fftVectorSize);
			}
			userdata = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * fftBatchSize, (void*)h_userdata, NULL);

			//Register the callback
			OPENCL_V_THROW (clFFTSetPlanCallback(plan_handle, "mulval", precallbackstr, NULL, 0, PRECALLBACK, userdata), "clFFTSetPlanCallback failed");
		}
		else if (in_layout == CLFFT_HERMITIAN_INTERLEAVED)
		{	
			char* precallbackstr = (precision == CLFFT_SINGLE) ? STRINGIFY(MULVAL_C2R) : STRINGIFY(MULVAL_DP);
			
			int *h_userdata = (int*)malloc(sizeof(int)*fftBatchSize);
			for(size_t b = 0; b < batch_size; b++)
			{
				size_t p3 = b * strides[3];
	
				for( size_t i = 0; i < fftVectorSizePadded; i = i + inStrides[0])
				{
					h_userdata[ p3 + i ] = SCALAR + i;
				}
			}
			userdata = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * fftBatchSize, (void*)h_userdata, NULL);

			//Register the callback
			OPENCL_V_THROW (clFFTSetPlanCallback(plan_handle, "mulval", precallbackstr, NULL, 0, PRECALLBACK, userdata), "clFFTSetPlanCallback failed");
		}

		//C2C PLANAR 
		if (in_layout == CLFFT_COMPLEX_PLANAR)
		{	
			char* precallbackstr = (precision == CLFFT_SINGLE) ? STRINGIFY(MULVAL_PLANAR) : STRINGIFY(MULVAL_PLANAR_DP);
			USER_DATA *h_userdata = (USER_DATA*)malloc(sizeof(USER_DATA) * fftBatchSize);
			for( size_t i = 0; i < fftBatchSize; i = i + inStrides[0])
			{
				h_userdata[i].scalar1 = SCALAR + (int)(i % fftVectorSize);
				h_userdata[i].scalar2 = SCALAR + (int)(i % fftVectorSize) + 1;
			}
			userdata = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(USER_DATA) * fftBatchSize, (void*)h_userdata, NULL);

			//Register the callback
			OPENCL_V_THROW (clFFTSetPlanCallback(plan_handle, "mulval", precallbackstr, STRINGIFY(STRUCT_USERDATA), 0, PRECALLBACK, userdata), "clFFTSetPlanCallback failed");
		}
	}

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


	cl_mem * BuffersOut = ( place == CLFFT_INPLACE ) ? NULL : &output_cl_mem_buffers[ 0 ];

	Timer tr;
	tr.Start();

	//	Loop as many times as the user specifies to average out the timings
	for( cl_uint i = 0; i < profile_count; ++i )
	{
		if( timer ) timer->Start( clFFTID );

		OPENCL_V_THROW( clfftEnqueueTransform( plan_handle, dir, 1, &queue, 0, NULL, &outEvent,
			&input_cl_mem_buffers[ 0 ], BuffersOut, clMedBuffer ),
			"clfftEnqueueTransform failed" );

		if( timer ) timer->Stop( clFFTID );
	}
	OPENCL_V_THROW( clFinish( queue ), "clFinish failed" );
	if(clMedBuffer) clReleaseMemObject(clMedBuffer);

	double wtime = tr.Sample()/((double)profile_count);
	size_t totalLen = 1;
	for(int i=0; i<dim; i++) totalLen *= lengths[i];
	double opsconst = 5.0 * (double)totalLen * log((double)totalLen) / log(2.0);

	if(profile_count > 1)
	{
		tout << "\nExecution wall time: " << 1000.0*wtime << " ms" << std::endl;
		tout << "Execution gflops: " << ((double)batch_size * opsconst)/(1000000000.0*wtime) << std::endl;
	}

	if( timer && (command_queue_flags & CL_QUEUE_PROFILING_ENABLE) )
	{
		//	Remove all timings that are outside of 2 stddev (keep 65% of samples); we ignore outliers to get a more consistent result
		timer->pruneOutliers( 2.0 );
		timer->Print( );
		timer->Reset( );
	}

	/*****************/
	FreeSharedLibrary( timerLibHandle );

		// Read and check output data
	// This check is not valid if the FFT is executed multiple times inplace.
	//
	if (( place == CLFFT_OUTOFPLACE )
	||  ( profile_count == 1))
	{
		compareWithReference<T>(in_layout, out_layout, outfftBatchSize, place, precision, queue, input_cl_mem_buffers, size_of_input_buffers_in_bytes, size_of_output_buffers_in_bytes,
								BuffersOut, lengths, strides, inStrides, outStrides, o_strides, batch_size, fftBatchSize, fftVectorSizePadded, outfftVectorSize, outfftVectorSizePadded, fftVectorSize,
								dim, dir, hasPrecallback);
	}

	OPENCL_V_THROW( clfftDestroyPlan( &plan_handle ), "clfftDestroyPlan failed" );
	OPENCL_V_THROW( clfftTeardown( ), "clfftTeardown failed" );

	cleanupCL( &context, &queue, countOf( input_cl_mem_buffers ), input_cl_mem_buffers, countOf( output_cl_mem_buffers ), output_cl_mem_buffers, &outEvent );
	return 0;
}

int main(int argc, char **argv)
{
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

	//callback
	bool hasPrecallback = true;

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
			( "gpu,g",         "Force selection of OpenCL GPU devices only" )
			( "cpu,c",         "Force selection of OpenCL CPU devices only" )
			( "all,a",         "Force selection of all OpenCL devices (default)" )
			( "outPlace,o",    "Out of place FFT transform (default: in place)" )
			( "double",		   "Double precision transform (default: single)" )
			( "inv",			"Backward transform (default: forward)" )
			( "dumpKernels,d", "FFT engine will dump generated OpenCL FFT kernels to disk (default: dump off)" )
			( "noprecall",		"Disable Precallback (default: precallback on)" )
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

		if( vm.count( "help" ) )
		{
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

		if( vm.count( "noprecall" ) )
		{
			hasPrecallback = false;
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
			transform<float>( lengths, iStrides, oStrides, batchSize, inLayout, outLayout, place, precision, dir, deviceType, deviceId, platformId, false, command_queue_flags, profile_count, setupData, hasPrecallback );
		else
			transform<double>( lengths, iStrides, oStrides, batchSize, inLayout, outLayout, place, precision, dir, deviceType, deviceId, platformId, false, command_queue_flags, profile_count, setupData, hasPrecallback );
	}
	catch( std::exception& e )
	{
		terr << _T( "clFFT error condition reported:" ) << std::endl << e.what() << std::endl;
		return 1;
	}
	return 0;
}