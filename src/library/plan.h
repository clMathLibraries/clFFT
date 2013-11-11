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
#if !defined( AMD_CLFFT_plan_H )
#define AMD_CLFFT_plan_H
#include <cstring>
#include "private.h"
#include "lock.h"
#include "generator.h"

namespace ARBITRARY {
	// TODO:  These arbitrary parameters should be tuned for the type of GPU
	//	being used.  These values are probably OK for Radeon 58xx and 68xx.
	enum {
		MAX_DIMS  = 3,
			//  The clEnqueuNDRangeKernel accepts a multi-dimensional domain array.
			//  The # of dimensions is arbitrary, but limited by the OpenCL implementation
			//  usually to 3 dimensions (CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS).
			//  The kernel generator also assumes a limit on the # of dimensions.

		SIMD_WIDTH = 64,
			//  Workgroup size.  This is the # of work items that share
			//  local data storage (LDS).  This # is best for Evergreen gpus,
			//  but might change in the future.

		LDS_BANK_BITS = 5,
		LDS_BANK_SIZE = (1 << LDS_BANK_BITS),
		LDS_PADDING   = false,//true,
			//  On AMD hardware, the low-order bits of the local_id enumerate
			//  the work items that access LDS in parallel.  Ideally, we will
			//  pad our LDS arrays so that these work items access different banks
			//  of the LDS.
			//  2 ** LDS_BANK_BITS is the number of LDS banks.
			//  If LDS_PADDING is non-zero, the kernel generator should pad the
			//  LDS arrays to reduce or eliminate bank conflicts.

		LDS_FRACTION_IDEAL = 6,    // i.e., 1/6th
		LDS_FRACTION_MAX   = 4,    // i.e., 1/4
			//  For best performance, each workgroup should use 1/IDEAL'th the amount of LDS
			//  revealed by clGetDeviceInfo (.. CL_DEVICE_LOCAL_MEM_SIZE, ...)
			//  However, we can use up to 1/MAX'th of LDS per workgroup when necessary to
			//  perform the FFT in a single pass instead of multiple passes.
			//  This tuning parameter is a good value for Evergreen gpus,
			//  but might change in the future.

		LDS_COMPLEX = false,
			//  This is the default value for FFTKernelGenKeyParams::fft_LdsComplex.
			//  The generated kernels require so many bytes of LDS for each single precision
			//..complex number in the vector.
			//  If LDS_COMPLEX, then we declare an LDS array of complex numbers (8 bytes each)
			//  and swap data between workitems with a single barrier.
			//  If ! LDS_COMPLEX, then we declare an LDS array or scalar numbers (4 bytes each)
			//  and swap data between workitems in two phases, with extra barriers.
			//  The former approach uses fewer instructions and barriers;
			//  The latter uses half as much LDS space, so twice as many wavefronts can be run
			//  in parallel.

		TWIDDLE_DEE = 4,
			//  4 bits per row of matrix.
	};
};

enum eConstantBuffer {
	/*	Layout of a constant buffer passed to the generated kernel
	 *	This needs to be know by the kernel generator and by the
	 *	framework code that creates the buffer and fills it at execution time.
	*/

	//	 [0] uint  NY   This is the batchsize for a 1D Array,
	//                    or the 2nd (Y dimension) for a 2D.
	//	 [1] uint  NZ   This is the batchsize for a 2D Array,
	//                    or the 3rd (Z dimension) for a 3D.
	//	 [2] uint  NW   This is the batchsize for a 3D Array,
	//                    or the 4th (W dimension) for a 4D.
	//	 [3] uint  N5   This is the batchsize for a 4D Array,
	//
	CLFFT_CB_NY = 0,
	CLFFT_CB_NZ,
	CLFFT_CB_NW,
	CLFFT_CB_N5,

	//	 [4] uint  ISX  Input data X stride (== 1 for row-major compact data)
	//	 [5] uint  ISY  Input data Y stride (== X for row-major compact data)
	//	 [6] uint  ISZ  Input data Z stride (== X*Y for row-major compact data)
	//	 [7] uint  ISW  Input data W stride (== X*Y*Z for row-major compact data)
	//	 [8] uint  IS5  Input data 5th stride
	//
	CLFFT_CB_ISX,
	CLFFT_CB_ISY,
	CLFFT_CB_ISZ,
	CLFFT_CB_ISW,
	CLFFT_CB_IS5,

	//	 [9] uint  OSX  Output data X stride
	//	[10] uint  OSY  Output data Y stride
	//	[11] uint  OSZ  Output data Z stride
	//	[12] uint  OSW  Output data W stride
	//	[13] uint  OS5  Output data 5th stride
	//
	CLFFT_CB_OSX,
	CLFFT_CB_OSY,
	CLFFT_CB_OSZ,
	CLFFT_CB_OSW,
	CLFFT_CB_OS5,

	CLFFT_CB_SIZE  = 32,
};

struct FFTKernelGenKeyParams {
	/*
	 *	This structure distills a subset of the fftPlan data,
	 *	including all information that is used to generate the OpenCL kernel.
	 *	This structure can be used as a key to reusing kernels that have already
	 *	been compiled.
	 */
	size_t                   fft_DataDim;       // Dimensionality of the data
	size_t                   fft_N[5];          // [0] is FFT size, e.g. 1024
	                                            // This must be <= size of LDS!
	size_t                   fft_inStride [5];  // input strides
	size_t                   fft_outStride[5];  // output strides

	clfftResultLocation   fft_placeness;
	clfftLayout           fft_inputLayout;
	clfftLayout           fft_outputLayout;
	clfftPrecision        fft_precision;
	double                   fft_fwdScale;
	double                   fft_backScale;

	size_t                   fft_SIMD;          // Assume this SIMD/workgroup size
	size_t                   fft_LDSsize;       // Limit the use of LDS to this many bytes.
	size_t                   fft_R;             // # of complex values to keep in working registers
	                                            // SIMD size * R must be <= size of LDS!
	size_t                   fft_MaxRadix;      // Limit the radix to this value.
	size_t					 fft_MaxWorkGroupSize; // Limit for work group size
	bool                     fft_LdsComplex;    // If true, store complex values in LDS memory
	                                            // If false, store scalare values in LDS.
	                                            // Generally, false will provide more efficient kernels,
	                                            // but not always.
	                                            // see FFTPlan::bLdsComplex and ARBITRARY::LDS_COMPLEX
	bool                     fft_ldsPadding;    // default padding is false
	bool                     fft_3StepTwiddle;  // This is one pass of the "3-step" algorithm;
	                                            // so extra twiddles are applied on output.
	bool                     fft_UseFMA;        // *** TODO
	bool                     fft_RCsimple;
};


//	Sorting operator for struct FFTKernelGenKeyParams, such that it can be used in a map
bool operator<( const FFTKernelGenKeyParams& lhs, const FFTKernelGenKeyParams& rhs);

//	The "envelope" is a set of limits imposed by the hardware
//	This will depend on the GPU(s) in the OpenCL context.
//	If there are multiple devices, this should be the least
//	common denominators.
//
struct FFTEnvelope {
	cl_ulong   limit_LocalMemSize;
	           //  this is the minimum of CL_DEVICE_LOCAL_MEM_SIZE
	size_t     limit_Dimensions;
	           //  this is the minimum of CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
	size_t     limit_Size[8];
	           //  these are the minimima of CL_DEVICE_MAX_WORK_ITEM_SIZES[0..n]
	size_t     limit_WorkGroupSize;
	           //  this is the minimum of CL_DEVICE_MAX_WORK_GROUP_SIZE

	// ??  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE

	FFTEnvelope ()
	:	limit_LocalMemSize (0)
	,	limit_Dimensions (0)
	,	limit_WorkGroupSize (0)
	{
		::memset (& limit_Size, 0, sizeof (limit_Size));
	}
};

class FFTRepo;

//	This class contains objects that are specific to a particular FFT transform, and the data herein is useful
//	for us to know ahead of transform time such that we can optimize for these settings
class	FFTPlan
{
	template <clfftGenerators G>
	clfftStatus GetWorkSizesPvt (std::vector<size_t> & globalws, std::vector<size_t> & localws) const;

	template <clfftGenerators G>
	clfftStatus GetKernelGenKeyPvt (FFTKernelGenKeyParams & params) const;

	template <clfftGenerators G>
	clfftStatus GenerateKernelPvt (FFTRepo& fftRepo,  const cl_command_queue commQueueFFT ) const;

	template <clfftGenerators G>
	clfftStatus GetMax1DLengthPvt (size_t *longest ) const;

public:
	bool baked;
	bool readFromFile;

	//	Properties provided by the user.
	clfftDim             dim;
	clfftLayout          inputLayout;
	clfftLayout          outputLayout;
	clfftResultLocation  placeness;
	clfftResultTransposed transposed;
	clfftPrecision       precision;
	cl_context              context;
	double                  forwardScale, backwardScale;
	size_t                  iDist, oDist;
	size_t                  batchsize;

	// Note the device passed to BakePlan, assuming we are baking for one device
	// TODO, change this logic for handling multiple GPUs/devices
	cl_device_id bakeDevice;

	//	Devices that the user specified in the context passed to the create function
	std::vector< cl_device_id > devices;

	//	Length of the FFT in each dimension
	std::vector< size_t >	length;

	//	Stride of the FFT in each dimension
	std::vector< size_t >	inStride, outStride;

	//	Hardware Limits
	FFTEnvelope                 envelope;

	//	Performance Tuning parameters
	bool                    bLdsComplex;	// see ARBITRARY::LDS_COMPLEX
	bool                    ldsPadding;     // see ARBITRARY::LDS_PADDING
	unsigned                uLdsFraction;	// see ARBITRARY::LDS_FRACTION_IDEAL

	// Reserved copy for large 1d, 2d, and 3d plan
	size_t tmpBufSize;
	cl_mem intBuffer;

	// for RC copies
	size_t	tmpBufSizeRC;
	cl_mem	intBufferRC;

	// for C-to-R transforms with largeness in Y or Z dimension
	size_t  tmpBufSizeC2R;
	cl_mem  intBufferC2R;

	//extra cache size for 2d and 3d
	size_t  cacheSize;
	size_t  large1D;
	bool    large2D;
	size_t  large1D_Xfactor;
	clfftPlanHandle planX;
	clfftPlanHandle planY;
	clfftPlanHandle planZ;

	bool transflag;
	clfftPlanHandle planTX;
	clfftPlanHandle planTY;
	clfftPlanHandle planTZ; //reserve for 3D transpose

	clfftPlanHandle planRCcopy;

	// Plan resources
	//
	cl_mem const_buffer;

	// Generator type
	clfftGenerators gen;

	// stored binaries
	size_t number_of_devices;

//TODO caching kernel binaries for later reload
#if 0
	std::unique_ptr<size_t[]> binary_sizes;
	std::vector< std::unique_ptr<char[]> > binaries;
#endif

	// Real-Complex simple flag
	// if this is set we do real to-and-from full complex using simple algorithm
	// where imaginary of input is set to zero in forward and imaginary not written in backward
	bool RCsimple;

	FFTPlan ()
	:	baked (false)
	,	readFromFile (false)
	,	dim (CLFFT_1D)
	,	inputLayout (CLFFT_COMPLEX_INTERLEAVED)
	,	outputLayout (CLFFT_COMPLEX_INTERLEAVED)
	,	placeness (CLFFT_INPLACE)
	,   transposed (CLFFT_NOTRANSPOSE)
	,	precision (CLFFT_SINGLE)
	,	context (NULL)
	,	forwardScale (1.0)
	,	backwardScale (1.0)
	,	iDist( 1 ), oDist( 1 )
	,	batchsize (1)
	,   tmpBufSize (0)
	,	intBuffer( NULL )
	,	tmpBufSizeRC (0)
	,	intBufferRC( NULL )
	,	tmpBufSizeC2R (0)
	,	intBufferC2R( NULL )
	,   large1D(0)
	,   large2D(false)
	,   planX( 0 )
	,   planY( 0 )
	,   planZ( 0 )
	,   transflag(false)
	,	RCsimple(false)
	,   planTX( 0 )
	,   planTY( 0 )
	,   planTZ( 0 )
	,	planRCcopy(0)
	,	const_buffer( NULL )
	,	bLdsComplex (ARBITRARY::LDS_COMPLEX)
	,   ldsPadding  (ARBITRARY::LDS_PADDING)
	,	uLdsFraction (0/*ARBITRARY::LDS_FRACTION_IDEAL*/)
	,   large1D_Xfactor(0)
	,   cacheSize(0)
	,	number_of_devices(0)
	,	gen(Stockham)
	{};


	size_t ElementSize() const;

	clfftStatus AllocateBuffers ();
	clfftStatus ReleaseBuffers ();

	clfftStatus GetWorkSizes (std::vector<size_t> & globalws, std::vector<size_t> & localws) const;
	clfftStatus GetKernelGenKey (FFTKernelGenKeyParams & params) const;
	clfftStatus GenerateKernel (FFTRepo & fftRepo, const cl_command_queue commQueueFFT) const;
	clfftStatus GetMax1DLength (size_t *longest ) const;

	void ResetBinarySizes();
	void ResetBinaries();

	clfftStatus CompressPlan();
	clfftStatus ConstructAndEnqueueConstantBuffers( cl_command_queue* commQueueFFT );

	clfftStatus GetEnvelope (const FFTEnvelope **) const;
	clfftStatus SetEnvelope ();

	~FFTPlan ()
	{
		ReleaseBuffers ();
	}
};

#endif // AMD_CLFFT_plan_H

