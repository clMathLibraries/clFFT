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


 // action.transpose.square.cpp provides the entry points of "baking"
 // square transpose kernels called in plan.cpp.
 // the actual kernel string generation is provided by generator.transpose.cpp

#include "stdafx.h"
#include <math.h>
#include "generator.transpose.h"
#include "action.transpose.square.h"
#include "generator.stockham.h"
#include "action.h"

FFTGeneratedTransposeSquareAction::FFTGeneratedTransposeSquareAction(clfftPlanHandle plHandle, FFTPlan * plan, cl_command_queue queue, clfftStatus & err)
    : FFTTransposeSquareAction(plHandle, plan, queue, err)
{
    if (err != CLFFT_SUCCESS)
    {
        // FFTTransposeSquareAction() failed, exit
        fprintf(stderr, "FFTTransposeSquareAction() failed!\n");
        return;
    }

    // Initialize the FFTAction::FFTKernelGenKeyParams member
    err = this->initParams();

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedTransposeSquareAction::initParams() failed!\n");
        return;
    }

    FFTRepo &fftRepo = FFTRepo::getInstance();

    err = this->generateKernel(fftRepo, queue);

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedTransposeSquareAction::generateKernel failed\n");
        return;
    }

    err = compileKernels( queue, plHandle, plan);

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedTransposeSquareAction::compileKernels failed\n");
        return;
    }

    err = CLFFT_SUCCESS;
}


bool FFTGeneratedTransposeSquareAction::buildForwardKernel()
{
    clfftLayout inputLayout = this->getSignatureData()->fft_inputLayout;
    clfftLayout outputLayout = this->getSignatureData()->fft_outputLayout;

    bool r2c_transform = (inputLayout == CLFFT_REAL);
    bool c2r_transform = (outputLayout == CLFFT_REAL);
    bool real_transform = (r2c_transform || c2r_transform);

    return (!real_transform) || r2c_transform;
}

bool FFTGeneratedTransposeSquareAction::buildBackwardKernel()
{
    clfftLayout inputLayout = this->getSignatureData()->fft_inputLayout;
    clfftLayout outputLayout = this->getSignatureData()->fft_outputLayout;

    bool r2c_transform = (inputLayout == CLFFT_REAL);
    bool c2r_transform = (outputLayout == CLFFT_REAL);
    bool real_transform = (r2c_transform || c2r_transform);

    return (!real_transform) || c2r_transform;
}


// These strings represent the names that are used as strKernel parameters
const std::string pmRealIn( "pmRealIn" );
const std::string pmImagIn( "pmImagIn" );
const std::string pmRealOut( "pmRealOut" );
const std::string pmImagOut( "pmImagOut" );
const std::string pmComplexIn( "pmComplexIn" );
const std::string pmComplexOut( "pmComplexOut" );





clfftStatus FFTGeneratedTransposeSquareAction::initParams ()
{

    this->signature.fft_precision    = this->plan->precision;
    this->signature.fft_placeness    = this->plan->placeness;
    this->signature.fft_inputLayout  = this->plan->inputLayout;
    this->signature.fft_outputLayout = this->plan->outputLayout;
    this->signature.fft_3StepTwiddle = false;

	this->signature.fft_realSpecial  = this->plan->realSpecial;

	this->signature.transOutHorizontal = this->plan->transOutHorizontal;	// using the twiddle front flag to specify horizontal write
														// we do this so as to reuse flags in FFTKernelGenKeyParams
														// and to avoid making a new one 

    ARG_CHECK( this->plan->inStride.size( ) == this->plan->outStride.size( ) );

    if( CLFFT_INPLACE == this->signature.fft_placeness )
    {
        //	If this is an in-place transform the
        //	input and output layout, dimensions and strides
        //	*MUST* be the same.
        //
        ARG_CHECK( this->signature.fft_inputLayout == this->signature.fft_outputLayout )

        for( size_t u = this->plan->inStride.size(); u-- > 0; )
        {
            ARG_CHECK( this->plan->inStride[u] == this->plan->outStride[u] );
        }
    }

	this->signature.fft_DataDim = this->plan->length.size() + 1;
	int i = 0;
	for(i = 0; i < (this->signature.fft_DataDim - 1); i++)
	{
        this->signature.fft_N[i]         = this->plan->length[i];
        this->signature.fft_inStride[i]  = this->plan->inStride[i];
        this->signature.fft_outStride[i] = this->plan->outStride[i];

	}
    this->signature.fft_inStride[i]  = this->plan->iDist;
    this->signature.fft_outStride[i] = this->plan->oDist;

    if (this->plan->large1D != 0) {
        ARG_CHECK (this->signature.fft_N[0] != 0)
        ARG_CHECK ((this->plan->large1D % this->signature.fft_N[0]) == 0)
        this->signature.fft_3StepTwiddle = true;
		ARG_CHECK ( this->plan->large1D  == (this->signature.fft_N[1] * this->signature.fft_N[0]) );
    }

    //	Query the devices in this context for their local memory sizes
    //	How we generate a kernel depends on the *minimum* LDS size for all devices.
    //
    const FFTEnvelope * pEnvelope = NULL;
    OPENCL_V( this->plan->GetEnvelope( &pEnvelope ), _T( "GetEnvelope failed" ) );
    BUG_CHECK( NULL != pEnvelope );

    // TODO:  Since I am going with a 2D workgroup size now, I need a better check than this 1D use
    // Check:  CL_DEVICE_MAX_WORK_GROUP_SIZE/CL_KERNEL_WORK_GROUP_SIZE
    // CL_DEVICE_MAX_WORK_ITEM_SIZES
    this->signature.fft_R = 1; // Dont think i'll use
    this->signature.fft_SIMD = pEnvelope->limit_WorkGroupSize; // Use devices maximum workgroup size

	//Set callback if specified
	if (this->plan->hasPreCallback)
	{
		this->signature.fft_hasPreCallback = true;
		this->signature.fft_preCallback = this->plan->preCallback;
	}
	if (this->plan->hasPostCallback)
	{
		this->signature.fft_hasPostCallback = true;
		this->signature.fft_postCallback = this->plan->postCallbackParam;
	}
	this->signature.limit_LocalMemSize = this->plan->envelope.limit_LocalMemSize;

    return CLFFT_SUCCESS;
}


static const size_t lwSize = 256;
static const size_t reShapeFactor = 2;  


//	OpenCL does not take unicode strings as input, so this routine returns only ASCII strings
//	Feed this generator the FFTPlan, and it returns the generated program as a string
clfftStatus FFTGeneratedTransposeSquareAction::generateKernel ( FFTRepo& fftRepo, const cl_command_queue commQueueFFT )
{
	//Requested local memory size by callback must not exceed the device LDS limits after factoring the LDS size required by main FFT kernel
	if ((this->signature.fft_hasPreCallback && this->signature.fft_preCallback.localMemSize > 0) || 
		(this->signature.fft_hasPostCallback && this->signature.fft_postCallback.localMemSize > 0))
	{
		assert(!(this->signature.fft_hasPreCallback && this->signature.fft_hasPostCallback));

		bool validLDSSize = false;
		size_t requestedCallbackLDS = 0;

		if (this->signature.fft_hasPreCallback && this->signature.fft_preCallback.localMemSize > 0)
			requestedCallbackLDS = this->signature.fft_preCallback.localMemSize;
		else if (this->signature.fft_hasPostCallback && this->signature.fft_postCallback.localMemSize > 0)
			requestedCallbackLDS = this->signature.fft_postCallback.localMemSize;
		
		validLDSSize = ((2 * this->plan->ElementSize() * 16 * reShapeFactor * 16 * reShapeFactor) + requestedCallbackLDS) < this->plan->envelope.limit_LocalMemSize;
	
		if(!validLDSSize)
		{
			fprintf(stderr, "Requested local memory size not available\n");
			return CLFFT_INVALID_ARG_VALUE;
		}
	}

    std::string programCode;
    OPENCL_V( clfft_transpose_generator::genTransposeKernelBatched( this->signature, programCode, lwSize, reShapeFactor ), _T( "GenerateTransposeKernel() failed!" ) );

    cl_int status = CL_SUCCESS;
    cl_device_id Device = NULL;
    status = clGetCommandQueueInfo(commQueueFFT, CL_QUEUE_DEVICE, sizeof(cl_device_id), &Device, NULL);
    OPENCL_V( status, _T( "clGetCommandQueueInfo failed" ) );

    cl_context QueueContext = NULL;
    status = clGetCommandQueueInfo(commQueueFFT, CL_QUEUE_CONTEXT, sizeof(cl_context), &QueueContext, NULL);
    OPENCL_V( status, _T( "clGetCommandQueueInfo failed" ) );


    OPENCL_V( fftRepo.setProgramCode( Transpose_SQUARE, this->getSignatureData(), programCode, Device, QueueContext ), _T( "fftRepo.setclString() failed!" ) );

    // Note:  See genFunctionPrototype( )
    if( this->signature.fft_3StepTwiddle )
    {
        OPENCL_V( fftRepo.setProgramEntryPoints( Transpose_SQUARE, this->getSignatureData(), "transpose_square_tw_fwd", "transpose_square_tw_back", Device, QueueContext ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );
    }
    else
    {
        OPENCL_V( fftRepo.setProgramEntryPoints( Transpose_SQUARE, this->getSignatureData(), "transpose_square", "transpose_square", Device, QueueContext ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );
    }

    return CLFFT_SUCCESS;
}


clfftStatus FFTGeneratedTransposeSquareAction::getWorkSizes( std::vector< size_t >& globalWS, std::vector< size_t >& localWS )
{

	size_t wg_slice;
	if (this->signature.fft_N[0] % (16 * reShapeFactor) == 0)
		wg_slice = this->signature.fft_N[0] / 16 / reShapeFactor;
	else
		wg_slice = (this->signature.fft_N[0] / (16*reShapeFactor)) + 1;

	size_t global_item_size = wg_slice*(wg_slice + 1) / 2 * 16 * 16 * this->plan->batchsize;

	for(int i = 2; i < this->signature.fft_DataDim - 1; i++)
	{
		global_item_size *= this->signature.fft_N[i];
	}

    globalWS.clear( );
	globalWS.push_back(global_item_size);

    localWS.clear( );
    localWS.push_back( lwSize );

    return CLFFT_SUCCESS;
}
