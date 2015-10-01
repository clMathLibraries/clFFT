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


// clfft.generator.Transpose.cpp : Dynamic run-time generator of openCL transpose kernels
//

// TODO: generalize the kernel to work with any size

#include "stdafx.h"

#include <math.h>
#include <iomanip>

#include "generator.transpose.square.h"
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



inline std::stringstream& clKernWrite( std::stringstream& rhs, const size_t tabIndex )
{
    rhs << std::setw( tabIndex ) << "";
    return rhs;
}



static void OffsetCalc(std::stringstream& transKernel, const FFTKernelGenKeyParams& params, bool input )
{
	const size_t *stride = input ? params.fft_inStride : params.fft_outStride;
	std::string offset = input ? "iOffset" : "oOffset";


	clKernWrite( transKernel, 3 ) << "size_t " << offset << " = 0;" << std::endl;
	clKernWrite( transKernel, 3 ) << "g_index = get_group_id(0);" << std::endl;


	for(size_t i = params.fft_DataDim - 2; i > 0 ; i--)
	{
		clKernWrite( transKernel, 3 ) << offset << " += (g_index/numGroupsY_" << i << ")*" << stride[i+1] << ";" << std::endl;
		clKernWrite( transKernel, 3 ) << "g_index = g_index % numGroupsY_" << i << ";" << std::endl;
	}

	clKernWrite( transKernel, 3 ) << std::endl;
}




// Small snippet of code that multiplies the twiddle factors into the butterfiles.  It is only emitted if the plan tells
// the generator that it wants the twiddle factors generated inside of the transpose
static clfftStatus genTwiddleMath( const FFTKernelGenKeyParams& params, std::stringstream& transKernel, const std::string& dtComplex, bool fwd )
{

	clKernWrite( transKernel, 9 ) << std::endl;

    clKernWrite( transKernel, 9 ) << dtComplex << " Wm = TW3step( (t_gx_p*32 + lidx) * (t_gy_p*32 + lidy + loop*8) );" << std::endl;
	clKernWrite( transKernel, 9 ) << dtComplex << " Wt = TW3step( (t_gy_p*32 + lidx) * (t_gx_p*32 + lidy + loop*8) );" << std::endl;
    clKernWrite( transKernel, 9 ) << dtComplex << " Tm, Tt;" << std::endl;

	if(fwd)
	{
		clKernWrite( transKernel, 9 ) << "Tm.x = ( Wm.x * tmpm.x ) - ( Wm.y * tmpm.y );" << std::endl;
		clKernWrite( transKernel, 9 ) << "Tm.y = ( Wm.y * tmpm.x ) + ( Wm.x * tmpm.y );" << std::endl;
		clKernWrite( transKernel, 9 ) << "Tt.x = ( Wt.x * tmpt.x ) - ( Wt.y * tmpt.y );" << std::endl;
		clKernWrite( transKernel, 9 ) << "Tt.y = ( Wt.y * tmpt.x ) + ( Wt.x * tmpt.y );" << std::endl;
	}
	else
	{
		clKernWrite( transKernel, 9 ) << "Tm.x =  ( Wm.x * tmpm.x ) + ( Wm.y * tmpm.y );" << std::endl;
		clKernWrite( transKernel, 9 ) << "Tm.y = -( Wm.y * tmpm.x ) + ( Wm.x * tmpm.y );" << std::endl;
		clKernWrite( transKernel, 9 ) << "Tt.x =  ( Wt.x * tmpt.x ) + ( Wt.y * tmpt.y );" << std::endl;
		clKernWrite( transKernel, 9 ) << "Tt.y = -( Wt.y * tmpt.x ) + ( Wt.x * tmpt.y );" << std::endl;
	}

    clKernWrite( transKernel, 9 ) << "tmpm.x = Tm.x;" << std::endl;
    clKernWrite( transKernel, 9 ) << "tmpm.y = Tm.y;" << std::endl;
    clKernWrite( transKernel, 9 ) << "tmpt.x = Tt.x;" << std::endl;
    clKernWrite( transKernel, 9 ) << "tmpt.y = Tt.y;" << std::endl;

	clKernWrite( transKernel, 9 ) << std::endl;

    return CLFFT_SUCCESS;
}

// These strings represent the names that are used as strKernel parameters
const std::string pmRealIn( "pmRealIn" );
const std::string pmImagIn( "pmImagIn" );
const std::string pmRealOut( "pmRealOut" );
const std::string pmImagOut( "pmImagOut" );
const std::string pmComplexIn( "pmComplexIn" );
const std::string pmComplexOut( "pmComplexOut" );

static clfftStatus genTransposePrototype( const FFTGeneratedTransposeSquareAction::Signature & params, const size_t& lwSize, const std::string& dtPlanar, const std::string& dtComplex, 
                                         const std::string &funcName, std::stringstream& transKernel, std::string& dtInput, std::string& dtOutput )
{

    // Declare and define the function
    clKernWrite( transKernel, 0 ) << "__attribute__(( reqd_work_group_size( " << lwSize << ", 1, 1 ) ))" << std::endl;
    clKernWrite( transKernel, 0 ) << "kernel void" << std::endl;

    clKernWrite( transKernel, 0 ) << funcName << "( ";

    switch( params.fft_inputLayout )
    {
    case CLFFT_COMPLEX_INTERLEAVED:
        dtInput = dtComplex;
		dtOutput = dtComplex;
        clKernWrite( transKernel, 0 ) << "global " << dtInput << "* restrict inputA";
		break;
    case CLFFT_COMPLEX_PLANAR:
        dtInput = dtPlanar;
		dtOutput = dtPlanar;
		clKernWrite(transKernel, 0) << "global " << dtInput << "* restrict inputA_R" << ", global " << dtInput << "* restrict inputA_I";
        break;
    case CLFFT_HERMITIAN_INTERLEAVED:
    case CLFFT_HERMITIAN_PLANAR:
		return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
    case CLFFT_REAL:
		dtInput = dtPlanar;
		dtOutput = dtPlanar;

		clKernWrite(transKernel, 0) << "global " << dtInput << "* restrict inputA";
		break;
    default:
        return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
    }

	if(params.fft_placeness == CLFFT_OUTOFPLACE)
	switch (params.fft_outputLayout)
	{
		case CLFFT_COMPLEX_INTERLEAVED:
			dtInput = dtComplex;
			dtOutput = dtComplex;
			clKernWrite(transKernel, 0) << ", global " << dtOutput << "* restrict outputA";
			break;
		case CLFFT_COMPLEX_PLANAR:
			dtInput = dtPlanar;
			dtOutput = dtPlanar;
			clKernWrite(transKernel, 0) << ", global " << dtOutput << "* restrict outputA_R" << ", global " << dtOutput << "* restrict outputA_I";
			break;
		case CLFFT_HERMITIAN_INTERLEAVED:
		case CLFFT_HERMITIAN_PLANAR:
			return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
		case CLFFT_REAL:
			dtInput = dtPlanar;
			dtOutput = dtPlanar;
			clKernWrite(transKernel, 0) << ", global " << dtOutput << "* restrict outputA";
			break;
		default:
			return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
	}

	if (params.fft_hasPreCallback)
	{
		if (params.fft_preCallback.localMemSize > 0)
		{
			clKernWrite( transKernel, 0 ) << ", __global void* userdata, __local void* localmem";
		}
		else
		{
			clKernWrite( transKernel, 0 ) << ", __global void* userdata";
		}
	}


    // Close the method signature
    clKernWrite( transKernel, 0 ) << " )\n{" << std::endl;
    return CLFFT_SUCCESS;
}

static clfftStatus genTransposeKernel( const FFTGeneratedTransposeSquareAction::Signature & params, std::string& strKernel, const size_t& lwSize, const size_t reShapeFactor)
{
    strKernel.reserve( 4096 );
    std::stringstream transKernel( std::stringstream::out );

    // These strings represent the various data types we read or write in the kernel, depending on how the plan
    // is configured
    std::string dtInput;        // The type read as input into kernel
    std::string dtOutput;       // The type written as output from kernel
    std::string dtPlanar;       // Fundamental type for planar arrays
    std::string dtComplex;      // Fundamental type for complex arrays

    // NOTE:  Enable only for debug
    // clKernWrite( transKernel, 0 ) << "#pragma OPENCL EXTENSION cl_amd_printf : enable\n" << std::endl;

	//if (params.fft_inputLayout != params.fft_outputLayout)
	//	return CLFFT_TRANSPOSED_NOTIMPLEMENTED;

    switch( params.fft_precision )
    {
    case CLFFT_SINGLE:
    case CLFFT_SINGLE_FAST:
        dtPlanar = "float";
        dtComplex = "float2";
        break;
    case CLFFT_DOUBLE:
    case CLFFT_DOUBLE_FAST:
        dtPlanar = "double";
        dtComplex = "double2";
		
		// Emit code that enables double precision in the kernel
        clKernWrite( transKernel, 0 ) << "#ifdef cl_khr_fp64" << std::endl; 
        clKernWrite( transKernel, 3 ) << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl;
        clKernWrite( transKernel, 0 ) << "#else" << std::endl;
        clKernWrite( transKernel, 3 ) <<  "#pragma OPENCL EXTENSION cl_amd_fp64 : enable" << std::endl;
        clKernWrite( transKernel, 0 ) << "#endif\n" << std::endl;

        break;
    default:
        return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
        break;
    }


	//	If twiddle computation has been requested, generate the lookup function
	if(params.fft_3StepTwiddle)
	{
		std::string str;
		StockhamGenerator::TwiddleTableLarge twLarge(params.fft_N[0] * params.fft_N[1]);
		if( (params.fft_precision == CLFFT_SINGLE) || (params.fft_precision == CLFFT_SINGLE_FAST) )
			twLarge.GenerateTwiddleTable<StockhamGenerator::P_SINGLE>(str);
		else
			twLarge.GenerateTwiddleTable<StockhamGenerator::P_DOUBLE>(str);
		clKernWrite( transKernel, 0 ) << str << std::endl;
		clKernWrite( transKernel, 0 ) << std::endl;
	}



    // This detects whether the input matrix is square
    bool notSquare = ( params.fft_N[ 0 ] == params.fft_N[ 1 ] ) ? false : true;

    if( notSquare && (params.fft_placeness == CLFFT_INPLACE) )
        return CLFFT_TRANSPOSED_NOTIMPLEMENTED;

	// This detects whether the input matrix is a multiple of 16*reshapefactor or not
	
	bool mult_of_16 = (params.fft_N[0] % (reShapeFactor * 16) == 0) ? true : false;
	


	for (size_t bothDir = 0; bothDir < 2; bothDir++)
	{
		bool fwd = bothDir ? false : true;

		//If pre-callback is set for the plan
		if (params.fft_hasPreCallback)
		{
			//Insert callback function code at the beginning 
			clKernWrite( transKernel, 0 ) << params.fft_preCallback.funcstring << std::endl;
			clKernWrite( transKernel, 0 ) << std::endl;
		}

		std::string funcName;
		if (params.fft_3StepTwiddle) // TODO
			funcName = fwd ? "transpose_square_tw_fwd" : "transpose_square_tw_back";
		else
			funcName = "transpose_square";


		// Generate kernel API
		genTransposePrototype(params, lwSize, dtPlanar, dtComplex, funcName, transKernel, dtInput, dtOutput);

		if (mult_of_16)
			clKernWrite(transKernel, 3) << "const int numGroupsY_1 = " << (params.fft_N[0] / 16 / reShapeFactor)*(params.fft_N[0] / 16 / reShapeFactor + 1) / 2 << ";" << std::endl;
		else
			clKernWrite(transKernel, 3) << "const int numGroupsY_1 = " << (params.fft_N[0] / (16 * reShapeFactor) + 1)*(params.fft_N[0] / (16 * reShapeFactor) + 1 + 1) / 2 << ";" << std::endl;


		for(int i = 2; i < params.fft_DataDim - 1; i++)
		{
			clKernWrite( transKernel, 3 ) << "const size_t numGroupsY_" << i << " = numGroupsY_" << i-1 << " * " << params.fft_N[i] << ";" << std::endl;
		}

		clKernWrite( transKernel, 3 ) << "size_t g_index;" << std::endl ;
		clKernWrite(transKernel, 3) << std::endl;

		OffsetCalc(transKernel, params, true);

		if(params.fft_placeness == CLFFT_OUTOFPLACE)
			OffsetCalc(transKernel, params, false);


		// Handle planar and interleaved right here
		switch (params.fft_inputLayout)
		{
			case CLFFT_COMPLEX_INTERLEAVED:
				//Do not advance offset when precallback is set as the starting address of global buffer is needed
				if (!params.fft_hasPreCallback)
				{
					clKernWrite(transKernel, 3) << "inputA += iOffset;" << std::endl;  // Set A ptr to the start of each slice
				}
				break;
			case CLFFT_COMPLEX_PLANAR:
				//Do not advance offset when precallback is set as the starting address of global buffer is needed
				if (!params.fft_hasPreCallback)
				{
					clKernWrite(transKernel, 3) << "inputA_R += iOffset;" << std::endl;  // Set A ptr to the start of each slice 
					clKernWrite(transKernel, 3) << "inputA_I += iOffset;" << std::endl;  // Set A ptr to the start of each slice 
				}				
				break;
			case CLFFT_HERMITIAN_INTERLEAVED:
			case CLFFT_HERMITIAN_PLANAR:
				return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			case CLFFT_REAL:
				break;
			default:
				return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
		}

		if(params.fft_placeness == CLFFT_OUTOFPLACE)
		{
			switch (params.fft_outputLayout)
			{
				case CLFFT_COMPLEX_INTERLEAVED:
					clKernWrite(transKernel, 3) << "outputA += oOffset;" << std::endl;  // Set A ptr to the start of each slice

					break;
				case CLFFT_COMPLEX_PLANAR:

					clKernWrite(transKernel, 3) << "outputA_R += oOffset;" << std::endl;  // Set A ptr to the start of each slice 
					clKernWrite(transKernel, 3) << "outputA_I += oOffset;" << std::endl;  // Set A ptr to the start of each slice 
					break;
				case CLFFT_HERMITIAN_INTERLEAVED:
				case CLFFT_HERMITIAN_PLANAR:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
				case CLFFT_REAL:
					break;
				default:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			}
		}
		else
		{
			switch (params.fft_inputLayout)
			{
				case CLFFT_COMPLEX_INTERLEAVED:
					if (params.fft_hasPreCallback)
					{
						clKernWrite(transKernel, 3) << "global " << dtInput << " *outputA = inputA + iOffset;" << std::endl; 
					}
					else
					{
						clKernWrite(transKernel, 3) << "global " << dtInput << " *outputA = inputA;" << std::endl; 
					}
					break;
				case CLFFT_COMPLEX_PLANAR:
					if (params.fft_hasPreCallback)
					{
						clKernWrite(transKernel, 3) << "global " << dtInput << " *outputA_R = inputA_R + iOffset;" << std::endl;
						clKernWrite(transKernel, 3) << "global " << dtInput << " *outputA_I = inputA_I + iOffset;" << std::endl;
					}
					else
					{
						clKernWrite(transKernel, 3) << "global " << dtInput << " *outputA_R = inputA_R;" << std::endl;
						clKernWrite(transKernel, 3) << "global " << dtInput << " *outputA_I = inputA_I;" << std::endl;
					}				
					break;
				case CLFFT_HERMITIAN_INTERLEAVED:
				case CLFFT_HERMITIAN_PLANAR:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
				case CLFFT_REAL:
					break;
				default:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			}
		}

		
		clKernWrite(transKernel, 3) << std::endl;

		
		// Now compute the corresponding y,x coordinates
		// for a triangular indexing
		if (mult_of_16)
			clKernWrite(transKernel, 3) << "float row = (" << -2.0f*params.fft_N[0] / 16 / reShapeFactor - 1 << "+sqrt((" << 4.0f*params.fft_N[0] / 16 / reShapeFactor*(params.fft_N[0] / 16 / reShapeFactor + 1) << "-8.0f*g_index- 7)))/ (-2.0f);" << std::endl;
		else
			clKernWrite(transKernel, 3) << "float row = (" << -2.0f*(params.fft_N[0] / (16 * reShapeFactor) + 1) - 1 << "+sqrt((" << 4.0f*(params.fft_N[0] / (16 * reShapeFactor) + 1)*(params.fft_N[0] / (16 * reShapeFactor) + 1 + 1) << "-8.0f*g_index- 7)))/ (-2.0f);" << std::endl;


		clKernWrite(transKernel, 3) << "if (row == (float)(int)row) row -= 1; " << std::endl;
		clKernWrite(transKernel, 3) << "const int t_gy = (int)row;" << std::endl;

		clKernWrite(transKernel, 3) << "" << std::endl;

		if (mult_of_16)
			clKernWrite(transKernel, 3) << "const int t_gx_p = g_index - " << (params.fft_N[0] / 16 / reShapeFactor) << "*t_gy + t_gy*(t_gy + 1) / 2;" << std::endl;
		else
			clKernWrite(transKernel, 3) << "const int t_gx_p = g_index - " << (params.fft_N[0] / (16 * reShapeFactor) + 1) << "*t_gy + t_gy*(t_gy + 1) / 2;" << std::endl;

		clKernWrite(transKernel, 3) << "const int t_gy_p = t_gx_p - t_gy;" << std::endl;


		clKernWrite(transKernel, 3) << "" << std::endl;

		clKernWrite(transKernel, 3) << "const int d_lidx = get_local_id(0) % 16;" << std::endl;
		clKernWrite(transKernel, 3) << "const int d_lidy = get_local_id(0) / 16;" << std::endl;

		clKernWrite(transKernel, 3) << "" << std::endl;

		clKernWrite(transKernel, 3) << "const int lidy = (d_lidy * 16 + d_lidx) /"<<(16 * reShapeFactor)<<";" << std::endl;
		clKernWrite(transKernel, 3) << "const int lidx = (d_lidy * 16 + d_lidx) %"<<(16 * reShapeFactor)<<";" << std::endl;

		clKernWrite(transKernel, 3) << "" << std::endl;

		clKernWrite(transKernel, 3) << "const int idx = lidx + t_gx_p*" << 16 * reShapeFactor << ";" << std::endl;
		clKernWrite(transKernel, 3) << "const int idy = lidy + t_gy_p*" << 16 * reShapeFactor << ";" << std::endl;

		clKernWrite(transKernel, 3) << "" << std::endl;

		clKernWrite(transKernel, 3) << "const int starting_index_yx = t_gy_p*" << 16 * reShapeFactor << " + t_gx_p*" << 16 * reShapeFactor*params.fft_N[0] << ";" << std::endl;

		clKernWrite(transKernel, 3) << "" << std::endl;

		clKernWrite(transKernel, 3) << "__local "<<dtComplex<<" xy_s[" << 16 * reShapeFactor * 16 * reShapeFactor << "];" << std::endl;
		clKernWrite(transKernel, 3) << "__local "<<dtComplex<<" yx_s[" << 16 * reShapeFactor * 16 * reShapeFactor << "];" << std::endl;
		
		clKernWrite(transKernel, 3) << dtComplex << " tmpm, tmpt;" << std::endl;

		clKernWrite(transKernel, 3) << "" << std::endl;

		// Step 1: Load both blocks into local memory
		// Here I load inputA for both blocks contiguously and write it contigously into
		// the corresponding shared memories.
		// Afterwards I use non-contiguous access from local memory and write contiguously
		// back into the arrays

		if (mult_of_16){
			clKernWrite(transKernel, 3) << "int index;" << std::endl;
			clKernWrite(transKernel, 3) << "for (int loop = 0; loop<" << reShapeFactor*reShapeFactor << "; ++loop){" << std::endl;
			clKernWrite(transKernel, 6) << "index = lidy*" << 16 * reShapeFactor << " + lidx + loop*256;" << std::endl;

			// Handle planar and interleaved right here
			switch (params.fft_inputLayout)
			{
				case CLFFT_COMPLEX_INTERLEAVED:
					{
						if (params.fft_hasPreCallback)
						{
							if (params.fft_preCallback.localMemSize > 0)
							{
								clKernWrite(transKernel, 6) << "tmpm = " << params.fft_preCallback.funcname << "(inputA, iOffset + (idy + loop * " << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx, userdata, localmem);" << std::endl;
								clKernWrite(transKernel, 6) << "tmpt = " << params.fft_preCallback.funcname << "(inputA, iOffset + (lidy + loop * " << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx, userdata, localmem);" << std::endl;
							}
							else
							{
								clKernWrite(transKernel, 6) << "tmpm = " << params.fft_preCallback.funcname << "(inputA, iOffset + (idy + loop * " << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx, userdata);" << std::endl;
								clKernWrite(transKernel, 6) << "tmpt = " << params.fft_preCallback.funcname << "(inputA, iOffset + (lidy + loop * " << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx, userdata);" << std::endl;
							}
						}
						else
						{
							clKernWrite(transKernel, 6) << "tmpm = inputA[(idy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];" << std::endl;
							clKernWrite(transKernel, 6) << "tmpt = inputA[(lidy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];" << std::endl;
						}
					}
					break;
				case CLFFT_COMPLEX_PLANAR:
					dtInput = dtPlanar;
					dtOutput = dtPlanar;
					if (params.fft_hasPreCallback)
					{
						if (params.fft_preCallback.localMemSize > 0)
						{
							clKernWrite(transKernel, 6) << "tmpm = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (idy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx, userdata, localmem);" << std::endl;
							clKernWrite(transKernel, 6) << "tmpt = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (lidy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx, userdata, localmem);" << std::endl;
						}
						else
						{
							clKernWrite(transKernel, 6) << "tmpm = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (idy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx, userdata);" << std::endl;
							clKernWrite(transKernel, 6) << "tmpt = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (lidy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx, userdata);" << std::endl;
						}
					}
					else
					{
						clKernWrite(transKernel, 6) << "tmpm.x = inputA_R[(idy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];" << std::endl;
						clKernWrite(transKernel, 6) << "tmpm.y = inputA_I[(idy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];" << std::endl;

						clKernWrite(transKernel, 6) << "tmpt.x = inputA_R[(lidy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];" << std::endl;
						clKernWrite(transKernel, 6) << "tmpt.y = inputA_I[(lidy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];" << std::endl;
					}
					break;
				case CLFFT_HERMITIAN_INTERLEAVED:
				case CLFFT_HERMITIAN_PLANAR:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
				case CLFFT_REAL:
					break;
				default:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			}

			// If requested, generate the Twiddle math to multiply constant values
			if( params.fft_3StepTwiddle )
				genTwiddleMath( params, transKernel, dtComplex, fwd );

			clKernWrite( transKernel, 6 ) << "xy_s[index] = tmpm; " << std::endl;
			clKernWrite( transKernel, 6 ) << "yx_s[index] = tmpt; " << std::endl;

			clKernWrite(transKernel, 3) << "}" << std::endl;

			clKernWrite(transKernel, 3) << "" << std::endl;

			clKernWrite(transKernel, 3) << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
			
			clKernWrite(transKernel, 3) << "" << std::endl;


			// Step2: Write from shared to global
			clKernWrite(transKernel, 3) << "for (int loop = 0; loop<" << reShapeFactor*reShapeFactor << "; ++loop){" << std::endl;
			clKernWrite(transKernel, 6) << "index = lidx*" << 16 * reShapeFactor << " + lidy + " << 16 / reShapeFactor << "*loop;" << std::endl;


			// Handle planar and interleaved right here
			switch (params.fft_outputLayout)
			{
				case CLFFT_COMPLEX_INTERLEAVED:
					clKernWrite(transKernel, 6) << "outputA[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx] = yx_s[index];" << std::endl;
					clKernWrite(transKernel, 6) << "outputA[(lidy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx+ starting_index_yx] = xy_s[index];" << std::endl;

					break;
				case CLFFT_COMPLEX_PLANAR:

					clKernWrite(transKernel, 6) << "outputA_R[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx] = yx_s[index].x;" << std::endl;
					clKernWrite(transKernel, 6) << "outputA_I[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx] = yx_s[index].y;" << std::endl;

					clKernWrite(transKernel, 6) << "outputA_R[(lidy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx+ starting_index_yx] = xy_s[index].x;" << std::endl;
					clKernWrite(transKernel, 6) << "outputA_I[(lidy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx+ starting_index_yx] = xy_s[index].y;" << std::endl;
					break;
				case CLFFT_HERMITIAN_INTERLEAVED:
				case CLFFT_HERMITIAN_PLANAR:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
				case CLFFT_REAL:
					break;
				default:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			}



			clKernWrite(transKernel, 3) << "}" << std::endl;

		}
		else{

			clKernWrite(transKernel, 3) << "int index;" << std::endl;
			clKernWrite(transKernel, 3) << "if ("<<params.fft_N[0]<<" - (t_gx_p + 1) *"<<16*reShapeFactor<<">0){" << std::endl;
			clKernWrite(transKernel, 6) << "for (int loop = 0; loop<" << reShapeFactor*reShapeFactor << "; ++loop){" << std::endl;
			clKernWrite(transKernel, 9) << "index = lidy*"<<16*reShapeFactor<<" + lidx + loop*256;" << std::endl;

			// Handle planar and interleaved right here
			switch (params.fft_inputLayout)
			{
				case CLFFT_COMPLEX_INTERLEAVED:
					if (params.fft_hasPreCallback)
					{
						if (params.fft_preCallback.localMemSize > 0)
						{
							clKernWrite(transKernel, 9) << "tmpm = " << params.fft_preCallback.funcname << "(inputA, iOffset + (idy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] << " + idx, userdata, localmem);" << std::endl;
							clKernWrite(transKernel, 9) << "tmpt = " << params.fft_preCallback.funcname << "(inputA, iOffset + (lidy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] <<" + lidx + starting_index_yx, userdata, localmem);" << std::endl;
						}
						else
						{
							clKernWrite(transKernel, 9) << "tmpm = " << params.fft_preCallback.funcname << "(inputA, iOffset + (idy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] << " + idx, userdata);" << std::endl;
							clKernWrite(transKernel, 9) << "tmpt = " << params.fft_preCallback.funcname << "(inputA, iOffset + (lidy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] <<" + lidx + starting_index_yx, userdata);" << std::endl;
						}
					}
					else
					{
						clKernWrite(transKernel, 9) << "tmpm = inputA[(idy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + idx];" << std::endl;
						clKernWrite(transKernel, 9) << "tmpt = inputA[(lidy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + lidx + starting_index_yx];" << std::endl;
					}
					break;
				case CLFFT_COMPLEX_PLANAR:
					dtInput = dtPlanar;
					dtOutput = dtPlanar;
					if (params.fft_hasPreCallback)
					{
						if (params.fft_preCallback.localMemSize > 0)
						{
							clKernWrite(transKernel, 9) << "tmpm = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (idy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + idx, userdata, localmem);" << std::endl;
							clKernWrite(transKernel, 9) << "tmpt = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (lidy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + lidx + starting_index_yx, userdata, localmem);" << std::endl;
						}
						else
						{
							clKernWrite(transKernel, 9) << "tmpm = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (idy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + idx, userdata);" << std::endl;
							clKernWrite(transKernel, 9) << "tmpt = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (lidy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + lidx + starting_index_yx, userdata);" << std::endl;
						}
					}
					else
					{
						clKernWrite(transKernel, 9) << "tmpm.x = inputA_R[(idy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + idx];" << std::endl;
						clKernWrite(transKernel, 9) << "tmpm.y = inputA_I[(idy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + idx];" << std::endl;

						clKernWrite(transKernel, 9) << "tmpt.x = inputA_R[(lidy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + lidx + starting_index_yx];" << std::endl;
						clKernWrite(transKernel, 9) << "tmpt.y = inputA_I[(lidy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + lidx + starting_index_yx];" << std::endl;
					}
					break;
				case CLFFT_HERMITIAN_INTERLEAVED:
				case CLFFT_HERMITIAN_PLANAR:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
				case CLFFT_REAL:
					break;
				default:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			}

			// If requested, generate the Twiddle math to multiply constant values
			if( params.fft_3StepTwiddle )
				genTwiddleMath( params, transKernel, dtComplex, fwd );

			clKernWrite(transKernel, 9) << "xy_s[index] = tmpm;" << std::endl;
			clKernWrite(transKernel, 9) << "yx_s[index] = tmpt;" << std::endl;
			clKernWrite(transKernel, 6) << "}" << std::endl;
			clKernWrite(transKernel, 3) << "}" << std::endl;

			clKernWrite(transKernel, 3) << "else{" << std::endl;
			clKernWrite(transKernel, 6) << "for (int loop = 0; loop<" << reShapeFactor*reShapeFactor << "; ++loop){" << std::endl;
			clKernWrite(transKernel, 9) << "index = lidy*" << 16 * reShapeFactor << " + lidx + loop*256;" << std::endl;


			// Handle planar and interleaved right here
			switch (params.fft_inputLayout)
			{
				case CLFFT_COMPLEX_INTERLEAVED:
					clKernWrite(transKernel, 9) << "if ((idy + loop*" << 16/reShapeFactor << ")<" << params.fft_N[0] << "&& idx<" << params.fft_N[0] << ")" << std::endl;		
					if (params.fft_hasPreCallback)
					{
						if (params.fft_preCallback.localMemSize > 0)
						{
							clKernWrite(transKernel, 12) << "tmpm = " << params.fft_preCallback.funcname << "(inputA, iOffset + (idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx, userdata, localmem);" << std::endl;
							clKernWrite(transKernel, 9) << "if ((t_gy_p *" <<16*reShapeFactor << " + lidx)<" << params.fft_N[0] << " && (t_gx_p * " << 16*reShapeFactor << " + lidy + loop*" << 16/reShapeFactor << ")<" << params.fft_N[0] << ") " << std::endl;
							clKernWrite(transKernel, 12) << "tmpt = " << params.fft_preCallback.funcname << "(inputA, iOffset + (lidy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx, userdata, localmem);" << std::endl;
						}
						else
						{
							clKernWrite(transKernel, 12) << "tmpm = " << params.fft_preCallback.funcname << "(inputA, iOffset + (idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx, userdata);" << std::endl;
							clKernWrite(transKernel, 9) << "if ((t_gy_p *" <<16*reShapeFactor << " + lidx)<" << params.fft_N[0] << " && (t_gx_p * " << 16*reShapeFactor << " + lidy + loop*" << 16/reShapeFactor << ")<" << params.fft_N[0] << ") " << std::endl;
							clKernWrite(transKernel, 12) << "tmpt = " << params.fft_preCallback.funcname << "(inputA, iOffset + (lidy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx, userdata);" << std::endl;
						}
					}
					else
					{
						clKernWrite(transKernel, 12) << "tmpm = inputA[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];" << std::endl;
						clKernWrite(transKernel, 9) << "if ((t_gy_p *" <<16*reShapeFactor << " + lidx)<" << params.fft_N[0] << " && (t_gx_p * " << 16*reShapeFactor << " + lidy + loop*" << 16/reShapeFactor << ")<" << params.fft_N[0] << ") " << std::endl;
						clKernWrite(transKernel, 12) << "tmpt = inputA[(lidy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];" << std::endl;
					}
					break;
				case CLFFT_COMPLEX_PLANAR:
					dtInput = dtPlanar;
					dtOutput = dtPlanar;
					clKernWrite(transKernel, 9) << "if ((idy + loop*" << 16/reShapeFactor << ")<" << params.fft_N[0] << "&& idx<" << params.fft_N[0] << ") {" << std::endl;			
					if (params.fft_hasPreCallback)
					{
						if (params.fft_preCallback.localMemSize > 0)
						{
							clKernWrite(transKernel, 12) << "tmpm = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx, userdata, localmem); }" << std::endl;
							clKernWrite(transKernel, 9) << "if ((t_gy_p *" <<16*reShapeFactor << " + lidx)<" << params.fft_N[0] << " && (t_gx_p * " << 16*reShapeFactor << " + lidy + loop*" << 16/reShapeFactor << ")<" << params.fft_N[0] << ") {" << std::endl;
							clKernWrite(transKernel, 12) << "tmpt = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (lidy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx, userdata, localmem); }" << std::endl;
						}
						else
						{
							clKernWrite(transKernel, 12) << "tmpm = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx, userdata); }" << std::endl;
							clKernWrite(transKernel, 9) << "if ((t_gy_p *" <<16*reShapeFactor << " + lidx)<" << params.fft_N[0] << " && (t_gx_p * " << 16*reShapeFactor << " + lidy + loop*" << 16/reShapeFactor << ")<" << params.fft_N[0] << ") {" << std::endl;
							clKernWrite(transKernel, 12) << "tmpt = " << params.fft_preCallback.funcname << "(inputA_R, inputA_I, iOffset + (lidy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx, userdata); }" << std::endl;
						}
					}
					else
					{
						clKernWrite(transKernel, 12) << "tmpm.x = inputA_R[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];" << std::endl;
						clKernWrite(transKernel, 12) << "tmpm.y = inputA_I[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx]; }" << std::endl;
						clKernWrite(transKernel, 9) << "if ((t_gy_p *" <<16*reShapeFactor << " + lidx)<" << params.fft_N[0] << " && (t_gx_p * " << 16*reShapeFactor << " + lidy + loop*" << 16/reShapeFactor << ")<" << params.fft_N[0] << ") {" << std::endl;
						clKernWrite(transKernel, 12) << "tmpt.x = inputA_R[(lidy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];" << std::endl;
						clKernWrite(transKernel, 12) << "tmpt.y = inputA_I[(lidy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx]; }" << std::endl;
					}
					break;
				case CLFFT_HERMITIAN_INTERLEAVED:
				case CLFFT_HERMITIAN_PLANAR:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
				case CLFFT_REAL:
					break;
				default:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			}


			// If requested, generate the Twiddle math to multiply constant values
			if( params.fft_3StepTwiddle )
				genTwiddleMath( params, transKernel, dtComplex, fwd );

			clKernWrite(transKernel, 9) << "xy_s[index] = tmpm;" << std::endl;
			clKernWrite(transKernel, 9) << "yx_s[index] = tmpt;" << std::endl;

			clKernWrite(transKernel, 9) << "}" << std::endl;
			clKernWrite(transKernel, 3) << "}" << std::endl;

			clKernWrite(transKernel, 3) << "" << std::endl;
			clKernWrite(transKernel, 3) << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
			clKernWrite(transKernel, 3) << "" << std::endl;

			// Step2: Write from shared to global

			clKernWrite(transKernel, 3) << "if (" << params.fft_N[0] << " - (t_gx_p + 1) *" << 16 * reShapeFactor << ">0){" << std::endl;
			clKernWrite(transKernel, 6) << "for (int loop = 0; loop<" << reShapeFactor*reShapeFactor << "; ++loop){" << std::endl;
			clKernWrite(transKernel, 9) << "index = lidx*" << 16 * reShapeFactor << " + lidy + "<<16/reShapeFactor<<"*loop ;" << std::endl;

			// Handle planar and interleaved right here
			switch (params.fft_outputLayout)
			{
				case CLFFT_COMPLEX_INTERLEAVED:
					clKernWrite(transKernel, 9) << "outputA[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx] = yx_s[index];" << std::endl;
					clKernWrite(transKernel, 9) << "outputA[(lidy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx] = xy_s[index]; " << std::endl;

					break;
				case CLFFT_COMPLEX_PLANAR:
					clKernWrite(transKernel, 9) << "outputA_R[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx] = yx_s[index].x;" << std::endl;
					clKernWrite(transKernel, 9) << "outputA_I[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx] = yx_s[index].y;" << std::endl;
					clKernWrite(transKernel, 9) << "outputA_R[(lidy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx] = xy_s[index].x; " << std::endl;
					clKernWrite(transKernel, 9) << "outputA_I[(lidy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx] = xy_s[index].y; " << std::endl;



					break;
				case CLFFT_HERMITIAN_INTERLEAVED:
				case CLFFT_HERMITIAN_PLANAR:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
				case CLFFT_REAL:
					break;
				default:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			}


			clKernWrite(transKernel, 6) << "}" << std::endl;
			clKernWrite(transKernel, 3) << "}" << std::endl;

			clKernWrite(transKernel, 3) << "else{" << std::endl;
			clKernWrite(transKernel, 6) << "for (int loop = 0; loop<" << reShapeFactor*reShapeFactor << "; ++loop){" << std::endl;
			
			clKernWrite(transKernel, 9) << "index = lidx*" << 16 * reShapeFactor << " + lidy + " << 16 / reShapeFactor << "*loop;" << std::endl;

			// Handle planar and interleaved right here
			switch (params.fft_outputLayout)
			{
				case CLFFT_COMPLEX_INTERLEAVED:
					clKernWrite(transKernel, 9) << "if ((idy + loop*" << 16 / reShapeFactor << ")<" << params.fft_N[0] << " && idx<" << params.fft_N[0] << ")" << std::endl;
					clKernWrite(transKernel, 12) << "outputA[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx] = yx_s[index]; " << std::endl;
					clKernWrite(transKernel, 9) << "if ((t_gy_p * " << 16 * reShapeFactor << " + lidx)<" << params.fft_N[0] << " && (t_gx_p * " << 16 * reShapeFactor << " + lidy + loop*" << 16 / reShapeFactor << ")<" << params.fft_N[0] << ")" << std::endl;
					clKernWrite(transKernel, 12) << "outputA[(lidy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx] = xy_s[index];" << std::endl;

					break;
				case CLFFT_COMPLEX_PLANAR:
					clKernWrite(transKernel, 9) << "if ((idy + loop*" << 16 / reShapeFactor << ")<" << params.fft_N[0] << " && idx<" << params.fft_N[0] << ") {" << std::endl;
					clKernWrite(transKernel, 12) << "outputA_R[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx] = yx_s[index].x; " << std::endl;
					clKernWrite(transKernel, 12) << "outputA_I[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx] = yx_s[index].y; }" << std::endl;
					clKernWrite(transKernel, 9) << "if ((t_gy_p * " << 16 * reShapeFactor << " + lidx)<" << params.fft_N[0] << " && (t_gx_p * " << 16 * reShapeFactor << " + lidy + loop*" << 16 / reShapeFactor << ")<" << params.fft_N[0] << ") {" << std::endl;
					clKernWrite(transKernel, 12) << "outputA_R[(lidy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx] = xy_s[index].x;" << std::endl;
					clKernWrite(transKernel, 12) << "outputA_I[(lidy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx] = xy_s[index].y; }" << std::endl;


					break;
				case CLFFT_HERMITIAN_INTERLEAVED:
				case CLFFT_HERMITIAN_PLANAR:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
				case CLFFT_REAL:
					break;
				default:
					return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			}


			clKernWrite(transKernel, 6) << "}" << std::endl; // end for
			clKernWrite(transKernel, 3) << "}" << std::endl; // end else

		
		}
		clKernWrite(transKernel, 0) << "}" << std::endl;
		
		strKernel = transKernel.str();
		
		if (!params.fft_3StepTwiddle)
			break;
	}

     return CLFFT_SUCCESS;
}


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

    return CLFFT_SUCCESS;
}


static const size_t lwSize = 256;
static const size_t reShapeFactor = 2;  


//	OpenCL does not take unicode strings as input, so this routine returns only ASCII strings
//	Feed this generator the FFTPlan, and it returns the generated program as a string
clfftStatus FFTGeneratedTransposeSquareAction::generateKernel ( FFTRepo& fftRepo, const cl_command_queue commQueueFFT )
{
	

    std::string programCode;
    OPENCL_V( genTransposeKernel( this->signature, programCode, lwSize, reShapeFactor ), _T( "GenerateTransposeKernel() failed!" ) );

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
