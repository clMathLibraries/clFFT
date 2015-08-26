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

#include "generator.transpose.inplace.h"
#include "generator.stockham.h"

#include "action.h"

FFTGeneratedTransposeInplaceAction::FFTGeneratedTransposeInplaceAction(clfftPlanHandle plHandle, FFTPlan * plan, cl_command_queue queue, clfftStatus & err)
    : FFTTransposeInplaceAction(plHandle, plan, queue, err)
{
    if (err != CLFFT_SUCCESS)
    {
        // FFTTransposeInplaceAction() failed, exit
        fprintf(stderr, "FFTTransposeInplaceAction() failed!\n");
        return;
    }

    // Initialize the FFTAction::FFTKernelGenKeyParams member
    err = this->initParams();

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedTransposeInplaceAction::initParams() failed!\n");
        return;
    }

    FFTRepo &fftRepo = FFTRepo::getInstance();

    err = this->generateKernel(fftRepo, queue);

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedTransposeInplaceAction::generateKernel failed\n");
        return;
    }

    err = compileKernels( queue, plHandle, plan);

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedTransposeInplaceAction::compileKernels failed\n");
        return;
    }

    err = CLFFT_SUCCESS;
}


bool FFTGeneratedTransposeInplaceAction::buildForwardKernel()
{
    clfftLayout inputLayout = this->getSignatureData()->fft_inputLayout;
    clfftLayout outputLayout = this->getSignatureData()->fft_outputLayout;

    bool r2c_transform = (inputLayout == CLFFT_REAL);
    bool c2r_transform = (outputLayout == CLFFT_REAL);
    bool real_transform = (r2c_transform || c2r_transform);

    return (!real_transform) || r2c_transform;
}

bool FFTGeneratedTransposeInplaceAction::buildBackwardKernel()
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
	clKernWrite( transKernel, 3 ) << "currDimIndex = groupIndex.y;" << std::endl;


	for(size_t i = params.fft_DataDim - 2; i > 0 ; i--)
	{
		clKernWrite( transKernel, 3 ) << offset << " += (currDimIndex/numGroupsY_" << i << ")*" << stride[i+1] << ";" << std::endl;
		clKernWrite( transKernel, 3 ) << "currDimIndex = currDimIndex % numGroupsY_" << i << ";" << std::endl;
	}

	clKernWrite( transKernel, 3 ) << "rowSizeinUnits = " << stride[1] << ";" << std::endl;

	if(params.transOutHorizontal)
	{
		if(input)
		{	
			clKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.y * wgUnroll * groupIndex.x;" << std::endl;
			clKernWrite( transKernel, 3 ) << offset << " += currDimIndex * wgTileExtent.x;" << std::endl;  
		}
		else
		{
			clKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.x * currDimIndex;" << std::endl;
			clKernWrite( transKernel, 3 ) << offset << " += groupIndex.x * wgTileExtent.y * wgUnroll;" << std::endl;
		}
	}
	else
	{
		if(input)
		{	
			clKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;" << std::endl;
			clKernWrite( transKernel, 3 ) << offset << " += groupIndex.x * wgTileExtent.x;" << std::endl;
		}
		else
		{
			clKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.x * groupIndex.x;" << std::endl;
			clKernWrite( transKernel, 3 ) << offset << " += currDimIndex * wgTileExtent.y * wgUnroll;" << std::endl;  
		}
	}

	clKernWrite( transKernel, 3 ) << std::endl;
}




// Small snippet of code that multiplies the twiddle factors into the butterfiles.  It is only emitted if the plan tells
// the generator that it wants the twiddle factors generated inside of the transpose
static clfftStatus genTwiddleMath( const FFTKernelGenKeyParams& params, std::stringstream& transKernel, const std::string& dtComplex, bool fwd )
{
    clKernWrite( transKernel, 9 ) << dtComplex << " W = TW3step( (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd) );" << std::endl;
    clKernWrite( transKernel, 9 ) << dtComplex << " T;" << std::endl;

	if(fwd)
	{
		clKernWrite( transKernel, 9 ) << "T.x = ( W.x * tmp.x ) - ( W.y * tmp.y );" << std::endl;
		clKernWrite( transKernel, 9 ) << "T.y = ( W.y * tmp.x ) + ( W.x * tmp.y );" << std::endl;
	}
	else
	{
		clKernWrite( transKernel, 9 ) << "T.x =  ( W.x * tmp.x ) + ( W.y * tmp.y );" << std::endl;
		clKernWrite( transKernel, 9 ) << "T.y = -( W.y * tmp.x ) + ( W.x * tmp.y );" << std::endl;
	}

    clKernWrite( transKernel, 9 ) << "tmp.x = T.x;" << std::endl;
    clKernWrite( transKernel, 9 ) << "tmp.y = T.y;" << std::endl;

    return CLFFT_SUCCESS;
}

// These strings represent the names that are used as strKernel parameters
const std::string pmRealIn( "pmRealIn" );
const std::string pmImagIn( "pmImagIn" );
const std::string pmRealOut( "pmRealOut" );
const std::string pmImagOut( "pmImagOut" );
const std::string pmComplexIn( "pmComplexIn" );
const std::string pmComplexOut( "pmComplexOut" );

static clfftStatus genTransposePrototype( const FFTGeneratedTransposeInplaceAction::Signature & params, const size_t& lwSize, const std::string& dtPlanar, const std::string& dtComplex, 
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




    // Close the method signature
    clKernWrite( transKernel, 0 ) << " )\n{" << std::endl;
    return CLFFT_SUCCESS;
}

static clfftStatus genTransposeKernel( const FFTGeneratedTransposeInplaceAction::Signature & params, std::string& strKernel, const size_t& lwSize, const size_t reShapeFactor)
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
	
	size_t grid_dim_0;

	if (mult_of_16)
		grid_dim_0 = params.fft_N[0] / 16 / reShapeFactor;
	else
		grid_dim_0 = params.fft_N[0] / (16 * reShapeFactor) + 1;


	for (size_t bothDir = 0; bothDir < 2; bothDir++)
	{
		bool fwd = bothDir ? false : true;

		std::string funcName;
		if (params.fft_3StepTwiddle) // TODO
			funcName = fwd ? "transpose_Inplace_tw_fwd" : "transpose_Inplace_tw_back";
		else
			funcName = "transpose_Inplace";


		// Generate kernel API
		genTransposePrototype(params, lwSize, dtPlanar, dtComplex, funcName, transKernel, dtInput, dtOutput);


		



		if (mult_of_16)
			clKernWrite(transKernel, 3) << "const int grid_dim = " << (params.fft_N[0] / 16 / reShapeFactor)*(params.fft_N[0] / 16 / reShapeFactor + 1) / 2 << ";" << std::endl;
		else
			clKernWrite(transKernel, 3) << "const int grid_dim = " << (params.fft_N[0] / (16 * reShapeFactor) + 1)*(params.fft_N[0] / (16 * reShapeFactor) + 1 + 1) / 2 << ";" << std::endl;

		clKernWrite(transKernel, 3) << "const int z = get_group_id(0) / grid_dim; " << std::endl;





		// Handle planar and interleaved right here
		switch (params.fft_inputLayout)
		{
			case CLFFT_COMPLEX_INTERLEAVED:
				clKernWrite(transKernel, 3) << "inputA = &inputA[z*" << params.fft_N[0] * params.fft_N[0] << "];" << std::endl;  // Set A ptr to the start of each slice " << std::endl;

				break;
			case CLFFT_COMPLEX_PLANAR:

				clKernWrite(transKernel, 3) << "inputA_R = &inputA_R[z*" << params.fft_N[0] * params.fft_N[0] << "];" << std::endl;  // Set A ptr to the start of each slice " << std::endl;
				clKernWrite(transKernel, 3) << "inputA_I = &inputA_I[z*" << params.fft_N[0] * params.fft_N[0] << "];" << std::endl;  // Set A ptr to the start of each slice " << std::endl;

				
				break;
			case CLFFT_HERMITIAN_INTERLEAVED:
			case CLFFT_HERMITIAN_PLANAR:
				return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			case CLFFT_REAL:
				break;
			default:
				return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
		}
		switch (params.fft_outputLayout)
		{
			case CLFFT_COMPLEX_INTERLEAVED:
				clKernWrite(transKernel, 3) << "outputA = &outputA[z*" << params.fft_N[0] * params.fft_N[0] << "];" << std::endl;  // Set A ptr to the start of each slice " << std::endl;

				break;
			case CLFFT_COMPLEX_PLANAR:

				clKernWrite(transKernel, 3) << "outputA_R = &outputA_R[z*" << params.fft_N[0] * params.fft_N[0] << "];" << std::endl;  // Set A ptr to the start of each slice " << std::endl;
				clKernWrite(transKernel, 3) << "outputA_I = &outputA_I[z*" << params.fft_N[0] * params.fft_N[0] << "];" << std::endl;  // Set A ptr to the start of each slice " << std::endl;
				break;
			case CLFFT_HERMITIAN_INTERLEAVED:
			case CLFFT_HERMITIAN_PLANAR:
				return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			case CLFFT_REAL:
				break;
			default:
				return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
		}

		
		clKernWrite(transKernel, 3) << "" << std::endl;

		clKernWrite(transKernel, 3) << "const int g_index = get_group_id(0) - z*grid_dim; " << std::endl;
		
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
					clKernWrite(transKernel, 6) << "xy_s[index] = inputA[(idy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];" << std::endl;
					clKernWrite(transKernel, 6) << "yx_s[index] = inputA[(lidy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];" << std::endl;
					break;
				case CLFFT_COMPLEX_PLANAR:
					dtInput = dtPlanar;
					dtOutput = dtPlanar;
					clKernWrite(transKernel, 6) << "xy_s[index].x = inputA_R[(idy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];" << std::endl;
					clKernWrite(transKernel, 6) << "xy_s[index].y = inputA_I[(idy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];" << std::endl;

					clKernWrite(transKernel, 6) << "yx_s[index].x = inputA_R[(lidy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];" << std::endl;
					clKernWrite(transKernel, 6) << "yx_s[index].y = inputA_I[(lidy + loop *" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];" << std::endl;

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
			clKernWrite(transKernel, 9) << "xy_s[index] = inputA[(idy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + idx];" << std::endl;
			clKernWrite(transKernel, 9) << "yx_s[index] = inputA[(lidy + loop*"<<16/reShapeFactor<<")*"<<params.fft_N[0]<<" + lidx + starting_index_yx];" << std::endl;
			clKernWrite(transKernel, 6) << "}" << std::endl;
			clKernWrite(transKernel, 3) << "}" << std::endl;

			clKernWrite(transKernel, 3) << "else{" << std::endl;
			clKernWrite(transKernel, 6) << "for (int loop = 0; loop<" << reShapeFactor*reShapeFactor << "; ++loop){" << std::endl;
			clKernWrite(transKernel, 9) << "index = lidy*" << 16 * reShapeFactor << " + lidx + loop*256;" << std::endl;
			clKernWrite(transKernel, 9) << "if ((idy + loop*" << 16/reShapeFactor << ")<" << params.fft_N[0] << "&& idx<" << params.fft_N[0] << ")" << std::endl;
			
			clKernWrite(transKernel, 12) << "xy_s[index] = inputA[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx];" << std::endl;
			clKernWrite(transKernel, 9) << "if ((t_gy_p *" <<16*reShapeFactor << " + lidx)<" << params.fft_N[0] << " && (t_gx_p * " << 16*reShapeFactor << " + lidy + loop*" << 16/reShapeFactor << ")<" << params.fft_N[0] << ") " << std::endl;
			clKernWrite(transKernel, 12) << "yx_s[index] = inputA[(lidy + loop*" << 16/reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx];" << std::endl;
			clKernWrite(transKernel, 9) << "}" << std::endl;
			clKernWrite(transKernel, 3) << "}" << std::endl;

			clKernWrite(transKernel, 3) << "" << std::endl;
			clKernWrite(transKernel, 3) << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
			clKernWrite(transKernel, 3) << "" << std::endl;

			// Step2: Write from shared to global

			clKernWrite(transKernel, 3) << "if (" << params.fft_N[0] << " - (t_gx_p + 1) *" << 16 * reShapeFactor << ">0){" << std::endl;
			clKernWrite(transKernel, 6) << "for (int loop = 0; loop<" << reShapeFactor*reShapeFactor << "; ++loop){" << std::endl;
			clKernWrite(transKernel, 9) << "index = lidx*" << 16 * reShapeFactor << " + lidy + "<<16/reShapeFactor<<"*loop ;" << std::endl;
			clKernWrite(transKernel, 9) << "outputA[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx] = yx_s[index];" << std::endl;
			clKernWrite(transKernel, 9) << "outputA[(lidy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx] = xy_s[index]; " << std::endl;


			clKernWrite(transKernel, 6) << "}" << std::endl;
			clKernWrite(transKernel, 3) << "}" << std::endl;

			clKernWrite(transKernel, 3) << "else{" << std::endl;
			clKernWrite(transKernel, 6) << "for (int loop = 0; loop<" << reShapeFactor*reShapeFactor << "; ++loop){" << std::endl;
			
			clKernWrite(transKernel, 9) << "index = lidx*" << 16 * reShapeFactor << " + lidy + " << 16 / reShapeFactor << "*loop;" << std::endl;
			clKernWrite(transKernel, 9) << "if ((idy + loop*" << 16 / reShapeFactor << ")<" << params.fft_N[0] << " && idx<" << params.fft_N[0] << ")" << std::endl;
			clKernWrite(transKernel, 12) << "outputA[(idy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + idx] = yx_s[index]; " << std::endl;
			clKernWrite(transKernel, 9) << "if ((t_gy_p * " << 16 * reShapeFactor << " + lidx)<" << params.fft_N[0] << " && (t_gx_p * " << 16 * reShapeFactor << " + lidy + loop*" << 16 / reShapeFactor << ")<" << params.fft_N[0] << ")" << std::endl;
			clKernWrite(transKernel, 12) << "outputA[(lidy + loop*" << 16 / reShapeFactor << ")*" << params.fft_N[0] << " + lidx + starting_index_yx] = xy_s[index];" << std::endl;

			clKernWrite(transKernel, 6) << "}" << std::endl; // end for
			clKernWrite(transKernel, 3) << "}" << std::endl; // end else

		
		}
		clKernWrite(transKernel, 0) << "}" << std::endl;
		
		strKernel = transKernel.str();
		//std::cout << strKernel;
		
		if (!params.fft_3StepTwiddle)
			break;
	}

    return CLFFT_SUCCESS;
}


clfftStatus FFTGeneratedTransposeInplaceAction::initParams ()
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

    return CLFFT_SUCCESS;
}

// Constants that specify the bounding sizes of the block that each workgroup will transpose
const size_t lwSize = 256;
const size_t reShapeFactor = 2;   // wgTileSize = { lwSize.x * reShapeFactor, lwSize.y / reShapeFactor }



//	OpenCL does not take unicode strings as input, so this routine returns only ASCII strings
//	Feed this generator the FFTPlan, and it returns the generated program as a string
clfftStatus FFTGeneratedTransposeInplaceAction::generateKernel ( FFTRepo& fftRepo, const cl_command_queue commQueueFFT )
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


    OPENCL_V( fftRepo.setProgramCode( Transpose_INPLACE, this->getSignatureData(), programCode, Device, QueueContext ), _T( "fftRepo.setclString() failed!" ) );

    // Note:  See genFunctionPrototype( )
    if( this->signature.fft_3StepTwiddle )
    {
        OPENCL_V( fftRepo.setProgramEntryPoints( Transpose_INPLACE, this->getSignatureData(), "transpose_Inplace_tw_fwd", "transpose_Inplace_tw_back", Device, QueueContext ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );
    }
    else
    {
        OPENCL_V( fftRepo.setProgramEntryPoints( Transpose_INPLACE, this->getSignatureData(), "transpose_Inplace", "transpose_Inplace", Device, QueueContext ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );
    }

    return CLFFT_SUCCESS;
}


clfftStatus FFTGeneratedTransposeInplaceAction::getWorkSizes( std::vector< size_t >& globalWS, std::vector< size_t >& localWS )
{

#if 0
    // We need to make sure that the global work size is evenly divisible by the local work size
    // Our transpose works in tiles, so divide tiles in each dimension to get count of blocks, rounding up for remainder items
	size_t numBlocksX = this->signature.transOutHorizontal ?
							DivRoundingUp(this->signature.fft_N[ 1 ], blockSize.y ) :
							DivRoundingUp(this->signature.fft_N[ 0 ], blockSize.x );
    size_t numBlocksY = this->signature.transOutHorizontal ?
							DivRoundingUp( this->signature.fft_N[ 0 ], blockSize.x ) :
							DivRoundingUp( this->signature.fft_N[ 1 ], blockSize.y );
    size_t numWIX = numBlocksX * lwSize.x;

    // Batches of matrices are lined up along the Y axis, 1 after the other
	size_t numWIY = numBlocksY * lwSize.y * this->plan->batchsize;
	// fft_DataDim has one more dimension than the actual fft data, which is devoted to batch.
	// dim from 2 to fft_DataDim - 2 are lined up along the Y axis
	for(int i = 2; i < this->signature.fft_DataDim - 1; i++)
	{
		numWIY *= this->signature.fft_N[i];
	}
#endif

	size_t wg_slice;
	if (this->signature.fft_N[0] % (16 * reShapeFactor) == 0)
		wg_slice = this->signature.fft_N[0] / reShapeFactor / 16;
	else
		wg_slice = ((this->signature.fft_N[0] / (16 * reShapeFactor) + 1) * 16 * reShapeFactor) / 16 / reShapeFactor;

	size_t global_item_size = wg_slice*(wg_slice + 1) / 2 * 16 * 16 * this->plan->batchsize;

    globalWS.clear( );
	globalWS.push_back(global_item_size);

    localWS.clear( );
    localWS.push_back( lwSize );

    return CLFFT_SUCCESS;
}
