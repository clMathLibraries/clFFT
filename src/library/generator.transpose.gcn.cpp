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

#include "generator.transpose.gcn.h"
#include "generator.stockham.h"

#include "action.h"

FFTGeneratedTransposeGCNAction::FFTGeneratedTransposeGCNAction(clfftPlanHandle plHandle, FFTPlan * plan, cl_command_queue queue, clfftStatus & err)
    : FFTTransposeGCNAction(plHandle, plan, queue, err)
{
    if (err != CLFFT_SUCCESS)
    {
        // FFTTransposeGCNAction() failed, exit
        fprintf(stderr, "FFTTransposeGCNAction() failed!\n");
        return;
    }

    // Initialize the FFTAction::FFTKernelGenKeyParams member
    err = this->initParams();

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedTransposeGCNAction::initParams() failed!\n");
        return;
    }

    FFTRepo &fftRepo = FFTRepo::getInstance();

    err = this->generateKernel(fftRepo, queue);

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedTransposeGCNAction::generateKernel failed\n");
        return;
    }

    err = compileKernels( queue, plHandle, plan);

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedTransposeGCNAction::compileKernels failed\n");
        return;
    }

    err = CLFFT_SUCCESS;
}


bool FFTGeneratedTransposeGCNAction::buildForwardKernel()
{
    clfftLayout inputLayout = this->getSignatureData()->fft_inputLayout;
    clfftLayout outputLayout = this->getSignatureData()->fft_outputLayout;

    bool r2c_transform = (inputLayout == CLFFT_REAL);
    bool c2r_transform = (outputLayout == CLFFT_REAL);
    bool real_transform = (r2c_transform || c2r_transform);

    return (!real_transform) || r2c_transform;
}

bool FFTGeneratedTransposeGCNAction::buildBackwardKernel()
{
    clfftLayout inputLayout = this->getSignatureData()->fft_inputLayout;
    clfftLayout outputLayout = this->getSignatureData()->fft_outputLayout;

    bool r2c_transform = (inputLayout == CLFFT_REAL);
    bool c2r_transform = (outputLayout == CLFFT_REAL);
    bool real_transform = (r2c_transform || c2r_transform);

    return (!real_transform) || c2r_transform;
}


// A structure that represents a bounding box or tile, with convenient names for the row and column addresses
// local work sizes
struct tile
{
    union
    {
        size_t x;
        size_t col;
    };

    union
    {
        size_t y;
        size_t row;
    };
};

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

static clfftStatus genTransposePrototype( const FFTGeneratedTransposeGCNAction::Signature & params, const tile& lwSize, const std::string& dtPlanar, const std::string& dtComplex, 
                                         const std::string &funcName, std::stringstream& transKernel, std::string& dtInput, std::string& dtOutput )
{

    // Declare and define the function
    clKernWrite( transKernel, 0 ) << "__attribute__(( reqd_work_group_size( " << lwSize.x << ", " << lwSize.y << ", 1 ) ))" << std::endl;
    clKernWrite( transKernel, 0 ) << "kernel void" << std::endl;

    clKernWrite( transKernel, 0 ) << funcName << "( ";

    switch( params.fft_inputLayout )
    {
    case CLFFT_COMPLEX_INTERLEAVED:
        dtInput = dtComplex;
        clKernWrite( transKernel, 0 ) << "global " << dtInput << "* restrict " << pmComplexIn;

        switch( params.fft_placeness )
        {
        case CLFFT_INPLACE:
            dtOutput = dtComplex;
            break;
        case CLFFT_OUTOFPLACE:
            switch( params.fft_outputLayout )
            {
            case CLFFT_COMPLEX_INTERLEAVED:
                dtOutput = dtComplex;
                clKernWrite( transKernel, 0 ) << ", global " << dtOutput << "* restrict " << pmComplexOut;
                break;
            case CLFFT_COMPLEX_PLANAR:
                dtOutput = dtPlanar;
                clKernWrite( transKernel, 0 ) << ", global " << dtOutput << "* restrict " << pmRealOut
                    << ", global " << dtOutput << "* restrict " << pmImagOut;
                break;
            case CLFFT_HERMITIAN_INTERLEAVED:
            case CLFFT_HERMITIAN_PLANAR:
            case CLFFT_REAL:
            default:
                return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
            }
            break;
        default:
            return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
        }
        break;
    case CLFFT_COMPLEX_PLANAR:
        dtInput = dtPlanar;
        clKernWrite( transKernel, 0 ) << "global " << dtInput << "* restrict " << pmRealIn << ", global " << dtInput << "* restrict " << pmImagIn;

        switch( params.fft_placeness )
        {
        case CLFFT_INPLACE:
            dtOutput = dtPlanar;
            break;
        case CLFFT_OUTOFPLACE:
            switch( params.fft_outputLayout )
            {
            case CLFFT_COMPLEX_INTERLEAVED:
                dtOutput = dtComplex;
                clKernWrite( transKernel, 0 ) << ", global " << dtOutput << "* restrict " << pmComplexOut;
                break;
            case CLFFT_COMPLEX_PLANAR:
                dtOutput = dtPlanar;
                clKernWrite( transKernel, 0 ) << ", global " << dtOutput << "* restrict " << pmRealOut
                    << ", global " << dtOutput << "* restrict " << pmImagOut;
                break;
            case CLFFT_HERMITIAN_INTERLEAVED:
            case CLFFT_HERMITIAN_PLANAR:
            case CLFFT_REAL:
            default:
                return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
            }
            break;
        default:
            return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
        }
        break;
    case CLFFT_HERMITIAN_INTERLEAVED:
    case CLFFT_HERMITIAN_PLANAR:
		return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
    case CLFFT_REAL:
		dtInput = dtPlanar;
		clKernWrite( transKernel, 0 ) << "global " << dtInput << "* restrict " << pmRealIn;

		switch( params.fft_placeness )
        {
        case CLFFT_INPLACE:
            dtOutput = dtPlanar;
            break;
        case CLFFT_OUTOFPLACE:
            switch( params.fft_outputLayout )
            {
            case CLFFT_COMPLEX_INTERLEAVED:
            case CLFFT_COMPLEX_PLANAR:
            case CLFFT_HERMITIAN_INTERLEAVED:
            case CLFFT_HERMITIAN_PLANAR:
				return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
            case CLFFT_REAL:
                dtOutput = dtPlanar;
                clKernWrite( transKernel, 0 ) << ", global " << dtOutput << "* restrict " << pmRealOut;
				break;
            default:
                return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
            }
            break;
        default:
            return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
        }
		break;
    default:
        return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
    }

    // Close the method signature
    clKernWrite( transKernel, 0 ) << " )\n{" << std::endl;

    return CLFFT_SUCCESS;
}

static clfftStatus genTransposeKernel( const FFTGeneratedTransposeGCNAction::Signature & params, std::string& strKernel, const tile& lwSize, const size_t reShapeFactor, 
                                            const size_t loopCount, const tile& blockSize, const size_t outRowPadding )
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


    clKernWrite( transKernel, 0 ) << "// Local structure to embody/capture tile dimensions" << std::endl;
    clKernWrite( transKernel, 0 ) << "typedef struct tag_Tile" << std::endl;
    clKernWrite( transKernel, 0 ) << "{" << std::endl;
    clKernWrite( transKernel, 3 ) << "size_t x;" << std::endl;
    clKernWrite( transKernel, 3 ) << "size_t y;" << std::endl;
    clKernWrite( transKernel, 0 ) << "} Tile;" << std::endl << std::endl;

    // This detects whether the input matrix is square
    bool notSquare = ( params.fft_N[ 0 ] == params.fft_N[ 1 ] ) ? false : true;

    if( notSquare && (params.fft_placeness == CLFFT_INPLACE) )
        return CLFFT_TRANSPOSED_NOTIMPLEMENTED;


	for(size_t bothDir=0; bothDir<2; bothDir++)
	{
		//	Generate the kernel entry point and parameter list
		//
		bool fwd = bothDir ? false : true;

		std::string funcName;
		if(params.fft_3StepTwiddle)
			funcName = fwd ? "transpose_gcn_tw_fwd" : "transpose_gcn_tw_back";
		else
			funcName = "transpose_gcn";

		genTransposePrototype( params, lwSize, dtPlanar, dtComplex, funcName, transKernel, dtInput, dtOutput );

		clKernWrite( transKernel, 3 ) << "const Tile localIndex = { get_local_id( 0 ), get_local_id( 1 ) }; " << std::endl;
		clKernWrite( transKernel, 3 ) << "const Tile localExtent = { get_local_size( 0 ), get_local_size( 1 ) }; " << std::endl;
		clKernWrite( transKernel, 3 ) << "const Tile groupIndex = { get_group_id( 0 ), get_group_id( 1 ) };" << std::endl;
		clKernWrite( transKernel, 3 ) << std::endl;



		clKernWrite( transKernel, 3 ) << "// Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block" << std::endl;
		clKernWrite( transKernel, 3 ) << "// Transpose of input & output blocks happens with the Offset calculation" << std::endl;
		clKernWrite( transKernel, 3 ) << "const size_t reShapeFactor = " << reShapeFactor << ";" << std::endl;
		clKernWrite( transKernel, 3 ) << "const size_t wgUnroll = " << loopCount << ";" << std::endl;
		clKernWrite( transKernel, 3 ) << "const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };" << std::endl;
		clKernWrite( transKernel, 3 ) << "const size_t tileSizeinUnits = wgTileExtent.x * wgTileExtent.y * wgUnroll;" << std::endl << std::endl;


		// This is the size of a matrix in the y dimension in units of group size; used to calculate stride[2] indexing
		//size_t numGroupsY = DivRoundingUp( params.fft_N[ 1 ], lwSize.y / reShapeFactor * loopCount );

		//numGroupY_1 is the number of cumulative work groups up to 1st dimension
		//numGroupY_2 is the number of cumulative work groups up to 2nd dimension and so forth

		size_t numGroupsTemp;
		if(params.transOutHorizontal)
			numGroupsTemp = DivRoundingUp( params.fft_N[0], blockSize.x );
		else
			numGroupsTemp = DivRoundingUp( params.fft_N[1], blockSize.y );

		clKernWrite( transKernel, 3 ) << "const size_t numGroupsY_1" << " = " << numGroupsTemp << ";" << std::endl;
		for(int i = 2; i < params.fft_DataDim - 1; i++)
		{
			numGroupsTemp *= params.fft_N[i];
			clKernWrite( transKernel, 3 ) << "const size_t numGroupsY_" << i << " = " << numGroupsTemp << ";" << std::endl;
		}


		// Generate the amount of local data share we need
		// Assumption: Even for planar data, we will still store values in LDS as interleaved
		tile ldsSize = { blockSize.x, blockSize.y };
		switch( params.fft_outputLayout )
		{
		case CLFFT_COMPLEX_INTERLEAVED:
		case CLFFT_COMPLEX_PLANAR:
			clKernWrite( transKernel, 3 ) << "// LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];" << std::endl;
			clKernWrite( transKernel, 3 ) << "local " << dtComplex << " lds[ " << ldsSize.x << " ][ " << ldsSize.y << " ];" << std::endl << std::endl;
			break;
		case CLFFT_HERMITIAN_INTERLEAVED:
		case CLFFT_HERMITIAN_PLANAR:
			return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
		case CLFFT_REAL:
			clKernWrite( transKernel, 3 ) << "local " << dtPlanar << " lds[ " << ldsSize.x << " ][ " << ldsSize.y << " ];" << std::endl << std::endl;
			break;
		}


		clKernWrite( transKernel, 3 ) << "size_t currDimIndex;" << std::endl ;
		clKernWrite( transKernel, 3 ) << "size_t rowSizeinUnits;" << std::endl << std::endl ;


		OffsetCalc(transKernel, params, true);


		switch( params.fft_inputLayout )
		{
		case CLFFT_COMPLEX_INTERLEAVED:
			clKernWrite( transKernel, 3 ) << "global " << dtInput << "* tileIn = " << pmComplexIn << " + iOffset;" << std::endl;
			break;
		case CLFFT_COMPLEX_PLANAR:
			clKernWrite( transKernel, 3 ) << "global " << dtInput << "* realTileIn = " << pmRealIn << " + iOffset;" << std::endl;
			clKernWrite( transKernel, 3 ) << "global " << dtInput << "* imagTileIn = " << pmImagIn << " + iOffset;" << std::endl;
			break;
		case CLFFT_HERMITIAN_INTERLEAVED:
		case CLFFT_HERMITIAN_PLANAR:
			return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
		case CLFFT_REAL:
			clKernWrite( transKernel, 3 ) << "global " << dtInput << "* tileIn = " << pmRealIn << " + iOffset;" << std::endl;
			break;
			
		}

		// This is the loop reading through the Tile
		if( params.fft_inputLayout == CLFFT_REAL )
		{
			clKernWrite( transKernel, 3 ) << dtPlanar << " tmp;" << std::endl;
		}
		else
		{
			clKernWrite( transKernel, 3 ) << dtComplex << " tmp;" << std::endl;
		}

		clKernWrite( transKernel, 3 ) << "rowSizeinUnits = " << params.fft_inStride[ 1 ] << ";" << std::endl; 
		clKernWrite( transKernel, 3 ) << std::endl << std::endl;

		//
		// Group index traversal is logical where X direction is horizontal in input buffer and vertical in output buffer
		// when transOutHorizontal is enabled X direction is vertical in input buffer and horizontal in output buffer
		// Not to be confused within a tile, where X is horizontal in input and vertical in output always
		


		bool branchingInGroupX = params.transOutHorizontal ? ((params.fft_N[1] % blockSize.y) != 0) : ((params.fft_N[0] % blockSize.x) != 0);
		bool branchingInGroupY = params.transOutHorizontal ? ((params.fft_N[0] % blockSize.x) != 0) : ((params.fft_N[1] % blockSize.y) != 0);
		bool branchingInBoth = branchingInGroupX && branchingInGroupY;
		bool branchingInAny = branchingInGroupX || branchingInGroupY;

		size_t branchBlocks = branchingInBoth ? 4 : ( branchingInAny ? 2 : 1 );

		size_t cornerGroupX = params.transOutHorizontal ? (params.fft_N[1] / blockSize.y) : (params.fft_N[0] / blockSize.x);
		size_t cornerGroupY = params.transOutHorizontal ? (params.fft_N[0] / blockSize.x) : (params.fft_N[1] / blockSize.y);

		std::string gIndexX = "groupIndex.x"; //params.transOutHorizontal ? "currDimIndex" : "groupIndex.x";
		std::string gIndexY = "currDimIndex"; //params.transOutHorizontal ? "groupIndex.x" : "currDimIndex";		
		
		std::string wIndexX = params.transOutHorizontal ? "yInd" : "xInd";
		std::string wIndexY = params.transOutHorizontal ? "xInd" : "yInd";
				
		size_t wIndexXEnd = params.transOutHorizontal ? params.fft_N[1] % blockSize.y : params.fft_N[0] % blockSize.x;
		size_t wIndexYEnd = params.transOutHorizontal ? params.fft_N[0] % blockSize.x : params.fft_N[1] % blockSize.y;


		for(size_t i = 0; i<branchBlocks; i++)
		{
			if(branchingInBoth)
				if(i == 0)
				{
					clKernWrite( transKernel, 3 ) << "if( (" << gIndexX << " == " << 
						cornerGroupX << ") && (" << gIndexY << " == " <<
						cornerGroupY << ") )" << std::endl;
					clKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else if(i == 1)
				{
					clKernWrite( transKernel, 3 ) << "else if( " << gIndexX << " == " << 
						cornerGroupX << " )" << std::endl;
					clKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else if(i == 2)
				{
					clKernWrite( transKernel, 3 ) << "else if( " << gIndexY << " == " <<
						cornerGroupY << " )" << std::endl;
					clKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else
				{
					clKernWrite( transKernel, 3 ) << "else" << std::endl;
					clKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
			else if(branchingInAny)
				if(i == 0)
				{
					if(branchingInGroupX)
					{
						clKernWrite( transKernel, 3 ) << "if( " << gIndexX << " == " << 
							cornerGroupX << " )" << std::endl;
						clKernWrite( transKernel, 3 ) << "{" << std::endl;
					}
					else
					{
						clKernWrite( transKernel, 3 ) << "if( " << gIndexY << " == " <<
							cornerGroupY << " )" << std::endl;
						clKernWrite( transKernel, 3 ) << "{" << std::endl;
					}
				}
				else
				{
					clKernWrite( transKernel, 3 ) << "else" << std::endl;
					clKernWrite( transKernel, 3 ) << "{" << std::endl;
				}



			clKernWrite( transKernel, 6 ) << "for( uint t=0; t < wgUnroll; t++ )" << std::endl;
			clKernWrite( transKernel, 6 ) << "{" << std::endl;

			clKernWrite( transKernel, 9 ) << "size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); " << std::endl;
			clKernWrite( transKernel, 9 ) << "size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; " << std::endl;

			// Calculating the index seperately enables easier debugging through tools
			clKernWrite( transKernel, 9 ) << "size_t gInd = xInd + rowSizeinUnits * yInd;" << std::endl;


			if(branchingInBoth)
			{
				if(i == 0)
				{
					clKernWrite( transKernel, 9 ) << std::endl;
					clKernWrite( transKernel, 9 ) << "if( (" << wIndexX << "< " << wIndexXEnd << ") && (" << wIndexY << " < " << wIndexYEnd << ") )" << std::endl;
					clKernWrite( transKernel, 9 ) << "{" << std::endl;
				}
				else if(i == 1)
				{
					clKernWrite( transKernel, 9 ) << std::endl;
					clKernWrite( transKernel, 9 ) << "if( (" << wIndexX << " < " << wIndexXEnd << ") )" << std::endl;
					clKernWrite( transKernel, 9 ) << "{" << std::endl;
				}
				else if(i == 2)
				{
					clKernWrite( transKernel, 9 ) << std::endl;
					clKernWrite( transKernel, 9 ) << "if( (" << wIndexY << " < " << wIndexYEnd << ") )" << std::endl;
					clKernWrite( transKernel, 9 ) << "{" << std::endl;
				}
				else
					clKernWrite( transKernel, 9 ) << "{" << std::endl;
			}
			else if(branchingInAny)
			{
				if(i == 0)
				{
					if(branchingInGroupX)
					{
						clKernWrite( transKernel, 9 ) << std::endl;
						clKernWrite( transKernel, 9 ) << "if( (" << wIndexX << " < " << wIndexXEnd << ") )" << std::endl;
						clKernWrite( transKernel, 9 ) << "{" << std::endl;
					}
					else
					{
						clKernWrite( transKernel, 9 ) << std::endl;
						clKernWrite( transKernel, 9 ) << "if( (" << wIndexY << " < " << wIndexYEnd << ") )" << std::endl;
						clKernWrite( transKernel, 9 ) << "{" << std::endl;
					}
				}
				else
					clKernWrite( transKernel, 9 ) << "{" << std::endl;
			}

			switch( params.fft_inputLayout )
			{
			case CLFFT_COMPLEX_INTERLEAVED:
				clKernWrite( transKernel, 9 ) << "tmp = tileIn[ gInd ];" << std::endl;
				break;
			case CLFFT_COMPLEX_PLANAR:
				clKernWrite( transKernel, 9 ) << "tmp.s0 = realTileIn[ gInd ];" << std::endl;
				clKernWrite( transKernel, 9 ) << "tmp.s1 = imagTileIn[ gInd ];" << std::endl;
				break;
			case CLFFT_HERMITIAN_INTERLEAVED:
			case CLFFT_HERMITIAN_PLANAR:
				return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			case CLFFT_REAL:
				clKernWrite( transKernel, 9 ) << "tmp = tileIn[ gInd ];" << std::endl;
				break;

			}

			if(branchingInAny)
			{
				clKernWrite( transKernel, 9 ) << "}" << std::endl;
				clKernWrite( transKernel, 9 ) << std::endl;
			}

			clKernWrite( transKernel, 9 ) << "// Transpose of Tile data happens here" << std::endl;


			// If requested, generate the Twiddle math to multiply constant values
			if( params.fft_3StepTwiddle )
				genTwiddleMath( params, transKernel, dtComplex, fwd );

			clKernWrite( transKernel, 9 ) << "lds[ xInd ][ yInd ] = tmp; " << std::endl;
			clKernWrite( transKernel, 6 ) << "}" << std::endl;

			if(branchingInAny)
				clKernWrite( transKernel, 3 ) << "}" << std::endl;
		}

		clKernWrite( transKernel, 3 ) << std::endl;
		clKernWrite( transKernel, 3 ) << "barrier( CLK_LOCAL_MEM_FENCE );" << std::endl;
		clKernWrite( transKernel, 3 ) << std::endl;

		OffsetCalc(transKernel, params, false);


		switch( params.fft_outputLayout )
		{
		case CLFFT_COMPLEX_INTERLEAVED:
			clKernWrite( transKernel, 3 ) << "global " << dtOutput << "* tileOut = " << pmComplexOut << " + oOffset;" << std::endl << std::endl;
			break;
		case CLFFT_COMPLEX_PLANAR:
			clKernWrite( transKernel, 3 ) << "global " << dtOutput << "* realTileOut = " << pmRealOut << " + oOffset;" << std::endl;
			clKernWrite( transKernel, 3 ) << "global " << dtOutput << "* imagTileOut = " << pmImagOut << " + oOffset;" << std::endl;
			break;
		case CLFFT_HERMITIAN_INTERLEAVED:
		case CLFFT_HERMITIAN_PLANAR:
			return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
		case CLFFT_REAL:
			clKernWrite( transKernel, 3 ) << "global " << dtOutput << "* tileOut = " << pmRealOut << " + oOffset;" << std::endl << std::endl;
			break;
		}

		// Write the transposed values from LDS into global memory
		clKernWrite( transKernel, 3 ) << "rowSizeinUnits = " << params.fft_outStride[ 1 ] << ";" << std::endl; 
		clKernWrite( transKernel, 3 ) << "const size_t transposeRatio = wgTileExtent.x / ( wgTileExtent.y * wgUnroll );" << std::endl;
		clKernWrite( transKernel, 3 ) << "const size_t groupingPerY = wgUnroll / wgTileExtent.y;" << std::endl;
		clKernWrite( transKernel, 3 ) << std::endl << std::endl;

		for(size_t i = 0; i<branchBlocks; i++)
		{
			if(branchingInBoth)
				if(i == 0)
				{
					clKernWrite( transKernel, 3 ) << "if( (" << gIndexX << " == " << 
						cornerGroupX << ") && (" << gIndexY << " == " <<
						cornerGroupY << ") )" << std::endl;
					clKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else if(i == 1)
				{
					clKernWrite( transKernel, 3 ) << "else if( " << gIndexX << " == " << 
						cornerGroupX << " )" << std::endl;
					clKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else if(i == 2)
				{
					clKernWrite( transKernel, 3 ) << "else if( " << gIndexY << " == " <<
						cornerGroupY << " )" << std::endl;
					clKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else
				{
					clKernWrite( transKernel, 3 ) << "else" << std::endl;
					clKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
			else if(branchingInAny)
				if(i == 0)
				{
					if(branchingInGroupX)
					{
						clKernWrite( transKernel, 3 ) << "if( " << gIndexX << " == " << 
							cornerGroupX << " )" << std::endl;
						clKernWrite( transKernel, 3 ) << "{" << std::endl;
					}
					else
					{
						clKernWrite( transKernel, 3 ) << "if( " << gIndexY << " == " <<
							cornerGroupY << " )" << std::endl;
						clKernWrite( transKernel, 3 ) << "{" << std::endl;
					}
				}
				else
				{
					clKernWrite( transKernel, 3 ) << "else" << std::endl;
					clKernWrite( transKernel, 3 ) << "{" << std::endl;
				}


			clKernWrite( transKernel, 6 ) << "for( uint t=0; t < wgUnroll; t++ )" << std::endl;
			clKernWrite( transKernel, 6 ) << "{" << std::endl;
			clKernWrite( transKernel, 9 ) << "size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % groupingPerY ); " << std::endl;
			clKernWrite( transKernel, 9 ) << "size_t yInd = localIndex.y/groupingPerY + t * (wgTileExtent.y * transposeRatio); " << std::endl;
			clKernWrite( transKernel, 9 ) << "tmp = lds[ yInd ][ xInd ]; " << std::endl;
			clKernWrite( transKernel, 9 ) << "size_t gInd = xInd + rowSizeinUnits * yInd;" << std::endl;

			if(branchingInBoth)
			{
				if(i == 0)
				{
					clKernWrite( transKernel, 9 ) << std::endl;
					clKernWrite( transKernel, 9 ) << "if( (" << wIndexY << " < " << wIndexXEnd << ") && (" << wIndexX << " < " << wIndexYEnd << ") )" << std::endl;
					clKernWrite( transKernel, 9 ) << "{" << std::endl;
				}
				else if(i == 1)
				{
					clKernWrite( transKernel, 9 ) << std::endl;
					clKernWrite( transKernel, 9 ) << "if( (" << wIndexY << " < " << wIndexXEnd << ") )" << std::endl;
					clKernWrite( transKernel, 9 ) << "{" << std::endl;

				}
				else if(i == 2)
				{
					clKernWrite( transKernel, 9 ) << std::endl;
					clKernWrite( transKernel, 9 ) << "if( (" << wIndexX << " < " << wIndexYEnd << ") )" << std::endl;
					clKernWrite( transKernel, 9 ) << "{" << std::endl;
				}
				else
					clKernWrite( transKernel, 9 ) << "{" << std::endl;
			}
			else if(branchingInAny)
			{
				if(i == 0)
				{
					if(branchingInGroupX)
					{
						clKernWrite( transKernel, 9 ) << std::endl;
						clKernWrite( transKernel, 9 ) << "if( (" << wIndexY << " < " << wIndexXEnd << ") )" << std::endl;
						clKernWrite( transKernel, 9 ) << "{" << std::endl;
					}
					else
					{
						clKernWrite( transKernel, 9 ) << std::endl;
						clKernWrite( transKernel, 9 ) << "if( (" << wIndexX << " < " << wIndexYEnd << ") )" << std::endl;
						clKernWrite( transKernel, 9 ) << "{" << std::endl;
					}
				}
				else
					clKernWrite( transKernel, 9 ) << "{" << std::endl;
			}

			switch( params.fft_outputLayout )
			{
			case CLFFT_COMPLEX_INTERLEAVED:
				clKernWrite( transKernel, 9 ) << "tileOut[ gInd ] = tmp;" << std::endl;
				break;
			case CLFFT_COMPLEX_PLANAR:
				clKernWrite( transKernel, 9 ) << "realTileOut[ gInd ] = tmp.s0;" << std::endl;
				clKernWrite( transKernel, 9 ) << "imagTileOut[ gInd ] = tmp.s1;" << std::endl;
				break;
			case CLFFT_HERMITIAN_INTERLEAVED:
			case CLFFT_HERMITIAN_PLANAR:
				return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
			case CLFFT_REAL:
				clKernWrite( transKernel, 9 ) << "tileOut[ gInd ] = tmp;" << std::endl;
				break;
			}

			if(branchingInAny)
			{
				clKernWrite( transKernel, 9 ) << "}" << std::endl;
			}

			clKernWrite( transKernel, 6 ) << "}" << std::endl;

			if(branchingInAny)
				clKernWrite( transKernel, 3 ) << "}" << std::endl;
		}

		clKernWrite( transKernel, 0 ) << "}\n" << std::endl;

		strKernel = transKernel.str( );
		//std::cout << strKernel;

		if(!params.fft_3StepTwiddle)
			break;
	}

    return CLFFT_SUCCESS;
}


clfftStatus FFTGeneratedTransposeGCNAction::initParams ()
{

    this->signature.fft_precision    = this->plan->precision;
    this->signature.fft_placeness    = this->plan->placeness;
    this->signature.fft_inputLayout  = this->plan->inputLayout;
    this->signature.fft_outputLayout = this->plan->outputLayout;
    this->signature.fft_3StepTwiddle = false;

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
const tile lwSize = { 16, 16 };
const size_t reShapeFactor = 4;   // wgTileSize = { lwSize.x * reShapeFactor, lwSize.y / reShapeFactor }
const size_t outRowPadding = 0;

// This is global, but should consider to be part of FFTPlan
size_t loopCount = 0;
tile blockSize = {0, 0};


//	OpenCL does not take unicode strings as input, so this routine returns only ASCII strings
//	Feed this generator the FFTPlan, and it returns the generated program as a string
clfftStatus FFTGeneratedTransposeGCNAction::generateKernel ( FFTRepo& fftRepo, const cl_command_queue commQueueFFT )
{
	
    switch( this->signature.fft_precision )
    {
    case CLFFT_SINGLE:
    case CLFFT_SINGLE_FAST:
        loopCount = 16;
        break;
    case CLFFT_DOUBLE:
    case CLFFT_DOUBLE_FAST:
        // Double precisions need about half the amount of LDS space as singles do
        loopCount = 8;
        break;
    default:
        return CLFFT_TRANSPOSED_NOTIMPLEMENTED;
        break;
    }

	blockSize.x = lwSize.x * reShapeFactor;
	blockSize.y = lwSize.y / reShapeFactor * loopCount;


    std::string programCode;
    OPENCL_V( genTransposeKernel( this->signature, programCode, lwSize, reShapeFactor, loopCount, blockSize, outRowPadding ), _T( "GenerateTransposeKernel() failed!" ) );

    cl_int status = CL_SUCCESS;
    cl_device_id Device = NULL;
    status = clGetCommandQueueInfo(commQueueFFT, CL_QUEUE_DEVICE, sizeof(cl_device_id), &Device, NULL);
    OPENCL_V( status, _T( "clGetCommandQueueInfo failed" ) );

    cl_context QueueContext = NULL;
    status = clGetCommandQueueInfo(commQueueFFT, CL_QUEUE_CONTEXT, sizeof(cl_context), &QueueContext, NULL);
    OPENCL_V( status, _T( "clGetCommandQueueInfo failed" ) );


    OPENCL_V( fftRepo.setProgramCode( Transpose_GCN, this->getSignatureData(), programCode, Device, QueueContext ), _T( "fftRepo.setclString() failed!" ) );

    // Note:  See genFunctionPrototype( )
    if( this->signature.fft_3StepTwiddle )
    {
        OPENCL_V( fftRepo.setProgramEntryPoints( Transpose_GCN, this->getSignatureData(), "transpose_gcn_tw_fwd", "transpose_gcn_tw_back", Device, QueueContext ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );
    }
    else
    {
        OPENCL_V( fftRepo.setProgramEntryPoints( Transpose_GCN, this->getSignatureData(), "transpose_gcn", "transpose_gcn", Device, QueueContext ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );
    }

    return CLFFT_SUCCESS;
}


clfftStatus FFTGeneratedTransposeGCNAction::getWorkSizes( std::vector< size_t >& globalWS, std::vector< size_t >& localWS )
{

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


    globalWS.clear( );
    globalWS.push_back( numWIX );
    globalWS.push_back( numWIY );

    localWS.clear( );
    localWS.push_back( lwSize.x );
    localWS.push_back( lwSize.y );

    return CLFFT_SUCCESS;
}
