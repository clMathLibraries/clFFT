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
#include "generator.transpose.h"

#define QUOTEMARK(x) #x

#define PLANNAR_READ(z0, z1, z2, z3, gp) \
		ss	<< INDENT2 << QUOTEMARK(z0 = gp[0*HSTRIDE/4*8];\n) \
			<< INDENT2 << QUOTEMARK(z1 = gp[1*HSTRIDE/4*8];\n) \
			<< INDENT2 << QUOTEMARK(z2 = gp[2*HSTRIDE/4*8];\n) \
			<< INDENT2 << QUOTEMARK(z3 = gp[3*HSTRIDE/4*8];\n);

#define INTERLEAVED_READ(z00, z01, z10, z11, z20, z21, z30, z31) \
		ss	<< INDENT2 << QUOTEMARK(z00 = gp[0*HSTRIDE/4*16];\n)     \
			<< INDENT2 << QUOTEMARK(z01 = gp[0*HSTRIDE/4*16 + 1];\n) \
			<< INDENT2 << QUOTEMARK(z10 = gp[1*HSTRIDE/4*16];\n)     \
			<< INDENT2 << QUOTEMARK(z11 = gp[1*HSTRIDE/4*16 + 1];\n) \
			<< INDENT2 << QUOTEMARK(z20 = gp[2*HSTRIDE/4*16];\n)     \
			<< INDENT2 << QUOTEMARK(z21 = gp[2*HSTRIDE/4*16 + 1];\n) \
			<< INDENT2 << QUOTEMARK(z30 = gp[3*HSTRIDE/4*16];\n)     \
			<< INDENT2 << QUOTEMARK(z31 = gp[3*HSTRIDE/4*16 + 1];\n);

#define PLANNAR_WRITE(z0, z1, z2, z3, gp) \
		ss  << INDENT2 << QUOTEMARK(gp[0*VSTRIDE/4*8] = z0;\n) \
			<< INDENT2 << QUOTEMARK(gp[1*VSTRIDE/4*8] = z1;\n) \
			<< INDENT2 << QUOTEMARK(gp[2*VSTRIDE/4*8] = z2;\n) \
			<< INDENT2 << QUOTEMARK(gp[3*VSTRIDE/4*8] = z3;\n);

#define INTERLEAVED_WRITE(z00, z01, z10, z11, z20, z21, z30, z31) \
		ss  << INDENT2 << QUOTEMARK(gp[0*VSTRIDE/4*16]   = z00;\n) \
			<< INDENT2 << QUOTEMARK(gp[0*VSTRIDE/4*16+1] = z01;\n) \
			<< INDENT2 << QUOTEMARK(gp[1*VSTRIDE/4*16]   = z10;\n) \
			<< INDENT2 << QUOTEMARK(gp[1*VSTRIDE/4*16+1] = z11;\n) \
			<< INDENT2 << QUOTEMARK(gp[2*VSTRIDE/4*16]   = z20;\n) \
			<< INDENT2 << QUOTEMARK(gp[2*VSTRIDE/4*16+1] = z21;\n) \
			<< INDENT2 << QUOTEMARK(gp[3*VSTRIDE/4*16]   = z30;\n) \
			<< INDENT2 << QUOTEMARK(gp[3*VSTRIDE/4*16+1] = z31;\n);

#define WRITE_TO_LDS(lp, jump, z0, z1, z2, z3, part) \
		ss	<< INDENT2 << QUOTEMARK(lp[0*jump] = z0.part;\n) \
			<< INDENT2 << QUOTEMARK(lp[1*jump] = z1.part;\n) \
			<< INDENT2 << QUOTEMARK(lp[2*jump] = z2.part;\n) \
			<< INDENT2 << QUOTEMARK(lp[3*jump] = z3.part;\n) \
			<< INDENT2 << QUOTEMARK(lp += jump*4;\n)          \
			<< "\n";

typedef enum inputoutputflag_
{
	PLANNAR_PLANNAR		= 1,
	PLANNAR_INTERLEAVED,
	INTERLEAVED_PLANNAR,
	INTERLEAVED_INTERLEAVED,
	ENDTRANSIO
} transio;

static clfftStatus GenerateTransposeKernel (FFTKernelGenKeyParams & params,
	std::string & kernel)
{
	kernel.reserve (8000);
	std::stringstream ss         (std::stringstream::out);

	const char * szIn0 = "";
	const char * szIn1 = "";
	const char * szOut0 = "";
	const char * szOut1 = "";
	const char * typeIn = "";
	const char * typeOut = "";
	const char * INDENT2 = "";
	const char * INDENT = "    ";
	const char * datatype="";
	const char * datatype2="";
	bool xyflag  = (params.fft_N[0] == params.fft_N[1]) ? false : true;
	transio iotype;


	if (params.fft_precision == CLFFT_SINGLE)
	{
		datatype  = "float";
		datatype2 = "float2";
	}
	else
	{
		datatype  = "double";
		datatype2 = "double2";
		ss << "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n\n";
	}

	size_t hstride = params.fft_N[0];
	size_t vstride = params.fft_N[1];

	ss << "#define HSTRIDE " << hstride << "\n";
	ss << "#define VSTRIDE " << vstride << "\n";
	if (xyflag)
	{
		ss << "#define DIMX " << hstride/32 << "\n";
		ss << "#define DIMY " << vstride/32 << "\n";
		ss << "#define DIM ((DIMX > DIMY) ? DIMX : DIMY) \n\n";
		INDENT2 = "        ";
	}
	else
	{
		ss << "#define DIM " << vstride/32 << "\n\n";
		INDENT2 = "    ";
	}

	//	Generate the kernel entry point and parameter list
	//
	ss << "__attribute__((reqd_work_group_size(" << unsigned(params.fft_SIMD) << ",1,1)))\n"
	   << "__kernel void\n"
	   << "fft_trans" << "(";

	if (xyflag && params.fft_placeness == CLFFT_INPLACE) return CLFFT_INVALID_ARG_VALUE;

	switch (params.fft_inputLayout) {
	case CLFFT_COMPLEX_INTERLEAVED:
		typeIn = datatype2;
		if (params.fft_placeness == CLFFT_INPLACE)
		{
			szIn0 = szOut0 = "gcomplx";
			typeOut = datatype2;
			ss  << "__global " << typeIn << " * restrict " << szIn0;
			iotype = INTERLEAVED_INTERLEAVED;
		}
		else
		{
			szIn0  = "gcomplxIn";
			ss  << "__global " << typeIn << " * restrict " << szIn0;

			if (params.fft_outputLayout == CLFFT_COMPLEX_PLANAR)
			{
				szOut0 = "grealOut";
				szOut1 = "gimagOut";
				typeOut = datatype;
				ss  <<", __global " << typeOut <<" * restrict " << szOut0
					<<", __global " << typeOut <<" * restrict " << szOut1;
				iotype = INTERLEAVED_PLANNAR;
			}
			else
			{
				szOut0 = "gcomplxOut";
				typeOut = datatype2;
				ss  <<", __global " << typeOut <<" * restrict " << szOut0;
				iotype = INTERLEAVED_INTERLEAVED;
			}
		}
		break;
	case CLFFT_COMPLEX_PLANAR:
		typeIn = datatype;
		if (params.fft_placeness == CLFFT_INPLACE)
		{
			szIn0 = szOut0 = "greal";
			szIn1 = szOut1 = "gimag";
			typeOut = datatype;
			ss << "__global " << typeIn << " * restrict " << szIn0 <<", __global " << typeIn <<" * restrict " << szIn1;
			iotype = PLANNAR_PLANNAR;
		}
		else
		{
			szIn0  = "greadIn";
			szIn1  = "gimagIn";
			ss  <<"__global " << typeIn << " * restrict " << szIn0 <<", __global " << typeIn <<" * restrict " << szIn1;
			if (params.fft_outputLayout == CLFFT_COMPLEX_PLANAR)
			{
				szOut0 = "grealOut";
				szOut1 = "gimagOut";
				typeOut = datatype;
				ss  << ", __global " << typeOut <<" * restrict " << szOut0 <<", __global " << typeOut <<" * restrict " << szOut1;
				iotype = PLANNAR_PLANNAR;
			}
			else
			{
				szOut0 = "gcomplxOut";
				typeOut = datatype2;
				ss  << ", __global " << typeOut <<" * restrict " << szOut0;
				iotype = PLANNAR_INTERLEAVED;
			}
		}
		break;
	default:
		return CLFFT_NOTIMPLEMENTED;
	}
	ss << ")\n{\n";

	// Support plannar and interleaved format
	switch (iotype)
	{
		case PLANNAR_INTERLEAVED:
			ss  << INDENT << "__local " << typeIn << " ldsa[2048];\n"
				<< INDENT << "__local " << typeIn << " ldsb[2048];\n";
			break;
		case INTERLEAVED_PLANNAR:
		case PLANNAR_PLANNAR:
		case INTERLEAVED_INTERLEAVED:
			ss  << INDENT << "__local " << typeIn << " ldsa[1024];\n"
				<< INDENT << "__local " << typeIn << " ldsb[1024];\n";
			break;
		default:
		return CLFFT_NOTIMPLEMENTED;
	}

	ss	<< INDENT << "uint gid = get_global_id(0);\n"
		<< INDENT << "uint me = gid & 0x3fU;\n"
		<< INDENT << "uint k = (gid >> 6) % ";

	// add batch support
	size_t batchnum = (vstride > hstride) ? ((vstride/32) * (vstride/32 +1) /2)
		: ((hstride/32) * (hstride/32 +1) /2);
	ss  << batchnum
		<< ";\n"
		<< "\n";

	ss	<< INDENT << "// Compute location of blocks\n"
		<< INDENT << "int l = DIM+0.5f - native_sqrt((DIM+0.5f)*(DIM+0.5f) - 2.0f * (float)as_int(k));\n"
		<< INDENT << "int kl = ((DIM*2+1 - l) * l) >> 1;\n"
		<< INDENT << "uint j = k - kl;\n"
		<< INDENT << "uint i = l + j;\n"
		<< "\n";

	ss  << INDENT << "uint goa, gob;\n"
		<< INDENT << "uint go = ((me & 0x7U) << 2) + ((gid>>6)/" << batchnum << ") * VSTRIDE * HSTRIDE;\n"
		<< INDENT << "__global " << datatype << "4 *gp;\n"
		<< INDENT << "__local " << datatype << "4 *lp4;\n"
		<< INDENT << "uint lo = ((me & 0x7U) << 7) + (me >> 3);\n"
		<< INDENT << "uint lot = (me<<2); \n";

	switch (iotype)
	{
	case PLANNAR_PLANNAR:
		ss  << INDENT << datatype <<"4 z0, z1, z2, z3;\n\n"
			<< INDENT << "__local " << typeIn <<" *lp;\n";
		break;
	case PLANNAR_INTERLEAVED:
		ss  << INDENT << "__global " << datatype << "4 *gpi;\n"
			<< INDENT << "__local " << typeIn <<" *lp;\n"
			<< INDENT << datatype << "4 z00, z01, z10, z11, z20, z21, z30, z31;\n\n";
		break;
	case INTERLEAVED_PLANNAR:
		ss  << INDENT << "__local " << typeOut <<" *lp;\n"
			<< INDENT << datatype << "4 z00, z01, z10, z11, z20, z21, z30, z31;\n\n";
		break;
	case INTERLEAVED_INTERLEAVED:
		ss  << INDENT << "__local " << typeIn <<" *lp;\n"
			<< INDENT << datatype << "4 z00, z01, z10, z11, z20, z21, z30, z31;\n\n";
		break;
	}

	if (xyflag)
	{
		ss	<< INDENT << "if ( i < DIMX  && j < DIMY) \n"
			<< INDENT << "{\n";
	}

	ss  << INDENT2 << "// Array offsets\n"
		<< INDENT2 << "goa = go + (i << 5) + j * (HSTRIDE*32) + (me >> 3)*HSTRIDE;\n"
		<<"\n";

	ss  << INDENT2 << "// Load A block\n"
		<< INDENT2 << "gp = (__global " << datatype << "4 *)(" << szIn0 <<" + goa);\n";

	switch (iotype)
	{
	case PLANNAR_PLANNAR:
		PLANNAR_READ(z0, z1, z2, z3, gp);
		break;
	case PLANNAR_INTERLEAVED:
		PLANNAR_READ(z00, z10, z20, z30, gp);
		ss << INDENT2 << "gpi = (__global " << datatype << "4 *)(" << szIn1 <<" + goa);\n";
		PLANNAR_READ(z01, z11, z21, z31, gpi);
		break;
	default:
		INTERLEAVED_READ(z00, z01, z10, z11, z20, z21, z30, z31);
		break;
	}
	ss	<< "\n";

	ss  << INDENT2 << "// Save into LDS\n";
	switch (iotype)
	{
	case PLANNAR_PLANNAR:
		ss	<< INDENT2 << "lp = ldsa + lo;\n";
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, x);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, y);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, z);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, w);
		break;
	case PLANNAR_INTERLEAVED:
		ss	<< INDENT2 << "lp = ldsa + lo*2;\n";
		WRITE_TO_LDS(lp, 16, z00, z10, z20, z30, x);
		WRITE_TO_LDS(lp, 16, z00, z10, z20, z30, y);
		WRITE_TO_LDS(lp, 16, z00, z10, z20, z30, z);
		WRITE_TO_LDS(lp, 16, z00, z10, z20, z30, w);
		ss  << INDENT2 << "lp = ldsa + lo*2 + 1;\n";
		WRITE_TO_LDS(lp, 16, z01, z11, z21, z31, x);
		WRITE_TO_LDS(lp, 16, z01, z11, z21, z31, y);
		WRITE_TO_LDS(lp, 16, z01, z11, z21, z31, z);
		WRITE_TO_LDS(lp, 16, z01, z11, z21, z31, w);
		break;
	case INTERLEAVED_INTERLEAVED:
		ss	<< INDENT2 << "lp = ldsa + lo;\n";
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, xy);
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, zw);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, xy);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, zw);
		break;
	case INTERLEAVED_PLANNAR:
		ss	<< INDENT2 << "lp = (__local " << typeOut << "*)ldsa + lo;\n";
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, x);
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, z);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, x);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, z);
		//next write to lp = ldsa+lo+1024
		ss  << INDENT2 << "lp += (1024-32*4);\n";
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, y);
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, w);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, y);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, w);
		break;
	}

	ss << INDENT;
	if (xyflag)	ss << "} ";
	ss <<"//End load A block\n\n";

	if (xyflag)
		ss  << INDENT << "if (i < DIMY  && j < DIMX) \n"
			<< INDENT << "{\n";

	ss  << INDENT2 << "//  Load B block\n"
		<< INDENT2 << "gob = go + (j << 5) + i * (HSTRIDE*32) + (me >> 3)*HSTRIDE;\n"
		<< INDENT2 << "gp = (__global " << datatype << "4 *)(" << szIn0 << " + gob);\n";

	switch (iotype)
	{
	case PLANNAR_PLANNAR:
		PLANNAR_READ(z0, z1, z2, z3, gp);
		break;
	case PLANNAR_INTERLEAVED:
		PLANNAR_READ(z00, z10, z20, z30, gp);
		ss << INDENT2 << "gpi = (__global " << datatype << "4 *)(" << szIn1 <<" + gob);\n";
		PLANNAR_READ(z01, z11, z21, z31, gpi);
		break;
	default:
		INTERLEAVED_READ(z00, z01, z10, z11, z20, z21, z30, z31);
		break;
	}
	ss	<< "\n";

	ss  << INDENT2 << "// Save into LDS\n";
	switch (iotype)
	{
	case PLANNAR_PLANNAR:
		ss	<< INDENT2 << "lp = ldsb + lo;\n";
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, x);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, y);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, z);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, w);
		break;
	case PLANNAR_INTERLEAVED:
		ss	<< INDENT2 << "lp = ldsb + lo*2;\n";
		WRITE_TO_LDS(lp, 16, z00, z10, z20, z30, x);
		WRITE_TO_LDS(lp, 16, z00, z10, z20, z30, y);
		WRITE_TO_LDS(lp, 16, z00, z10, z20, z30, z);
		WRITE_TO_LDS(lp, 16, z00, z10, z20, z30, w);
		ss  << INDENT2 << "lp = ldsb + lo*2 + 1;\n";
		WRITE_TO_LDS(lp, 16, z01, z11, z21, z31, x);
		WRITE_TO_LDS(lp, 16, z01, z11, z21, z31, y);
		WRITE_TO_LDS(lp, 16, z01, z11, z21, z31, z);
		WRITE_TO_LDS(lp, 16, z01, z11, z21, z31, w);
		break;
	case INTERLEAVED_INTERLEAVED:
		ss	<< INDENT2 << "lp = ldsb + lo;\n";
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, xy);
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, zw);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, xy);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, zw);
		break;
	case INTERLEAVED_PLANNAR:
		ss	<< INDENT2 << "lp = (__local " << typeOut << "*) ldsb + lo;\n";
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, x);
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, z);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, x);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, z);
		//next write to lp = ldsa+lo+1024
		ss  << INDENT2 << "lp += (1024-32*4);\n";
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, y);
		WRITE_TO_LDS(lp, 8, z00, z10, z20, z30, w);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, y);
		WRITE_TO_LDS(lp, 8, z01, z11, z21, z31, w);
		break;
	}

	ss  << INDENT;
	if (xyflag) ss<< "} ";
	ss  << "// End load B block\n\n";

	ss  << INDENT << "barrier(CLK_LOCAL_MEM_FENCE);\n"
		<< "\n";

	if (xyflag) ss  << INDENT << "if (i < DIMY  && j < DIMX)\n"
		<< INDENT << "{\n";

	ss  << INDENT2 << "// write A block\n";

	ss  << INDENT2 << "goa = go + (i << 5) + j * (VSTRIDE*32) + (me >> 3)*VSTRIDE;\n"
		<< INDENT2 << "gp = (__global " << datatype << "4 *)(" << szOut0 << " + goa);\n";

	switch (iotype)
	{
	case PLANNAR_PLANNAR:
		ss  << INDENT2 << "lp4 = (__local " << datatype << "4 *)(ldsb + lot);\n\n";
		ss  << INDENT2 << "z0 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z1 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z2 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z3 = lp4[0];\n\n";

		PLANNAR_WRITE(z0, z1, z2, z3, gp);
		break;
	case INTERLEAVED_PLANNAR:
		ss  << INDENT2 << "lp4 = (__local " << datatype << "4 *)((__local " << typeOut << "*)ldsb + lot);\n\n";
		ss  << INDENT2 << "z00 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z10 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z20 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z30 = lp4[0];\n\n";

		PLANNAR_WRITE(z00, z10, z20, z30, gp);

		ss  << INDENT2 << "gp = (__global " << datatype << "4 *)(" << szOut1 << " + goa);\n";
		ss  << INDENT2 << "lp4 += (256 - 32*6);\n";
		ss  << INDENT2 << "z01 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z11 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z21 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z31 = lp4[0];\n\n";

		PLANNAR_WRITE(z01, z11, z21, z31, gp);
		break;

	case PLANNAR_INTERLEAVED:
	case INTERLEAVED_INTERLEAVED:
		if (iotype == PLANNAR_INTERLEAVED)
			ss  << INDENT2 << "lp4 = (__local " << datatype << "4 *)(ldsb + lot*2);\n";
		else
			ss  << INDENT2 << "lp4 = (__local " << datatype << "4 *)(ldsb + lot);\n";

		ss  << INDENT2 << "z00 = lp4[0];\n"
			<< INDENT2 << "z01 = lp4[1];\n"
			<< INDENT2 << "lp4 += 32*4;\n"
			<< "\n";

		ss  << INDENT2 << "z10 = lp4[0];\n"
			<< INDENT2 << "z11 = lp4[1];\n"
			<< INDENT2 << "lp4 += 32*4;\n"
			<< "\n";

		ss	<< INDENT2 << "z20 = lp4[0];\n"
			<< INDENT2 << "z21 = lp4[1];\n"
			<< INDENT2 << "lp4 += 32*4;\n"
			<< "\n";

		ss	<< INDENT2 << "z30 = lp4[0];\n"
			<< INDENT2 << "z31 = lp4[1];\n\n";

		INTERLEAVED_WRITE(z00, z01, z10, z11, z20, z21, z30, z31);
		break;
	}
	ss << "\n";

	ss  << INDENT;
	if (xyflag) ss << "} ";
	ss  << "// End write A block;\n\n";

	if (xyflag)	ss  << INDENT << "if (i < DIMX  && j < DIMY)\n"
		<< INDENT << "{\n";

	ss  << INDENT2 << "// write B block\n\n";
	ss	<< INDENT2 << "gob = go + (j << 5) + i * (VSTRIDE*32) + (me >> 3)*VSTRIDE;\n"
		<< INDENT2 << "gp = (__global " << datatype << "4 *)(" << szOut0 << " + gob);\n";

	switch (iotype)
	{
	case PLANNAR_PLANNAR:
		ss  << INDENT2 << "lp4 = (__local " << datatype << "4 *)(ldsa + lot);\n\n";
		ss  << INDENT2 << "z0 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z1 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z2 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z3 = lp4[0];\n\n";

		PLANNAR_WRITE(z0, z1, z2, z3, gp);
		break;
	case INTERLEAVED_PLANNAR:
		ss  << INDENT2 << "lp4 = (__local " << datatype << "4 *)((__local " << typeOut << "*)ldsa + lot);\n\n";
		ss  << INDENT2 << "z00 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z10 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z20 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z30 = lp4[0];\n\n";

		PLANNAR_WRITE(z00, z10, z20, z30, gp);

		ss  << INDENT2 << "gp = (__global " << datatype << "4 *)(" << szOut1 << " + gob);\n";
		ss  << INDENT2 << "lp4 += (256 - 32*6);\n";
		ss  << INDENT2 << "z01 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z11 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z21 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss	<< INDENT2 << "z31 = lp4[0];\n\n";

		PLANNAR_WRITE(z01, z11, z21, z31, gp);
		break;

	case PLANNAR_INTERLEAVED:
	case INTERLEAVED_INTERLEAVED:
		if (iotype == PLANNAR_INTERLEAVED)
			ss  << INDENT2 << "lp4 = (__local " << datatype << "4 *)(ldsa + lot*2);\n\n";
		else
			ss  << INDENT2 << "lp4 = (__local " << datatype << "4 *)(ldsa + lot);\n\n";

		ss  << INDENT2 << "z00 = lp4[0];\n"
			<< INDENT2 << "z01 = lp4[1];\n"
			<< INDENT2 << "lp4 += 32*4;\n"
			<< "\n";

		ss  << INDENT2 << "z10 = lp4[0];\n"
			<< INDENT2 << "z11 = lp4[1];\n"
			<< INDENT2 << "lp4 += 32*4;\n"
			<< "\n";

		ss	<< INDENT2 << "z20 = lp4[0];\n"
			<< INDENT2 << "z21 = lp4[1];\n"
			<< INDENT2 << "lp4 += 32*4;\n"
			<< "\n";

		ss	<< INDENT2 << "z30 = lp4[0];\n"
			<< INDENT2 << "z31 = lp4[1];\n\n";

		INTERLEAVED_WRITE(z00, z01, z10, z11, z20, z21, z30, z31);
		break;
	}
	ss << "\n";

	ss  << INDENT;
	if(xyflag) ss  << "} ";
	ss  << "// End write B block;\n\n";

	if (iotype == PLANNAR_PLANNAR)
	{
		ss  << INDENT << "// Identical handling for imaginary data\n"
			<< INDENT << "barrier(CLK_LOCAL_MEM_FENCE);\n"
			<< "\n";

		if (xyflag) ss	<< INDENT << "if (i < DIMX  && j < DIMY)\n"
			<< INDENT << "{\n";

		ss  << INDENT2 << "//load A block\n"
			<< INDENT2 << "goa = go + (i << 5) + j * (HSTRIDE*32) + (me >> 3)*HSTRIDE;\n"
			<< INDENT2 << "gp = (__global " << datatype << "4 *)(" << szIn1 << " + goa);\n"
			<< "\n";

		PLANNAR_READ(z0, z1, z2, z3, gp);

		ss  << INDENT2 << "lp = ldsa + lo;\n"
			<< "\n";

		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, x);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, y);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, z);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, w);

		ss  << INDENT;
		if (xyflag) ss  << "} ";
		ss  << "//end load A block\n\n";

		if (xyflag) ss	<< INDENT << "if (i < DIMY  && j < DIMX)\n"
			<< INDENT << "{\n";

		ss  << INDENT2 << "//load B block\n"
			<< INDENT2 << "gob = go + (j << 5) + i * (HSTRIDE*32) + (me >> 3)*HSTRIDE;\n"
			<< INDENT2 << "gp = (__global " << datatype << "4 *)(" << szIn1 <<" + gob);\n"
			<< "\n";

		PLANNAR_READ(z0, z1, z2, z3, gp);

		ss  << INDENT2 << "lp = ldsb + lo;\n"
			<< "\n";

		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, x);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, y);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, z);
		WRITE_TO_LDS(lp, 8, z0, z1, z2, z3, w);

		ss  << INDENT;
		if (xyflag) ss << "} ";
		ss  << "// end load B block\n\n";

		ss  << INDENT << "barrier(CLK_LOCAL_MEM_FENCE);\n"
			<< "\n";

		if (xyflag) ss	<< INDENT << "if (i < DIMY  && j < DIMX)\n"
			<< INDENT << "{\n";

		ss  << INDENT2 << "//Write A block\n"
		    << INDENT2 << "lp4 = (__local " << datatype << "4 *)(ldsb + lot);\n"
			<< "\n";

		ss  << INDENT2 << "z0 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z1 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z2 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z3 = lp4[0];\n"
			<< "\n";

		ss  << INDENT2 << "goa = go + (i << 5) + j * (VSTRIDE*32) + (me >> 3)*VSTRIDE;\n"
			<< INDENT2 << "gp = (__global " << datatype << "4 *)(" << szOut1 << " + goa);\n"
			<< "\n";

		PLANNAR_WRITE(z0, z1, z2, z3, gp);

		ss  << INDENT;
		if (xyflag) ss  << "} ";
		ss  << "// end write A block\n\n";

		if (xyflag) ss	<< INDENT << "if (i < DIMX  && j < DIMY)\n"
			<< INDENT << "{\n";

		ss  << INDENT2 << "//write B block\n"
			<< INDENT2 << "lp4 = (__local " << datatype << "4 *)(ldsa + lot);\n"
			<< "\n";

		ss  << INDENT2 << "z0 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z1 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z2 = lp4[0];\n"
			<< INDENT2 << "lp4 += 32*2;\n"
			<< "\n";

		ss  << INDENT2 << "z3 = lp4[0];\n"
			<< "\n";

		ss  << INDENT2 << "gob = go + (j << 5) + i * (VSTRIDE*32) + (me >> 3)*VSTRIDE;\n"
			<< INDENT2 << "gp = (__global " << datatype << "4 *)(" << szOut1 << " + gob);\n";
		PLANNAR_WRITE(z0, z1, z2, z3, gp);

		ss  << INDENT;
		if (xyflag) ss  << "} ";
		ss << "// end write B block\n";
	}

	ss << "}\n\n";
	kernel = ss.str();
	return CLFFT_SUCCESS;
}

template<>
clfftStatus FFTPlan::GetKernelGenKeyPvt<Transpose> (FFTKernelGenKeyParams & params) const
{

	//	Query the devices in this context for their local memory sizes
	//	How we generate a kernel depends on the *minimum* LDS size for all devices.
	//
	const FFTEnvelope * pEnvelope = NULL;
	OPENCL_V(const_cast<FFTPlan*>(this)->GetEnvelope (& pEnvelope), _T("GetEnvelope failed"));
	BUG_CHECK (NULL != pEnvelope);

	::memset( &params, 0, sizeof( params ) );
	params.fft_precision    = this->precision;
	params.fft_placeness    = this->placeness;
	params.fft_inputLayout  = this->inputLayout;

	ARG_CHECK (this->inStride.size() == this->outStride.size())

	if (CLFFT_INPLACE == this->placeness) {
		//	If this is an in-place transform the
		//	input and output layout, dimensions and strides
		//	*MUST* be the same.
		//
		ARG_CHECK (this->inputLayout == this->outputLayout)
		params.fft_outputLayout = this->inputLayout;
		for (size_t u = this->inStride.size(); u-- > 0; ) {
			ARG_CHECK (this->inStride[u] == this->outStride[u]);
		}
	} else {
		params.fft_outputLayout = this->outputLayout;
	}

	//we only support 2D transpose
	switch (this->inStride.size()) {
		//	2-D array is a 3-D data structure
		//	2-D unit is a speical case of 2-D array.
	case 2:
		ARG_CHECK(this->length   .size() > 1);
		ARG_CHECK(this->outStride.size() > 1);
		params.fft_DataDim      = 3;
		params.fft_N[0]         = this->length[0];
		params.fft_N[1]         = this->length[1];
		params.fft_inStride[0]  = this->inStride[0];
		params.fft_inStride[1]  = this->inStride[1];
		params.fft_inStride[2]  = this->iDist;
		params.fft_outStride[0] = this->outStride[0];
		params.fft_outStride[1] = this->outStride[1];
		params.fft_outStride[2] = this->oDist;
		break;
	default:
		ARG_CHECK (false);
	}

	//ToDO: work group size setup
	params.fft_R = 32; // divide the element into 32x32 blocks
	params.fft_SIMD = 64; //work group size

	return CLFFT_SUCCESS;
}

template<>
clfftStatus FFTPlan::GetWorkSizesPvt<Transpose> (std::vector<size_t> & globalWS, std::vector<size_t> & localWS) const
{
	//	How many numbers per workitem in the generated kernel?
	FFTKernelGenKeyParams fftParams;
	//	Translate the user plan into the structure that we use to map plans to clPrograms
	OPENCL_V( this->GetKernelGenKeyPvt<Transpose>( fftParams ), _T("GetKernelGenKey() failed!") );

	unsigned long long count, count0, count1;
	count0 = DivRoundingUp<unsigned long long> (this->length[0], fftParams.fft_R);
	count1 = DivRoundingUp<unsigned long long> (this->length[1], fftParams.fft_R);
	count  = (count0>count1) ? count0 : count1;
	count  = (count * (count+1)) /2;
	count *= fftParams.fft_SIMD;
	count *= this->batchsize;

	globalWS.push_back( static_cast< size_t >( count ) );
	localWS.push_back( fftParams.fft_SIMD );

	return	CLFFT_SUCCESS;
}


//	OpenCL does not take unicode strings as input, so this routine returns only ASCII strings
//	Feed this generator the FFTPlan, and it returns the generated program as a string
template<>
clfftStatus FFTPlan::GenerateKernelPvt<Transpose> ( FFTRepo& fftRepo, const cl_command_queue commQueueFFT ) const
{
	FFTKernelGenKeyParams params;
	OPENCL_V( this->GetKernelGenKeyPvt<Transpose> (params), _T("GetKernelGenKey() failed!") );

	std::string programCode;
	OPENCL_V( GenerateTransposeKernel( params, programCode ), _T( "GenerateTransposeKernel() failed!" ) );

  cl_int status = CL_SUCCESS;
  cl_context QueueContext = NULL;
  status = clGetCommandQueueInfo(commQueueFFT, CL_QUEUE_CONTEXT, sizeof(cl_context), &QueueContext, NULL);

  OPENCL_V( status, _T( "clGetCommandQueueInfo failed" ) );

  OPENCL_V( fftRepo.setProgramCode( Transpose, params, programCode, QueueContext ), _T( "fftRepo.setclString() failed!" ) );
	OPENCL_V( fftRepo.setProgramEntryPoints( Transpose, params, "fft_trans", "fft_trans",QueueContext ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );

	return CLFFT_SUCCESS;
}
