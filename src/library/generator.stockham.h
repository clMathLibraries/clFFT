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

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

#pragma once
#if !defined( AMD_CLFFT_generator_stockham_H )
#define AMD_CLFFT_generator_stockham_H
#include <stdio.h>
#include "private.h"
#include "repo.h"
#include "plan.h"

typedef union {
	cl_float f;
	cl_uint  u;
	cl_int   i;
} cb_t;

namespace StockhamGenerator
{
	// Precision
	enum Precision
	{
		P_SINGLE,
		P_DOUBLE,
	};

	template <Precision PR>
	inline size_t PrecisionWidth()
	{
		switch(PR)
		{
		case P_SINGLE:	return 1;
		case P_DOUBLE:	return 2;
		default:		assert(false); return 1;
		}
	}

	template <Precision PR>
	inline std::string ClPragma()
	{
		switch(PR)
		{
		case P_SINGLE:	return "";
		case P_DOUBLE:	return	"\n#ifdef cl_khr_fp64\n"
								"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
								"#else\n"
								"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
								"#endif\n\n";
		default:		assert(false); return "";
		}
	}

	// Convert unsigned integers to string
	inline std::string SztToStr(size_t i)
	{
		std::stringstream ss;
		ss << i;
		return ss.str();
	}

	inline std::string FloatToStr(double f)
	{
		std::stringstream ss;
		ss.precision(16);
		ss << std::scientific << f;
		return ss.str();
	}


	//	Find the smallest power of 2 that is >= n; return its power of 2 factor
	//	e.g., CeilPo2 (7) returns 3 : (2^3 >= 7)
	inline size_t CeilPo2 (size_t n)
	{
		size_t v = 1, t = 0;
		while(v < n)
		{
			v <<= 1;
			t++;
		}

		return t;
	}

	inline size_t FloorPo2 (size_t n)
	//	return the largest power of 2 that is <= n.
	//	e.g., FloorPo2 (7) returns 4.
	// *** TODO use x86 BSR instruction, using compiler intrinsics.
	{
		size_t tmp;
		while (0 != (tmp = n & (n-1)))
			n = tmp;
		return n;
	}

	typedef std::pair<std::string,std::string> stringpair;
	inline stringpair ComplexMul(const char *type, const char * a, const char * b, bool forward = true)
	{
		stringpair result;
		result.first = "(";
		result.first += type;
		result.first += ") ((";
		result.first += a;
		result.first += ".x * ";
		result.first += b;
		result.first += (forward ? ".x - " : ".x + ");
		result.first += a;
		result.first += ".y * ";
		result.first += b;
		result.first += ".y),";
		result.second = "(";
		result.second += a;
		result.second += ".y * ";
		result.second += b;
		result.second += (forward ? ".x + " : ".x - ");
		result.second += a;
		result.second += ".x * ";
		result.second += b;
		result.second += ".y))";
		return result;
	}


	// Register data base types
	template <Precision PR>
	inline std::string RegBaseType(size_t count)
	{
		switch(PR)
		{
		case P_SINGLE:
			switch(count)
			{
			case 1: return "float";
			case 2: return "float2";
			case 4: return "float4";
			default: assert(false); return "";
			}
			break;
		case P_DOUBLE:
			switch(count)
			{
			case 1: return "double";
			case 2: return "double2";
			case 4: return "double4";
			default: assert(false); return "";
			}
			break;
		default:
			assert(false); return "";
		}
	}

	template <Precision PR>
	inline std::string FloatSuffix()
	{
		// Suffix for constants
		std::string sfx;
		switch(PR)
		{
		case P_SINGLE: sfx = "f"; break;
		case P_DOUBLE: sfx = "";  break;
		default: assert(false);
		}

		return sfx;
	}

	inline std::string ButterflyName(size_t radix, size_t count, bool fwd)
	{
		std::string str;
		if(fwd) str += "Fwd";
		else	str += "Inv";
		str += "Rad"; str += SztToStr(radix);
		str += "B"; str += SztToStr(count);
		return str;
	}

	inline std::string PassName(size_t pos, bool fwd)
	{
		std::string str;
		if(fwd) str += "Fwd";
		else	str += "Inv";
		str += "Pass"; str += SztToStr(pos);
		return str;
	}

	inline std::string TwTableName()
	{
		return "twiddles";
	}

	inline std::string TwTableLargeName()
	{
		return "twiddle_dee";
	}

	inline std::string TwTableLargeFunc()
	{
		return "TW3step";
	}


};

#endif

