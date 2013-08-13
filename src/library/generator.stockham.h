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
#if !defined( AMD_CLFFT_generator_stockham_H )
#define AMD_CLFFT_generator_stockham_H
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

	// FFT butterfly
    template <Precision PR>
    class Butterfly
    {
		size_t radix;		// Base radix
        size_t count;       // Number of basic butterflies, valid values: 1,2,4
		bool fwd;			// FFT direction
		bool cReg;			// registers are complex numbers, .x (real), .y(imag)

		size_t BitReverse (size_t n, size_t N) const
		{
			return (N < 2) ? n : (BitReverse (n >> 1, N >> 1) | ((n & 1) != 0 ? (N >> 1) : 0));
		}

		void GenerateButterflyStr(std::string &bflyStr) const
		{
			std::string regType = cReg ? RegBaseType<PR>(2) : RegBaseType<PR>(count);

			// Function attribute
			bflyStr += "__attribute__((always_inline)) void \n";

			// Function name
			bflyStr += ButterflyName(radix, count, fwd);

			// Function Arguments
			bflyStr += "(";
			for(size_t i=0;;i++)
			{
				if(cReg)
				{
					bflyStr += regType; bflyStr += " *R";
					if(radix & (radix-1))	bflyStr += SztToStr(i);
					else					bflyStr += SztToStr(BitReverse(i,radix));
				}
				else
				{
					bflyStr += regType; bflyStr += " *R"; bflyStr += SztToStr(i); bflyStr += ", ";	// real arguments
					bflyStr += regType; bflyStr += " *I"; bflyStr += SztToStr(i);					// imaginary arguments
				}

				if(i == radix-1)
				{
					bflyStr += ")";
					break;
				}
				else
				{
					bflyStr += ", ";
				}
			}

			bflyStr += "\n{\n\n";


			// Temporary variables
			// Allocate temporary variables if we are not using complex registers (cReg = 0) or if cReg is true, then
			// allocate temporary variables only for non power-of-2 radices
			if( (radix & (radix-1)) || (!cReg) )
			{
				bflyStr += "\t";
				if(cReg)
					bflyStr += RegBaseType<PR>(1);
				else
					bflyStr += regType;

				for(size_t i=0;;i++)
				{
					bflyStr += " TR"; bflyStr += SztToStr(i); bflyStr += ",";	// real arguments
					bflyStr += " TI"; bflyStr += SztToStr(i);					// imaginary arguments

					if(i == radix-1)
					{
						bflyStr += ";";
						break;
					}
					else
					{
						bflyStr += ",";
					}
				}
			}
			else
			{
				bflyStr += "\t";
				bflyStr += RegBaseType<PR>(2);
				bflyStr += " T;";
			}


			bflyStr += "\n\n\t";

			// Butterfly for different radices
			switch(radix)
			{
			case 2:
				{
					if(cReg)
					{
						bflyStr +=
						"(*R1) = (*R0) - (*R1);\n\t"
						"(*R0) = 2.0f * (*R0) - (*R1);\n\t";
					}
					else
					{
						bflyStr +=
						"TR0 = (*R0) + (*R1);\n\t"
						"TI0 = (*I0) + (*I1);\n\t"
						"TR1 = (*R0) - (*R1);\n\t"
						"TI1 = (*I0) - (*I1);\n\t";
					}

				} break;
			case 3:
				{
					if(fwd)
					{
						if(cReg)
						{
							bflyStr +=
							"TR0 = (*R0).x + (*R1).x + (*R2).x;\n\t"
							"TR1 = ((*R0).x - C3QA*((*R1).x + (*R2).x)) + C3QB*((*R1).y - (*R2).y);\n\t"
							"TR2 = ((*R0).x - C3QA*((*R1).x + (*R2).x)) - C3QB*((*R1).y - (*R2).y);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*R0).y + (*R1).y + (*R2).y;\n\t"
							"TI1 = ((*R0).y - C3QA*((*R1).y + (*R2).y)) - C3QB*((*R1).x - (*R2).x);\n\t"
							"TI2 = ((*R0).y - C3QA*((*R1).y + (*R2).y)) + C3QB*((*R1).x - (*R2).x);\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = *R0 + *R1 + *R2;\n\t"
							"TR1 = (*R0 - C3QA*(*R1 + *R2)) + C3QB*(*I1 - *I2);\n\t"
							"TR2 = (*R0 - C3QA*(*R1 + *R2)) - C3QB*(*I1 - *I2);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = *I0 + *I1 + *I2;\n\t"
							"TI1 = (*I0 - C3QA*(*I1 + *I2)) - C3QB*(*R1 - *R2);\n\t"
							"TI2 = (*I0 - C3QA*(*I1 + *I2)) + C3QB*(*R1 - *R2);\n\t";
						}
					}
					else
					{
						if(cReg)
						{
							bflyStr +=
							"TR0 = (*R0).x + (*R1).x + (*R2).x;\n\t"
							"TR1 = ((*R0).x - C3QA*((*R1).x + (*R2).x)) - C3QB*((*R1).y - (*R2).y);\n\t"
							"TR2 = ((*R0).x - C3QA*((*R1).x + (*R2).x)) + C3QB*((*R1).y - (*R2).y);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*R0).y + (*R1).y + (*R2).y;\n\t"
							"TI1 = ((*R0).y - C3QA*((*R1).y + (*R2).y)) + C3QB*((*R1).x - (*R2).x);\n\t"
							"TI2 = ((*R0).y - C3QA*((*R1).y + (*R2).y)) - C3QB*((*R1).x - (*R2).x);\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = *R0 + *R1 + *R2;\n\t"
							"TR1 = (*R0 - C3QA*(*R1 + *R2)) - C3QB*(*I1 - *I2);\n\t"
							"TR2 = (*R0 - C3QA*(*R1 + *R2)) + C3QB*(*I1 - *I2);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = *I0 + *I1 + *I2;\n\t"
							"TI1 = (*I0 - C3QA*(*I1 + *I2)) + C3QB*(*R1 - *R2);\n\t"
							"TI2 = (*I0 - C3QA*(*I1 + *I2)) - C3QB*(*R1 - *R2);\n\t";
						}
					}
				} break;
			case 4:
				{
					if(fwd)
					{
						if(cReg)
						{
							bflyStr +=
							"(*R1) = (*R0) - (*R1);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R1);\n\t"
							"(*R3) = (*R2) - (*R3);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R3);\n\t"
							"\n\t"
							"(*R2) = (*R0) - (*R2);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R2);\n\t"
							"(*R3) = (*R1) + (fvect2)(-(*R3).y, (*R3).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R3);\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = (*R0) + (*R2) + (*R1) + (*R3);\n\t"
							"TR1 = (*R0) - (*R2) + (*I1) - (*I3);\n\t"
							"TR2 = (*R0) + (*R2) - (*R1) - (*R3);\n\t"
							"TR3 = (*R0) - (*R2) - (*I1) + (*I3);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*I0) + (*I2) + (*I1) + (*I3);\n\t"
							"TI1 = (*I0) - (*I2) - (*R1) + (*R3);\n\t"
							"TI2 = (*I0) + (*I2) - (*I1) - (*I3);\n\t"
							"TI3 = (*I0) - (*I2) + (*R1) - (*R3);\n\t";
						}
					}
					else
					{
						if(cReg)
						{
							bflyStr +=
							"(*R1) = (*R0) - (*R1);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R1);\n\t"
							"(*R3) = (*R2) - (*R3);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R3);\n\t"
							"\n\t"
							"(*R2) = (*R0) - (*R2);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R2);\n\t"
							"(*R3) = (*R1) + (fvect2)((*R3).y, -(*R3).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R3);\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = (*R0) + (*R2) + (*R1) + (*R3);\n\t"
							"TR1 = (*R0) - (*R2) - (*I1) + (*I3);\n\t"
							"TR2 = (*R0) + (*R2) - (*R1) - (*R3);\n\t"
							"TR3 = (*R0) - (*R2) + (*I1) - (*I3);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*I0) + (*I2) + (*I1) + (*I3);\n\t"
							"TI1 = (*I0) - (*I2) + (*R1) - (*R3);\n\t"
							"TI2 = (*I0) + (*I2) - (*I1) - (*I3);\n\t"
							"TI3 = (*I0) - (*I2) - (*R1) + (*R3);\n\t";
						}
					}
				} break;
			case 5:
				{
					if(fwd)
					{
						if(cReg)
						{
							bflyStr +=
							"TR0 = (*R0).x + (*R1).x + (*R2).x + (*R3).x + (*R4).x;\n\t"
							"TR1 = ((*R0).x - C5QC*((*R2).x + (*R3).x)) + C5QB*((*R1).y - (*R4).y) + C5QD*((*R2).y - (*R3).y) + C5QA*(((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));\n\t"
							"TR4 = ((*R0).x - C5QC*((*R2).x + (*R3).x)) - C5QB*((*R1).y - (*R4).y) - C5QD*((*R2).y - (*R3).y) + C5QA*(((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));\n\t"
							"TR2 = ((*R0).x - C5QC*((*R1).x + (*R4).x)) - C5QB*((*R2).y - (*R3).y) + C5QD*((*R1).y - (*R4).y) + C5QA*(((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));\n\t"
							"TR3 = ((*R0).x - C5QC*((*R1).x + (*R4).x)) + C5QB*((*R2).y - (*R3).y) - C5QD*((*R1).y - (*R4).y) + C5QA*(((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*R0).y + (*R1).y + (*R2).y + (*R3).y + (*R4).y;\n\t"
							"TI1 = ((*R0).y - C5QC*((*R2).y + (*R3).y)) - C5QB*((*R1).x - (*R4).x) - C5QD*((*R2).x - (*R3).x) + C5QA*(((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));\n\t"
							"TI4 = ((*R0).y - C5QC*((*R2).y + (*R3).y)) + C5QB*((*R1).x - (*R4).x) + C5QD*((*R2).x - (*R3).x) + C5QA*(((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));\n\t"
							"TI2 = ((*R0).y - C5QC*((*R1).y + (*R4).y)) + C5QB*((*R2).x - (*R3).x) - C5QD*((*R1).x - (*R4).x) + C5QA*(((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));\n\t"
							"TI3 = ((*R0).y - C5QC*((*R1).y + (*R4).y)) - C5QB*((*R2).x - (*R3).x) + C5QD*((*R1).x - (*R4).x) + C5QA*(((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = *R0 + *R1 + *R2 + *R3 + *R4;\n\t"
							"TR1 = (*R0 - C5QC*(*R2 + *R3)) + C5QB*(*I1 - *I4) + C5QD*(*I2 - *I3) + C5QA*((*R1 - *R2) + (*R4 - *R3));\n\t"
							"TR4 = (*R0 - C5QC*(*R2 + *R3)) - C5QB*(*I1 - *I4) - C5QD*(*I2 - *I3) + C5QA*((*R1 - *R2) + (*R4 - *R3));\n\t"
							"TR2 = (*R0 - C5QC*(*R1 + *R4)) - C5QB*(*I2 - *I3) + C5QD*(*I1 - *I4) + C5QA*((*R2 - *R1) + (*R3 - *R4));\n\t"
							"TR3 = (*R0 - C5QC*(*R1 + *R4)) + C5QB*(*I2 - *I3) - C5QD*(*I1 - *I4) + C5QA*((*R2 - *R1) + (*R3 - *R4));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = *I0 + *I1 + *I2 + *I3 + *I4;\n\t"
							"TI1 = (*I0 - C5QC*(*I2 + *I3)) - C5QB*(*R1 - *R4) - C5QD*(*R2 - *R3) + C5QA*((*I1 - *I2) + (*I4 - *I3));\n\t"
							"TI4 = (*I0 - C5QC*(*I2 + *I3)) + C5QB*(*R1 - *R4) + C5QD*(*R2 - *R3) + C5QA*((*I1 - *I2) + (*I4 - *I3));\n\t"
							"TI2 = (*I0 - C5QC*(*I1 + *I4)) + C5QB*(*R2 - *R3) - C5QD*(*R1 - *R4) + C5QA*((*I2 - *I1) + (*I3 - *I4));\n\t"
							"TI3 = (*I0 - C5QC*(*I1 + *I4)) - C5QB*(*R2 - *R3) + C5QD*(*R1 - *R4) + C5QA*((*I2 - *I1) + (*I3 - *I4));\n\t";
						}
					}
					else
					{
						if(cReg)
						{
							bflyStr +=
							"TR0 = (*R0).x + (*R1).x + (*R2).x + (*R3).x + (*R4).x;\n\t"
							"TR1 = ((*R0).x - C5QC*((*R2).x + (*R3).x)) - C5QB*((*R1).y - (*R4).y) - C5QD*((*R2).y - (*R3).y) + C5QA*(((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));\n\t"
							"TR4 = ((*R0).x - C5QC*((*R2).x + (*R3).x)) + C5QB*((*R1).y - (*R4).y) + C5QD*((*R2).y - (*R3).y) + C5QA*(((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));\n\t"
							"TR2 = ((*R0).x - C5QC*((*R1).x + (*R4).x)) + C5QB*((*R2).y - (*R3).y) - C5QD*((*R1).y - (*R4).y) + C5QA*(((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));\n\t"
							"TR3 = ((*R0).x - C5QC*((*R1).x + (*R4).x)) - C5QB*((*R2).y - (*R3).y) + C5QD*((*R1).y - (*R4).y) + C5QA*(((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*R0).y + (*R1).y + (*R2).y + (*R3).y + (*R4).y;\n\t"
							"TI1 = ((*R0).y - C5QC*((*R2).y + (*R3).y)) + C5QB*((*R1).x - (*R4).x) + C5QD*((*R2).x - (*R3).x) + C5QA*(((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));\n\t"
							"TI4 = ((*R0).y - C5QC*((*R2).y + (*R3).y)) - C5QB*((*R1).x - (*R4).x) - C5QD*((*R2).x - (*R3).x) + C5QA*(((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));\n\t"
							"TI2 = ((*R0).y - C5QC*((*R1).y + (*R4).y)) - C5QB*((*R2).x - (*R3).x) + C5QD*((*R1).x - (*R4).x) + C5QA*(((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));\n\t"
							"TI3 = ((*R0).y - C5QC*((*R1).y + (*R4).y)) + C5QB*((*R2).x - (*R3).x) - C5QD*((*R1).x - (*R4).x) + C5QA*(((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = *R0 + *R1 + *R2 + *R3 + *R4;\n\t"
							"TR1 = (*R0 - C5QC*(*R2 + *R3)) - C5QB*(*I1 - *I4) - C5QD*(*I2 - *I3) + C5QA*((*R1 - *R2) + (*R4 - *R3));\n\t"
							"TR4 = (*R0 - C5QC*(*R2 + *R3)) + C5QB*(*I1 - *I4) + C5QD*(*I2 - *I3) + C5QA*((*R1 - *R2) + (*R4 - *R3));\n\t"
							"TR2 = (*R0 - C5QC*(*R1 + *R4)) + C5QB*(*I2 - *I3) - C5QD*(*I1 - *I4) + C5QA*((*R2 - *R1) + (*R3 - *R4));\n\t"
							"TR3 = (*R0 - C5QC*(*R1 + *R4)) - C5QB*(*I2 - *I3) + C5QD*(*I1 - *I4) + C5QA*((*R2 - *R1) + (*R3 - *R4));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = *I0 + *I1 + *I2 + *I3 + *I4;\n\t"
							"TI1 = (*I0 - C5QC*(*I2 + *I3)) + C5QB*(*R1 - *R4) + C5QD*(*R2 - *R3) + C5QA*((*I1 - *I2) + (*I4 - *I3));\n\t"
							"TI4 = (*I0 - C5QC*(*I2 + *I3)) - C5QB*(*R1 - *R4) - C5QD*(*R2 - *R3) + C5QA*((*I1 - *I2) + (*I4 - *I3));\n\t"
							"TI2 = (*I0 - C5QC*(*I1 + *I4)) - C5QB*(*R2 - *R3) + C5QD*(*R1 - *R4) + C5QA*((*I2 - *I1) + (*I3 - *I4));\n\t"
							"TI3 = (*I0 - C5QC*(*I1 + *I4)) + C5QB*(*R2 - *R3) - C5QD*(*R1 - *R4) + C5QA*((*I2 - *I1) + (*I3 - *I4));\n\t";
						}
					}
				} break;
			case 6:
				{
					if(fwd)
					{
						if(cReg)
						{
							bflyStr +=
							"TR0 = (*R0).x + (*R2).x + (*R4).x;\n\t"
							"TR2 = ((*R0).x - C3QA*((*R2).x + (*R4).x)) + C3QB*((*R2).y - (*R4).y);\n\t"
							"TR4 = ((*R0).x - C3QA*((*R2).x + (*R4).x)) - C3QB*((*R2).y - (*R4).y);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*R0).y + (*R2).y + (*R4).y;\n\t"
							"TI2 = ((*R0).y - C3QA*((*R2).y + (*R4).y)) - C3QB*((*R2).x - (*R4).x);\n\t"
							"TI4 = ((*R0).y - C3QA*((*R2).y + (*R4).y)) + C3QB*((*R2).x - (*R4).x);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TR1 = (*R1).x + (*R3).x + (*R5).x;\n\t"
							"TR3 = ((*R1).x - C3QA*((*R3).x + (*R5).x)) + C3QB*((*R3).y - (*R5).y);\n\t"
							"TR5 = ((*R1).x - C3QA*((*R3).x + (*R5).x)) - C3QB*((*R3).y - (*R5).y);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI1 = (*R1).y + (*R3).y + (*R5).y;\n\t"
							"TI3 = ((*R1).y - C3QA*((*R3).y + (*R5).y)) - C3QB*((*R3).x - (*R5).x);\n\t"
							"TI5 = ((*R1).y - C3QA*((*R3).y + (*R5).y)) + C3QB*((*R3).x - (*R5).x);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0).x = TR0 + TR1;\n\t"
							"(*R1).x = TR2 + ( C3QA*TR3 + C3QB*TI3);\n\t"
							"(*R2).x = TR4 + (-C3QA*TR5 + C3QB*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0).y = TI0 + TI1;\n\t"
							"(*R1).y = TI2 + (-C3QB*TR3 + C3QA*TI3);\n\t"
							"(*R2).y = TI4 + (-C3QB*TR5 - C3QA*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R3).x = TR0 - TR1;\n\t"
							"(*R4).x = TR2 - ( C3QA*TR3 + C3QB*TI3);\n\t"
							"(*R5).x = TR4 - (-C3QA*TR5 + C3QB*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R3).y = TI0 - TI1;\n\t"
							"(*R4).y = TI2 - (-C3QB*TR3 + C3QA*TI3);\n\t"
							"(*R5).y = TI4 - (-C3QB*TR5 - C3QA*TI5);\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = *R0 + *R2 + *R4;\n\t"
							"TR2 = (*R0 - C3QA*(*R2 + *R4)) + C3QB*(*I2 - *I4);\n\t"
							"TR4 = (*R0 - C3QA*(*R2 + *R4)) - C3QB*(*I2 - *I4);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = *I0 + *I2 + *I4;\n\t"
							"TI2 = (*I0 - C3QA*(*I2 + *I4)) - C3QB*(*R2 - *R4);\n\t"
							"TI4 = (*I0 - C3QA*(*I2 + *I4)) + C3QB*(*R2 - *R4);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TR1 = *R1 + *R3 + *R5;\n\t"
							"TR3 = (*R1 - C3QA*(*R3 + *R5)) + C3QB*(*I3 - *I5);\n\t"
							"TR5 = (*R1 - C3QA*(*R3 + *R5)) - C3QB*(*I3 - *I5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI1 = *I1 + *I3 + *I5;\n\t"
							"TI3 = (*I1 - C3QA*(*I3 + *I5)) - C3QB*(*R3 - *R5);\n\t"
							"TI5 = (*I1 - C3QA*(*I3 + *I5)) + C3QB*(*R3 - *R5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0) = TR0 + TR1;\n\t"
							"(*R1) = TR2 + ( C3QA*TR3 + C3QB*TI3);\n\t"
							"(*R2) = TR4 + (-C3QA*TR5 + C3QB*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*I0) = TI0 + TI1;\n\t"
							"(*I1) = TI2 + (-C3QB*TR3 + C3QA*TI3);\n\t"
							"(*I2) = TI4 + (-C3QB*TR5 - C3QA*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R3) = TR0 - TR1;\n\t"
							"(*R4) = TR2 - ( C3QA*TR3 + C3QB*TI3);\n\t"
							"(*R5) = TR4 - (-C3QA*TR5 + C3QB*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*I3) = TI0 - TI1;\n\t"
							"(*I4) = TI2 - (-C3QB*TR3 + C3QA*TI3);\n\t"
							"(*I5) = TI4 - (-C3QB*TR5 - C3QA*TI5);\n\t";
						}
					}
					else
					{
						if(cReg)
						{
							bflyStr +=
							"TR0 = (*R0).x + (*R2).x + (*R4).x;\n\t"
							"TR2 = ((*R0).x - C3QA*((*R2).x + (*R4).x)) - C3QB*((*R2).y - (*R4).y);\n\t"
							"TR4 = ((*R0).x - C3QA*((*R2).x + (*R4).x)) + C3QB*((*R2).y - (*R4).y);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*R0).y + (*R2).y + (*R4).y;\n\t"
							"TI2 = ((*R0).y - C3QA*((*R2).y + (*R4).y)) + C3QB*((*R2).x - (*R4).x);\n\t"
							"TI4 = ((*R0).y - C3QA*((*R2).y + (*R4).y)) - C3QB*((*R2).x - (*R4).x);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TR1 = (*R1).x + (*R3).x + (*R5).x;\n\t"
							"TR3 = ((*R1).x - C3QA*((*R3).x + (*R5).x)) - C3QB*((*R3).y - (*R5).y);\n\t"
							"TR5 = ((*R1).x - C3QA*((*R3).x + (*R5).x)) + C3QB*((*R3).y - (*R5).y);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI1 = (*R1).y + (*R3).y + (*R5).y;\n\t"
							"TI3 = ((*R1).y - C3QA*((*R3).y + (*R5).y)) + C3QB*((*R3).x - (*R5).x);\n\t"
							"TI5 = ((*R1).y - C3QA*((*R3).y + (*R5).y)) - C3QB*((*R3).x - (*R5).x);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0).x = TR0 + TR1;\n\t"
							"(*R1).x = TR2 + ( C3QA*TR3 - C3QB*TI3);\n\t"
							"(*R2).x = TR4 + (-C3QA*TR5 - C3QB*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0).y = TI0 + TI1;\n\t"
							"(*R1).y = TI2 + ( C3QB*TR3 + C3QA*TI3);\n\t"
							"(*R2).y = TI4 + ( C3QB*TR5 - C3QA*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R3).x = TR0 - TR1;\n\t"
							"(*R4).x = TR2 - ( C3QA*TR3 - C3QB*TI3);\n\t"
							"(*R5).x = TR4 - (-C3QA*TR5 - C3QB*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R3).y = TI0 - TI1;\n\t"
							"(*R4).y = TI2 - ( C3QB*TR3 + C3QA*TI3);\n\t"
							"(*R5).y = TI4 - ( C3QB*TR5 - C3QA*TI5);\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = *R0 + *R2 + *R4;\n\t"
							"TR2 = (*R0 - C3QA*(*R2 + *R4)) - C3QB*(*I2 - *I4);\n\t"
							"TR4 = (*R0 - C3QA*(*R2 + *R4)) + C3QB*(*I2 - *I4);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = *I0 + *I2 + *I4;\n\t"
							"TI2 = (*I0 - C3QA*(*I2 + *I4)) + C3QB*(*R2 - *R4);\n\t"
							"TI4 = (*I0 - C3QA*(*I2 + *I4)) - C3QB*(*R2 - *R4);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TR1 = *R1 + *R3 + *R5;\n\t"
							"TR3 = (*R1 - C3QA*(*R3 + *R5)) - C3QB*(*I3 - *I5);\n\t"
							"TR5 = (*R1 - C3QA*(*R3 + *R5)) + C3QB*(*I3 - *I5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI1 = *I1 + *I3 + *I5;\n\t"
							"TI3 = (*I1 - C3QA*(*I3 + *I5)) + C3QB*(*R3 - *R5);\n\t"
							"TI5 = (*I1 - C3QA*(*I3 + *I5)) - C3QB*(*R3 - *R5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0) = TR0 + TR1;\n\t"
							"(*R1) = TR2 + ( C3QA*TR3 - C3QB*TI3);\n\t"
							"(*R2) = TR4 + (-C3QA*TR5 - C3QB*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*I0) = TI0 + TI1;\n\t"
							"(*I1) = TI2 + ( C3QB*TR3 + C3QA*TI3);\n\t"
							"(*I2) = TI4 + ( C3QB*TR5 - C3QA*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R3) = TR0 - TR1;\n\t"
							"(*R4) = TR2 - ( C3QA*TR3 - C3QB*TI3);\n\t"
							"(*R5) = TR4 - (-C3QA*TR5 - C3QB*TI5);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*I3) = TI0 - TI1;\n\t"
							"(*I4) = TI2 - ( C3QB*TR3 + C3QA*TI3);\n\t"
							"(*I5) = TI4 - ( C3QB*TR5 - C3QA*TI5);\n\t";
						}
					}
				} break;
			case 8:
				{
					if(fwd)
					{
						if(cReg)
						{
							bflyStr +=
							"(*R1) = (*R0) - (*R1);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R1);\n\t"
							"(*R3) = (*R2) - (*R3);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R3);\n\t"
							"(*R5) = (*R4) - (*R5);\n\t"
							"(*R4) = 2.0f * (*R4) - (*R5);\n\t"
							"(*R7) = (*R6) - (*R7);\n\t"
							"(*R6) = 2.0f * (*R6) - (*R7);\n\t"
							"\n\t"
							"(*R2) = (*R0) - (*R2);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R2);\n\t"
							"(*R3) = (*R1) + (fvect2)(-(*R3).y, (*R3).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R3);\n\t"
							"(*R6) = (*R4) - (*R6);\n\t"
							"(*R4) = 2.0f * (*R4) - (*R6);\n\t"
							"(*R7) = (*R5) + (fvect2)(-(*R7).y, (*R7).x);\n\t"
							"(*R5) = 2.0f * (*R5) - (*R7);\n\t"
							"\n\t"
							"(*R4) = (*R0) - (*R4);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R4);\n\t"
							"(*R5) = ((*R1) - C8Q * (*R5)) - C8Q * (fvect2)((*R5).y, -(*R5).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R5);\n\t"
							"(*R6) = (*R2) + (fvect2)(-(*R6).y, (*R6).x);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R6);\n\t"
							"(*R7) = ((*R3) + C8Q * (*R7)) - C8Q * (fvect2)((*R7).y, -(*R7).x);\n\t"
							"(*R3) = 2.0f * (*R3) - (*R7);\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = (*R0) + (*R4) + (*R2) + (*R6) +     (*R1)             +     (*R3)             +     (*R5)             +     (*R7)            ;\n\t"
							"TR1 = (*R0) - (*R4) + (*I2) - (*I6) + C8Q*(*R1) + C8Q*(*I1) - C8Q*(*R3) + C8Q*(*I3) - C8Q*(*R5) - C8Q*(*I5) + C8Q*(*R7) - C8Q*(*I7);\n\t"
							"TR2 = (*R0) + (*R4) - (*R2) - (*R6)             +     (*I1)             -     (*I3)             +     (*I5)             -     (*I7);\n\t"
							"TR3 = (*R0) - (*R4) - (*I2) + (*I6) - C8Q*(*R1) + C8Q*(*I1) + C8Q*(*R3) + C8Q*(*I3) + C8Q*(*R5) - C8Q*(*I5) - C8Q*(*R7) - C8Q*(*I7);\n\t"
							"TR4 = (*R0) + (*R4) + (*R2) + (*R6) -     (*R1)             -     (*R3)             -     (*R5)             -     (*R7)            ;\n\t"
							"TR5 = (*R0) - (*R4) + (*I2) - (*I6) - C8Q*(*R1) - C8Q*(*I1) + C8Q*(*R3) - C8Q*(*I3) + C8Q*(*R5) + C8Q*(*I5) - C8Q*(*R7) + C8Q*(*I7);\n\t"
							"TR6 = (*R0) + (*R4) - (*R2) - (*R6)             -    (*I1)              +     (*I3)             -     (*I5)             +     (*I7);\n\t"
							"TR7 = (*R0) - (*R4) - (*I2) + (*I6) + C8Q*(*R1) - C8Q*(*I1) - C8Q*(*R3) - C8Q*(*I3) - C8Q*(*R5) + C8Q*(*I5) + C8Q*(*R7) + C8Q*(*I7);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*I0) + (*I4) + (*I2) + (*I6)             +     (*I1)             +     (*I3)             +     (*I5)             +     (*I7);\n\t"
							"TI1 = (*I0) - (*I4) - (*R2) + (*R6) - C8Q*(*R1) + C8Q*(*I1) - C8Q*(*R3) - C8Q*(*I3) + C8Q*(*R5) - C8Q*(*I5) + C8Q*(*R7) + C8Q*(*I7);\n\t"
							"TI2 = (*I0) + (*I4) - (*I2) - (*I6) -     (*R1)             +     (*R3)             -     (*R5)             +     (*R7)            ;\n\t"
							"TI3 = (*I0) - (*I4) + (*R2) - (*R6) - C8Q*(*R1) - C8Q*(*I1) - C8Q*(*R3) + C8Q*(*I3) + C8Q*(*R5) + C8Q*(*I5) + C8Q*(*R7) - C8Q*(*I7);\n\t"
							"TI4 = (*I0) + (*I4) + (*I2) + (*I6)             -    (*I1)              -     (*I3)             -     (*I5)             -     (*I7);\n\t"
							"TI5 = (*I0) - (*I4) - (*R2) + (*R6) + C8Q*(*R1) - C8Q*(*I1) + C8Q*(*R3) + C8Q*(*I3) - C8Q*(*R5) + C8Q*(*I5) - C8Q*(*R7) - C8Q*(*I7);\n\t"
							"TI6 = (*I0) + (*I4) - (*I2) - (*I6) +     (*R1)             -     (*R3)             +     (*R5)             -     (*R7)            ;\n\t"
							"TI7 = (*I0) - (*I4) + (*R2) - (*R6) + C8Q*(*R1) + C8Q*(*I1) + C8Q*(*R3) - C8Q*(*I3) - C8Q*(*R5) - C8Q*(*I5) - C8Q*(*R7) + C8Q*(*I7);\n\t";
						}
					}
					else
					{
						if(cReg)
						{
							bflyStr +=
							"(*R1) = (*R0) - (*R1);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R1);\n\t"
							"(*R3) = (*R2) - (*R3);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R3);\n\t"
							"(*R5) = (*R4) - (*R5);\n\t"
							"(*R4) = 2.0f * (*R4) - (*R5);\n\t"
							"(*R7) = (*R6) - (*R7);\n\t"
							"(*R6) = 2.0f * (*R6) - (*R7);\n\t"
							"\n\t"
							"(*R2) = (*R0) - (*R2);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R2);\n\t"
							"(*R3) = (*R1) + (fvect2)((*R3).y, -(*R3).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R3);\n\t"
							"(*R6) = (*R4) - (*R6);\n\t"
							"(*R4) = 2.0f * (*R4) - (*R6);\n\t"
							"(*R7) = (*R5) + (fvect2)((*R7).y, -(*R7).x);\n\t"
							"(*R5) = 2.0f * (*R5) - (*R7);\n\t"
							"\n\t"
							"(*R4) = (*R0) - (*R4);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R4);\n\t"
							"(*R5) = ((*R1) - C8Q * (*R5)) + C8Q * (fvect2)((*R5).y, -(*R5).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R5);\n\t"
							"(*R6) = (*R2) + (fvect2)((*R6).y, -(*R6).x);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R6);\n\t"
							"(*R7) = ((*R3) + C8Q * (*R7)) + C8Q * (fvect2)((*R7).y, -(*R7).x);\n\t"
							"(*R3) = 2.0f * (*R3) - (*R7);\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = (*R0) + (*R4) + (*R2) + (*R6) +     (*R1)             +     (*R3)             +     (*R5)             +     (*R7)            ;\n\t"
							"TR1 = (*R0) - (*R4) - (*I2) + (*I6) + C8Q*(*R1) - C8Q*(*I1) - C8Q*(*R3) - C8Q*(*I3) - C8Q*(*R5) + C8Q*(*I5) + C8Q*(*R7) + C8Q*(*I7);\n\t"
							"TR2 = (*R0) + (*R4) - (*R2) - (*R6)             -     (*I1)             +     (*I3)             -     (*I5)             +     (*I7);\n\t"
							"TR3 = (*R0) - (*R4) + (*I2) - (*I6) - C8Q*(*R1) - C8Q*(*I1) + C8Q*(*R3) - C8Q*(*I3) + C8Q*(*R5) + C8Q*(*I5) - C8Q*(*R7) + C8Q*(*I7);\n\t"
							"TR4 = (*R0) + (*R4) + (*R2) + (*R6) -     (*R1)             -    (*R3)              -     (*R5)             -     (*R7)            ;\n\t"
							"TR5 = (*R0) - (*R4) - (*I2) + (*I6) - C8Q*(*R1) + C8Q*(*I1) + C8Q*(*R3) + C8Q*(*I3) + C8Q*(*R5) - C8Q*(*I5) - C8Q*(*R7) - C8Q*(*I7);\n\t"
							"TR6 = (*R0) + (*R4) - (*R2) - (*R6)             +     (*I1)             -     (*I3)             +     (*I5)             -     (*I7);\n\t"
							"TR7 = (*R0) - (*R4) + (*I2) - (*I6) + C8Q*(*R1) + C8Q*(*I1) - C8Q*(*R3) + C8Q*(*I3) - C8Q*(*R5) - C8Q*(*I5) + C8Q*(*R7) - C8Q*(*I7);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*I0) + (*I4) + (*I2) + (*I6)             +     (*I1)             +    (*I3)              +     (*I5)             +     (*I7);\n\t"
							"TI1 = (*I0) - (*I4) + (*R2) - (*R6) + C8Q*(*R1) + C8Q*(*I1) + C8Q*(*R3) - C8Q*(*I3) - C8Q*(*R5) - C8Q*(*I5) - C8Q*(*R7) + C8Q*(*I7);\n\t"
							"TI2 = (*I0) + (*I4) - (*I2) - (*I6) +     (*R1)             -     (*R3)             +     (*R5)             -     (*R7)            ;\n\t"
							"TI3 = (*I0) - (*I4) - (*R2) + (*R6) + C8Q*(*R1) - C8Q*(*I1) + C8Q*(*R3) + C8Q*(*I3) - C8Q*(*R5) + C8Q*(*I5) - C8Q*(*R7) - C8Q*(*I7);\n\t"
							"TI4 = (*I0) + (*I4) + (*I2) + (*I6)             -     (*I1)             -     (*I3)             -     (*I5)             -     (*I7);\n\t"
							"TI5 = (*I0) - (*I4) + (*R2) - (*R6) - C8Q*(*R1) - C8Q*(*I1) - C8Q*(*R3) + C8Q*(*I3) + C8Q*(*R5) + C8Q*(*I5) + C8Q*(*R7) - C8Q*(*I7);\n\t"
							"TI6 = (*I0) + (*I4) - (*I2) - (*I6) -     (*R1)             +     (*R3)             -     (*R5)             +     (*R7)            ;\n\t"
							"TI7 = (*I0) - (*I4) - (*R2) + (*R6) - C8Q*(*R1) + C8Q*(*I1) - C8Q*(*R3) - C8Q*(*I3) + C8Q*(*R5) - C8Q*(*I5) + C8Q*(*R7) + C8Q*(*I7);\n\t";
						}
					}
				} break;
			case 10:
				{
					if(fwd)
					{
						if(cReg)
						{
							bflyStr +=
							"TR0 = (*R0).x + (*R2).x + (*R4).x + (*R6).x + (*R8).x;\n\t"
							"TR2 = ((*R0).x - C5QC*((*R4).x + (*R6).x)) + C5QB*((*R2).y - (*R8).y) + C5QD*((*R4).y - (*R6).y) + C5QA*(((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));\n\t"
							"TR8 = ((*R0).x - C5QC*((*R4).x + (*R6).x)) - C5QB*((*R2).y - (*R8).y) - C5QD*((*R4).y - (*R6).y) + C5QA*(((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));\n\t"
							"TR4 = ((*R0).x - C5QC*((*R2).x + (*R8).x)) - C5QB*((*R4).y - (*R6).y) + C5QD*((*R2).y - (*R8).y) + C5QA*(((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));\n\t"
							"TR6 = ((*R0).x - C5QC*((*R2).x + (*R8).x)) + C5QB*((*R4).y - (*R6).y) - C5QD*((*R2).y - (*R8).y) + C5QA*(((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*R0).y + (*R2).y + (*R4).y + (*R6).y + (*R8).y;\n\t"
							"TI2 = ((*R0).y - C5QC*((*R4).y + (*R6).y)) - C5QB*((*R2).x - (*R8).x) - C5QD*((*R4).x - (*R6).x) + C5QA*(((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));\n\t"
							"TI8 = ((*R0).y - C5QC*((*R4).y + (*R6).y)) + C5QB*((*R2).x - (*R8).x) + C5QD*((*R4).x - (*R6).x) + C5QA*(((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));\n\t"
							"TI4 = ((*R0).y - C5QC*((*R2).y + (*R8).y)) + C5QB*((*R4).x - (*R6).x) - C5QD*((*R2).x - (*R8).x) + C5QA*(((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));\n\t"
							"TI6 = ((*R0).y - C5QC*((*R2).y + (*R8).y)) - C5QB*((*R4).x - (*R6).x) + C5QD*((*R2).x - (*R8).x) + C5QA*(((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TR1 = (*R1).x + (*R3).x + (*R5).x + (*R7).x + (*R9).x;\n\t"
							"TR3 = ((*R1).x - C5QC*((*R5).x + (*R7).x)) + C5QB*((*R3).y - (*R9).y) + C5QD*((*R5).y - (*R7).y) + C5QA*(((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));\n\t"
							"TR9 = ((*R1).x - C5QC*((*R5).x + (*R7).x)) - C5QB*((*R3).y - (*R9).y) - C5QD*((*R5).y - (*R7).y) + C5QA*(((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));\n\t"
							"TR5 = ((*R1).x - C5QC*((*R3).x + (*R9).x)) - C5QB*((*R5).y - (*R7).y) + C5QD*((*R3).y - (*R9).y) + C5QA*(((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));\n\t"
							"TR7 = ((*R1).x - C5QC*((*R3).x + (*R9).x)) + C5QB*((*R5).y - (*R7).y) - C5QD*((*R3).y - (*R9).y) + C5QA*(((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI1 = (*R1).y + (*R3).y + (*R5).y + (*R7).y + (*R9).y;\n\t"
							"TI3 = ((*R1).y - C5QC*((*R5).y + (*R7).y)) - C5QB*((*R3).x - (*R9).x) - C5QD*((*R5).x - (*R7).x) + C5QA*(((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));\n\t"
							"TI9 = ((*R1).y - C5QC*((*R5).y + (*R7).y)) + C5QB*((*R3).x - (*R9).x) + C5QD*((*R5).x - (*R7).x) + C5QA*(((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));\n\t"
							"TI5 = ((*R1).y - C5QC*((*R3).y + (*R9).y)) + C5QB*((*R5).x - (*R7).x) - C5QD*((*R3).x - (*R9).x) + C5QA*(((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));\n\t"
							"TI7 = ((*R1).y - C5QC*((*R3).y + (*R9).y)) - C5QB*((*R5).x - (*R7).x) + C5QD*((*R3).x - (*R9).x) + C5QA*(((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0).x = TR0 + TR1;\n\t"
							"(*R1).x = TR2 + ( C5QE*TR3 + C5QD*TI3);\n\t"
							"(*R2).x = TR4 + ( C5QA*TR5 + C5QB*TI5);\n\t"
							"(*R3).x = TR6 + (-C5QA*TR7 + C5QB*TI7);\n\t"
							"(*R4).x = TR8 + (-C5QE*TR9 + C5QD*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0).y = TI0 + TI1;\n\t"
							"(*R1).y = TI2 + (-C5QD*TR3 + C5QE*TI3);\n\t"
							"(*R2).y = TI4 + (-C5QB*TR5 + C5QA*TI5);\n\t"
							"(*R3).y = TI6 + (-C5QB*TR7 - C5QA*TI7);\n\t"
							"(*R4).y = TI8 + (-C5QD*TR9 - C5QE*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R5).x = TR0 - TR1;\n\t"
							"(*R6).x = TR2 - ( C5QE*TR3 + C5QD*TI3);\n\t"
							"(*R7).x = TR4 - ( C5QA*TR5 + C5QB*TI5);\n\t"
							"(*R8).x = TR6 - (-C5QA*TR7 + C5QB*TI7);\n\t"
							"(*R9).x = TR8 - (-C5QE*TR9 + C5QD*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R5).y = TI0 - TI1;\n\t"
							"(*R6).y = TI2 - (-C5QD*TR3 + C5QE*TI3);\n\t"
							"(*R7).y = TI4 - (-C5QB*TR5 + C5QA*TI5);\n\t"
							"(*R8).y = TI6 - (-C5QB*TR7 - C5QA*TI7);\n\t"
							"(*R9).y = TI8 - (-C5QD*TR9 - C5QE*TI9);\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = *R0 + *R2 + *R4 + *R6 + *R8;\n\t"
							"TR2 = (*R0 - C5QC*(*R4 + *R6)) + C5QB*(*I2 - *I8) + C5QD*(*I4 - *I6) + C5QA*((*R2 - *R4) + (*R8 - *R6));\n\t"
							"TR8 = (*R0 - C5QC*(*R4 + *R6)) - C5QB*(*I2 - *I8) - C5QD*(*I4 - *I6) + C5QA*((*R2 - *R4) + (*R8 - *R6));\n\t"
							"TR4 = (*R0 - C5QC*(*R2 + *R8)) - C5QB*(*I4 - *I6) + C5QD*(*I2 - *I8) + C5QA*((*R4 - *R2) + (*R6 - *R8));\n\t"
							"TR6 = (*R0 - C5QC*(*R2 + *R8)) + C5QB*(*I4 - *I6) - C5QD*(*I2 - *I8) + C5QA*((*R4 - *R2) + (*R6 - *R8));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = *I0 + *I2 + *I4 + *I6 + *I8;\n\t"
							"TI2 = (*I0 - C5QC*(*I4 + *I6)) - C5QB*(*R2 - *R8) - C5QD*(*R4 - *R6) + C5QA*((*I2 - *I4) + (*I8 - *I6));\n\t"
							"TI8 = (*I0 - C5QC*(*I4 + *I6)) + C5QB*(*R2 - *R8) + C5QD*(*R4 - *R6) + C5QA*((*I2 - *I4) + (*I8 - *I6));\n\t"
							"TI4 = (*I0 - C5QC*(*I2 + *I8)) + C5QB*(*R4 - *R6) - C5QD*(*R2 - *R8) + C5QA*((*I4 - *I2) + (*I6 - *I8));\n\t"
							"TI6 = (*I0 - C5QC*(*I2 + *I8)) - C5QB*(*R4 - *R6) + C5QD*(*R2 - *R8) + C5QA*((*I4 - *I2) + (*I6 - *I8));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TR1 = *R1 + *R3 + *R5 + *R7 + *R9;\n\t"
							"TR3 = (*R1 - C5QC*(*R5 + *R7)) + C5QB*(*I3 - *I9) + C5QD*(*I5 - *I7) + C5QA*((*R3 - *R5) + (*R9 - *R7));\n\t"
							"TR9 = (*R1 - C5QC*(*R5 + *R7)) - C5QB*(*I3 - *I9) - C5QD*(*I5 - *I7) + C5QA*((*R3 - *R5) + (*R9 - *R7));\n\t"
							"TR5 = (*R1 - C5QC*(*R3 + *R9)) - C5QB*(*I5 - *I7) + C5QD*(*I3 - *I9) + C5QA*((*R5 - *R3) + (*R7 - *R9));\n\t"
							"TR7 = (*R1 - C5QC*(*R3 + *R9)) + C5QB*(*I5 - *I7) - C5QD*(*I3 - *I9) + C5QA*((*R5 - *R3) + (*R7 - *R9));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI1 = *I1 + *I3 + *I5 + *I7 + *I9;\n\t"
							"TI3 = (*I1 - C5QC*(*I5 + *I7)) - C5QB*(*R3 - *R9) - C5QD*(*R5 - *R7) + C5QA*((*I3 - *I5) + (*I9 - *I7));\n\t"
							"TI9 = (*I1 - C5QC*(*I5 + *I7)) + C5QB*(*R3 - *R9) + C5QD*(*R5 - *R7) + C5QA*((*I3 - *I5) + (*I9 - *I7));\n\t"
							"TI5 = (*I1 - C5QC*(*I3 + *I9)) + C5QB*(*R5 - *R7) - C5QD*(*R3 - *R9) + C5QA*((*I5 - *I3) + (*I7 - *I9));\n\t"
							"TI7 = (*I1 - C5QC*(*I3 + *I9)) - C5QB*(*R5 - *R7) + C5QD*(*R3 - *R9) + C5QA*((*I5 - *I3) + (*I7 - *I9));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0) = TR0 + TR1;\n\t"
							"(*R1) = TR2 + ( C5QE*TR3 + C5QD*TI3);\n\t"
							"(*R2) = TR4 + ( C5QA*TR5 + C5QB*TI5);\n\t"
							"(*R3) = TR6 + (-C5QA*TR7 + C5QB*TI7);\n\t"
							"(*R4) = TR8 + (-C5QE*TR9 + C5QD*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*I0) = TI0 + TI1;\n\t"
							"(*I1) = TI2 + (-C5QD*TR3 + C5QE*TI3);\n\t"
							"(*I2) = TI4 + (-C5QB*TR5 + C5QA*TI5);\n\t"
							"(*I3) = TI6 + (-C5QB*TR7 - C5QA*TI7);\n\t"
							"(*I4) = TI8 + (-C5QD*TR9 - C5QE*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R5) = TR0 - TR1;\n\t"
							"(*R6) = TR2 - ( C5QE*TR3 + C5QD*TI3);\n\t"
							"(*R7) = TR4 - ( C5QA*TR5 + C5QB*TI5);\n\t"
							"(*R8) = TR6 - (-C5QA*TR7 + C5QB*TI7);\n\t"
							"(*R9) = TR8 - (-C5QE*TR9 + C5QD*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*I5) = TI0 - TI1;\n\t"
							"(*I6) = TI2 - (-C5QD*TR3 + C5QE*TI3);\n\t"
							"(*I7) = TI4 - (-C5QB*TR5 + C5QA*TI5);\n\t"
							"(*I8) = TI6 - (-C5QB*TR7 - C5QA*TI7);\n\t"
							"(*I9) = TI8 - (-C5QD*TR9 - C5QE*TI9);\n\t";
						}
					}
					else
					{
						if(cReg)
						{
							bflyStr +=
							"TR0 = (*R0).x + (*R2).x + (*R4).x + (*R6).x + (*R8).x;\n\t"
							"TR2 = ((*R0).x - C5QC*((*R4).x + (*R6).x)) - C5QB*((*R2).y - (*R8).y) - C5QD*((*R4).y - (*R6).y) + C5QA*(((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));\n\t"
							"TR8 = ((*R0).x - C5QC*((*R4).x + (*R6).x)) + C5QB*((*R2).y - (*R8).y) + C5QD*((*R4).y - (*R6).y) + C5QA*(((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));\n\t"
							"TR4 = ((*R0).x - C5QC*((*R2).x + (*R8).x)) + C5QB*((*R4).y - (*R6).y) - C5QD*((*R2).y - (*R8).y) + C5QA*(((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));\n\t"
							"TR6 = ((*R0).x - C5QC*((*R2).x + (*R8).x)) - C5QB*((*R4).y - (*R6).y) + C5QD*((*R2).y - (*R8).y) + C5QA*(((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = (*R0).y + (*R2).y + (*R4).y + (*R6).y + (*R8).y;\n\t"
							"TI2 = ((*R0).y - C5QC*((*R4).y + (*R6).y)) + C5QB*((*R2).x - (*R8).x) + C5QD*((*R4).x - (*R6).x) + C5QA*(((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));\n\t"
							"TI8 = ((*R0).y - C5QC*((*R4).y + (*R6).y)) - C5QB*((*R2).x - (*R8).x) - C5QD*((*R4).x - (*R6).x) + C5QA*(((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));\n\t"
							"TI4 = ((*R0).y - C5QC*((*R2).y + (*R8).y)) - C5QB*((*R4).x - (*R6).x) + C5QD*((*R2).x - (*R8).x) + C5QA*(((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));\n\t"
							"TI6 = ((*R0).y - C5QC*((*R2).y + (*R8).y)) + C5QB*((*R4).x - (*R6).x) - C5QD*((*R2).x - (*R8).x) + C5QA*(((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TR1 = (*R1).x + (*R3).x + (*R5).x + (*R7).x + (*R9).x;\n\t"
							"TR3 = ((*R1).x - C5QC*((*R5).x + (*R7).x)) - C5QB*((*R3).y - (*R9).y) - C5QD*((*R5).y - (*R7).y) + C5QA*(((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));\n\t"
							"TR9 = ((*R1).x - C5QC*((*R5).x + (*R7).x)) + C5QB*((*R3).y - (*R9).y) + C5QD*((*R5).y - (*R7).y) + C5QA*(((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));\n\t"
							"TR5 = ((*R1).x - C5QC*((*R3).x + (*R9).x)) + C5QB*((*R5).y - (*R7).y) - C5QD*((*R3).y - (*R9).y) + C5QA*(((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));\n\t"
							"TR7 = ((*R1).x - C5QC*((*R3).x + (*R9).x)) - C5QB*((*R5).y - (*R7).y) + C5QD*((*R3).y - (*R9).y) + C5QA*(((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI1 = (*R1).y + (*R3).y + (*R5).y + (*R7).y + (*R9).y;\n\t"
							"TI3 = ((*R1).y - C5QC*((*R5).y + (*R7).y)) + C5QB*((*R3).x - (*R9).x) + C5QD*((*R5).x - (*R7).x) + C5QA*(((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));\n\t"
							"TI9 = ((*R1).y - C5QC*((*R5).y + (*R7).y)) - C5QB*((*R3).x - (*R9).x) - C5QD*((*R5).x - (*R7).x) + C5QA*(((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));\n\t"
							"TI5 = ((*R1).y - C5QC*((*R3).y + (*R9).y)) - C5QB*((*R5).x - (*R7).x) + C5QD*((*R3).x - (*R9).x) + C5QA*(((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));\n\t"
							"TI7 = ((*R1).y - C5QC*((*R3).y + (*R9).y)) + C5QB*((*R5).x - (*R7).x) - C5QD*((*R3).x - (*R9).x) + C5QA*(((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0).x = TR0 + TR1;\n\t"
							"(*R1).x = TR2 + ( C5QE*TR3 - C5QD*TI3);\n\t"
							"(*R2).x = TR4 + ( C5QA*TR5 - C5QB*TI5);\n\t"
							"(*R3).x = TR6 + (-C5QA*TR7 - C5QB*TI7);\n\t"
							"(*R4).x = TR8 + (-C5QE*TR9 - C5QD*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0).y = TI0 + TI1;\n\t"
							"(*R1).y = TI2 + ( C5QD*TR3 + C5QE*TI3);\n\t"
							"(*R2).y = TI4 + ( C5QB*TR5 + C5QA*TI5);\n\t"
							"(*R3).y = TI6 + ( C5QB*TR7 - C5QA*TI7);\n\t"
							"(*R4).y = TI8 + ( C5QD*TR9 - C5QE*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R5).x = TR0 - TR1;\n\t"
							"(*R6).x = TR2 - ( C5QE*TR3 - C5QD*TI3);\n\t"
							"(*R7).x = TR4 - ( C5QA*TR5 - C5QB*TI5);\n\t"
							"(*R8).x = TR6 - (-C5QA*TR7 - C5QB*TI7);\n\t"
							"(*R9).x = TR8 - (-C5QE*TR9 - C5QD*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R5).y = TI0 - TI1;\n\t"
							"(*R6).y = TI2 - ( C5QD*TR3 + C5QE*TI3);\n\t"
							"(*R7).y = TI4 - ( C5QB*TR5 + C5QA*TI5);\n\t"
							"(*R8).y = TI6 - ( C5QB*TR7 - C5QA*TI7);\n\t"
							"(*R9).y = TI8 - ( C5QD*TR9 - C5QE*TI9);\n\t";
						}
						else
						{
							bflyStr +=
							"TR0 = *R0 + *R2 + *R4 + *R6 + *R8;\n\t"
							"TR2 = (*R0 - C5QC*(*R4 + *R6)) - C5QB*(*I2 - *I8) - C5QD*(*I4 - *I6) + C5QA*((*R2 - *R4) + (*R8 - *R6));\n\t"
							"TR8 = (*R0 - C5QC*(*R4 + *R6)) + C5QB*(*I2 - *I8) + C5QD*(*I4 - *I6) + C5QA*((*R2 - *R4) + (*R8 - *R6));\n\t"
							"TR4 = (*R0 - C5QC*(*R2 + *R8)) + C5QB*(*I4 - *I6) - C5QD*(*I2 - *I8) + C5QA*((*R4 - *R2) + (*R6 - *R8));\n\t"
							"TR6 = (*R0 - C5QC*(*R2 + *R8)) - C5QB*(*I4 - *I6) + C5QD*(*I2 - *I8) + C5QA*((*R4 - *R2) + (*R6 - *R8));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI0 = *I0 + *I2 + *I4 + *I6 + *I8;\n\t"
							"TI2 = (*I0 - C5QC*(*I4 + *I6)) + C5QB*(*R2 - *R8) + C5QD*(*R4 - *R6) + C5QA*((*I2 - *I4) + (*I8 - *I6));\n\t"
							"TI8 = (*I0 - C5QC*(*I4 + *I6)) - C5QB*(*R2 - *R8) - C5QD*(*R4 - *R6) + C5QA*((*I2 - *I4) + (*I8 - *I6));\n\t"
							"TI4 = (*I0 - C5QC*(*I2 + *I8)) - C5QB*(*R4 - *R6) + C5QD*(*R2 - *R8) + C5QA*((*I4 - *I2) + (*I6 - *I8));\n\t"
							"TI6 = (*I0 - C5QC*(*I2 + *I8)) + C5QB*(*R4 - *R6) - C5QD*(*R2 - *R8) + C5QA*((*I4 - *I2) + (*I6 - *I8));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TR1 = *R1 + *R3 + *R5 + *R7 + *R9;\n\t"
							"TR3 = (*R1 - C5QC*(*R5 + *R7)) - C5QB*(*I3 - *I9) - C5QD*(*I5 - *I7) + C5QA*((*R3 - *R5) + (*R9 - *R7));\n\t"
							"TR9 = (*R1 - C5QC*(*R5 + *R7)) + C5QB*(*I3 - *I9) + C5QD*(*I5 - *I7) + C5QA*((*R3 - *R5) + (*R9 - *R7));\n\t"
							"TR5 = (*R1 - C5QC*(*R3 + *R9)) + C5QB*(*I5 - *I7) - C5QD*(*I3 - *I9) + C5QA*((*R5 - *R3) + (*R7 - *R9));\n\t"
							"TR7 = (*R1 - C5QC*(*R3 + *R9)) - C5QB*(*I5 - *I7) + C5QD*(*I3 - *I9) + C5QA*((*R5 - *R3) + (*R7 - *R9));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"TI1 = *I1 + *I3 + *I5 + *I7 + *I9;\n\t"
							"TI3 = (*I1 - C5QC*(*I5 + *I7)) + C5QB*(*R3 - *R9) + C5QD*(*R5 - *R7) + C5QA*((*I3 - *I5) + (*I9 - *I7));\n\t"
							"TI9 = (*I1 - C5QC*(*I5 + *I7)) - C5QB*(*R3 - *R9) - C5QD*(*R5 - *R7) + C5QA*((*I3 - *I5) + (*I9 - *I7));\n\t"
							"TI5 = (*I1 - C5QC*(*I3 + *I9)) - C5QB*(*R5 - *R7) + C5QD*(*R3 - *R9) + C5QA*((*I5 - *I3) + (*I7 - *I9));\n\t"
							"TI7 = (*I1 - C5QC*(*I3 + *I9)) + C5QB*(*R5 - *R7) - C5QD*(*R3 - *R9) + C5QA*((*I5 - *I3) + (*I7 - *I9));\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R0) = TR0 + TR1;\n\t"
							"(*R1) = TR2 + ( C5QE*TR3 - C5QD*TI3);\n\t"
							"(*R2) = TR4 + ( C5QA*TR5 - C5QB*TI5);\n\t"
							"(*R3) = TR6 + (-C5QA*TR7 - C5QB*TI7);\n\t"
							"(*R4) = TR8 + (-C5QE*TR9 - C5QD*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*I0) = TI0 + TI1;\n\t"
							"(*I1) = TI2 + ( C5QD*TR3 + C5QE*TI3);\n\t"
							"(*I2) = TI4 + ( C5QB*TR5 + C5QA*TI5);\n\t"
							"(*I3) = TI6 + ( C5QB*TR7 - C5QA*TI7);\n\t"
							"(*I4) = TI8 + ( C5QD*TR9 - C5QE*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*R5) = TR0 - TR1;\n\t"
							"(*R6) = TR2 - ( C5QE*TR3 - C5QD*TI3);\n\t"
							"(*R7) = TR4 - ( C5QA*TR5 - C5QB*TI5);\n\t"
							"(*R8) = TR6 - (-C5QA*TR7 - C5QB*TI7);\n\t"
							"(*R9) = TR8 - (-C5QE*TR9 - C5QD*TI9);\n\t";

							bflyStr += "\n\t";

							bflyStr +=
							"(*I5) = TI0 - TI1;\n\t"
							"(*I6) = TI2 - ( C5QD*TR3 + C5QE*TI3);\n\t"
							"(*I7) = TI4 - ( C5QB*TR5 + C5QA*TI5);\n\t"
							"(*I8) = TI6 - ( C5QB*TR7 - C5QA*TI7);\n\t"
							"(*I9) = TI8 - ( C5QD*TR9 - C5QE*TI9);\n\t";
						}
					}
				} break;
			case 16:
				{
					if(fwd)
					{
						if(cReg)
						{
							bflyStr +=

							"(*R1) = (*R0) - (*R1);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R1);\n\t"
							"(*R3) = (*R2) - (*R3);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R3);\n\t"
							"(*R5) = (*R4) - (*R5);\n\t"
							"(*R4) = 2.0f * (*R4) - (*R5);\n\t"
							"(*R7) = (*R6) - (*R7);\n\t"
							"(*R6) = 2.0f * (*R6) - (*R7);\n\t"
							"(*R9) = (*R8) - (*R9);\n\t"
							"(*R8) = 2.0f * (*R8) - (*R9);\n\t"
							"(*R11) = (*R10) - (*R11);\n\t"
							"(*R10) = 2.0f * (*R10) - (*R11);\n\t"
							"(*R13) = (*R12) - (*R13);\n\t"
							"(*R12) = 2.0f * (*R12) - (*R13);\n\t"
							"(*R15) = (*R14) - (*R15);\n\t"
							"(*R14) = 2.0f * (*R14) - (*R15);\n\t"
							"\n\t"
							"(*R2) = (*R0) - (*R2);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R2);\n\t"
							"(*R3) = (*R1) + (fvect2)(-(*R3).y, (*R3).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R3);\n\t"
							"(*R6) = (*R4) - (*R6);\n\t"
							"(*R4) = 2.0f * (*R4) - (*R6);\n\t"
							"(*R7) = (*R5) + (fvect2)(-(*R7).y, (*R7).x);\n\t"
							"(*R5) = 2.0f * (*R5) - (*R7);\n\t"
							"(*R10) = (*R8) - (*R10);\n\t"
							"(*R8) = 2.0f * (*R8) - (*R10);\n\t"
							"(*R11) = (*R9) + (fvect2)(-(*R11).y, (*R11).x);\n\t"
							"(*R9) = 2.0f * (*R9) - (*R11);\n\t"
							"(*R14) = (*R12) - (*R14);\n\t"
							"(*R12) = 2.0f * (*R12) - (*R14);\n\t"
							"(*R15) = (*R13) + (fvect2)(-(*R15).y, (*R15).x);\n\t"
							"(*R13) = 2.0f * (*R13) - (*R15);\n\t"
							"\n\t"
							"(*R4) = (*R0) - (*R4);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R4);\n\t"
							"(*R5) = ((*R1) - C8Q * (*R5)) - C8Q * (fvect2)((*R5).y, -(*R5).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R5);\n\t"
							"(*R6) = (*R2) + (fvect2)(-(*R6).y, (*R6).x);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R6);\n\t"
							"(*R7) = ((*R3) + C8Q * (*R7)) - C8Q * (fvect2)((*R7).y, -(*R7).x);\n\t"
							"(*R3) = 2.0f * (*R3) - (*R7);\n\t"
							"(*R12) = (*R8) - (*R12);\n\t"
							"(*R8) = 2.0f * (*R8) - (*R12);\n\t"
							"(*R13) = ((*R9) - C8Q * (*R13)) - C8Q * (fvect2)((*R13).y, -(*R13).x);\n\t"
							"(*R9) = 2.0f * (*R9) - (*R13);\n\t"
							"(*R14) = (*R10) + (fvect2)(-(*R14).y, (*R14).x);\n\t"
							"(*R10) = 2.0f * (*R10) - (*R14);\n\t"
							"(*R15) = ((*R11) + C8Q * (*R15)) - C8Q * (fvect2)((*R15).y, -(*R15).x);\n\t"
							"(*R11) = 2.0f * (*R11) - (*R15);\n\t"
							"\n\t"
							"(*R8) = (*R0) - (*R8);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R8);\n\t"
							"(*R9) = ((*R1) - 0.92387953251128675612818318939679 * (*R9)) - 0.3826834323650897717284599840304 * (fvect2)((*R9).y, -(*R9).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R9);\n\t"
							"(*R10) = ((*R2) - C8Q * (*R10)) - C8Q * (fvect2)((*R10).y, -(*R10).x);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R10);\n\t"
							"(*R11) = ((*R3) - 0.3826834323650897717284599840304 * (*R11)) - 0.92387953251128675612818318939679 * (fvect2)((*R11).y, -(*R11).x);\n\t"
							"(*R3) = 2.0f * (*R3) - (*R11);\n\t"
							"(*R12) = (*R4) + (fvect2)(-(*R12).y, (*R12).x);\n\t"
							"(*R4) = 2.0f * (*R4) - (*R12);\n\t"
							"(*R13) = ((*R5) + 0.3826834323650897717284599840304 * (*R13)) - 0.92387953251128675612818318939679 * (fvect2)((*R13).y, -(*R13).x);\n\t"
							"(*R5) = 2.0f * (*R5) - (*R13);\n\t"
							"(*R14) = ((*R6) + C8Q * (*R14)) - C8Q * (fvect2)((*R14).y, -(*R14).x);\n\t"
							"(*R6) = 2.0f * (*R6) - (*R14);\n\t"
							"(*R15) = ((*R7) + 0.92387953251128675612818318939679 * (*R15)) - 0.3826834323650897717284599840304 * (fvect2)((*R15).y, -(*R15).x);\n\t"
							"(*R7) = 2.0f * (*R7) - (*R15);\n\t";

						}
						else
							assert(false);
					}
					else
					{
						if(cReg)
						{
							bflyStr +=

							"(*R1) = (*R0) - (*R1);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R1);\n\t"
							"(*R3) = (*R2) - (*R3);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R3);\n\t"
							"(*R5) = (*R4) - (*R5);\n\t"
							"(*R4) = 2.0f * (*R4) - (*R5);\n\t"
							"(*R7) = (*R6) - (*R7);\n\t"
							"(*R6) = 2.0f * (*R6) - (*R7);\n\t"
							"(*R9) = (*R8) - (*R9);\n\t"
							"(*R8) = 2.0f * (*R8) - (*R9);\n\t"
							"(*R11) = (*R10) - (*R11);\n\t"
							"(*R10) = 2.0f * (*R10) - (*R11);\n\t"
							"(*R13) = (*R12) - (*R13);\n\t"
							"(*R12) = 2.0f * (*R12) - (*R13);\n\t"
							"(*R15) = (*R14) - (*R15);\n\t"
							"(*R14) = 2.0f * (*R14) - (*R15);\n\t"
							"\n\t"
							"(*R2) = (*R0) - (*R2);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R2);\n\t"
							"(*R3) = (*R1) + (fvect2)((*R3).y, -(*R3).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R3);\n\t"
							"(*R6) = (*R4) - (*R6);\n\t"
							"(*R4) = 2.0f * (*R4) - (*R6);\n\t"
							"(*R7) = (*R5) + (fvect2)((*R7).y, -(*R7).x);\n\t"
							"(*R5) = 2.0f * (*R5) - (*R7);\n\t"
							"(*R10) = (*R8) - (*R10);\n\t"
							"(*R8) = 2.0f * (*R8) - (*R10);\n\t"
							"(*R11) = (*R9) + (fvect2)((*R11).y, -(*R11).x);\n\t"
							"(*R9) = 2.0f * (*R9) - (*R11);\n\t"
							"(*R14) = (*R12) - (*R14);\n\t"
							"(*R12) = 2.0f * (*R12) - (*R14);\n\t"
							"(*R15) = (*R13) + (fvect2)((*R15).y, -(*R15).x);\n\t"
							"(*R13) = 2.0f * (*R13) - (*R15);\n\t"
							"\n\t"
							"(*R4) = (*R0) - (*R4);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R4);\n\t"
							"(*R5) = ((*R1) - C8Q * (*R5)) + C8Q * (fvect2)((*R5).y, -(*R5).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R5);\n\t"
							"(*R6) = (*R2) + (fvect2)((*R6).y, -(*R6).x);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R6);\n\t"
							"(*R7) = ((*R3) + C8Q * (*R7)) + C8Q * (fvect2)((*R7).y, -(*R7).x);\n\t"
							"(*R3) = 2.0f * (*R3) - (*R7);\n\t"
							"(*R12) = (*R8) - (*R12);\n\t"
							"(*R8) = 2.0f * (*R8) - (*R12);\n\t"
							"(*R13) = ((*R9) - C8Q * (*R13)) + C8Q * (fvect2)((*R13).y, -(*R13).x);\n\t"
							"(*R9) = 2.0f * (*R9) - (*R13);\n\t"
							"(*R14) = (*R10) + (fvect2)((*R14).y, -(*R14).x);\n\t"
							"(*R10) = 2.0f * (*R10) - (*R14);\n\t"
							"(*R15) = ((*R11) + C8Q * (*R15)) + C8Q * (fvect2)((*R15).y, -(*R15).x);\n\t"
							"(*R11) = 2.0f * (*R11) - (*R15);\n\t"
 							"\n\t"
							"(*R8) = (*R0) - (*R8);\n\t"
							"(*R0) = 2.0f * (*R0) - (*R8);\n\t"
							"(*R9) = ((*R1) - 0.92387953251128675612818318939679 * (*R9)) + 0.3826834323650897717284599840304 * (fvect2)((*R9).y, -(*R9).x);\n\t"
							"(*R1) = 2.0f * (*R1) - (*R9);\n\t"
							"(*R10) = ((*R2) - C8Q * (*R10)) + C8Q * (fvect2)((*R10).y, -(*R10).x);\n\t"
							"(*R2) = 2.0f * (*R2) - (*R10);\n\t"
							"(*R11) = ((*R3) - 0.3826834323650897717284599840304 * (*R11)) + 0.92387953251128675612818318939679 * (fvect2)((*R11).y, -(*R11).x);\n\t"
							"(*R3) = 2.0f * (*R3) - (*R11);\n\t"
							"(*R12) = (*R4) + (fvect2)((*R12).y, -(*R12).x);\n\t"
							"(*R4) = 2.0f * (*R4) - (*R12);\n\t"
							"(*R13) = ((*R5) + 0.3826834323650897717284599840304 * (*R13)) + 0.92387953251128675612818318939679 * (fvect2)((*R13).y, -(*R13).x);\n\t"
							"(*R5) = 2.0f * (*R5) - (*R13);\n\t"
							"(*R14) = ((*R6) + C8Q * (*R14)) + C8Q * (fvect2)((*R14).y, -(*R14).x);\n\t"
							"(*R6) = 2.0f * (*R6) - (*R14);\n\t"
							"(*R15) = ((*R7) + 0.92387953251128675612818318939679 * (*R15)) + 0.3826834323650897717284599840304 * (fvect2)((*R15).y, -(*R15).x);\n\t"
							"(*R7) = 2.0f * (*R7) - (*R15);\n\t";

						}
						else
							assert(false);
					}
				} break;
			default:
				assert(false);
			}

			bflyStr += "\n\t";

			// Assign results
			if( (radix & (radix-1)) || (!cReg) )
			{
				if( (radix != 10) && (radix != 6) )
				{
				for(size_t i=0; i<radix;i++)
				{
					if(cReg)
					{
						bflyStr += "((*R"; bflyStr += SztToStr(i); bflyStr += ").x) = TR"; bflyStr += SztToStr(i); bflyStr += "; ";
						bflyStr += "((*R"; bflyStr += SztToStr(i); bflyStr += ").y) = TI"; bflyStr += SztToStr(i); bflyStr += ";\n\t";
					}
					else
					{
						bflyStr += "(*R"; bflyStr += SztToStr(i); bflyStr += ") = TR"; bflyStr += SztToStr(i); bflyStr += "; ";
						bflyStr += "(*I"; bflyStr += SztToStr(i); bflyStr += ") = TI"; bflyStr += SztToStr(i); bflyStr += ";\n\t";
					}
				}
				}
			}
			else
			{
				for(size_t i=0; i<radix;i++)
				{
					size_t j = BitReverse(i, radix);

					if(i < j)
					{
						bflyStr += "T = (*R"; bflyStr += SztToStr(i); bflyStr += "); (*R";
						bflyStr += SztToStr(i); bflyStr += ") = (*R"; bflyStr += SztToStr(j); bflyStr += "); (*R";
						bflyStr += SztToStr(j); bflyStr += ") = T;\n\t";
					}
				}
			}

			bflyStr += "\n}\n";
		}

	public:
		Butterfly(size_t radixVal, size_t countVal, bool fwdVal, bool cRegVal) : radix(radixVal), count(countVal), fwd(fwdVal), cReg(cRegVal) {}

		void GenerateButterfly(std::string &bflyStr) const
		{
			assert(count <= 4);
			if(count > 0)
				GenerateButterflyStr(bflyStr);
		}
    };

};

#endif

