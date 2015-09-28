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



	// Twiddle factors table for large N
	// used in 3-step algorithm
    class TwiddleTableLarge
    {
        size_t N; // length
		size_t X, Y;
		size_t tableSize;
		double *wc, *ws; // cosine, sine arrays

	public:
		TwiddleTableLarge(size_t length) : N(length)
		{
			X = size_t(1) << ARBITRARY::TWIDDLE_DEE;
			Y = DivRoundingUp<size_t> (CeilPo2(N), ARBITRARY::TWIDDLE_DEE);
			tableSize = X * Y;

			// Allocate memory for the tables
			wc = new double[tableSize];
			ws = new double[tableSize];
		}

		~TwiddleTableLarge()
		{
			// Free
			delete[] wc;
			delete[] ws;
		}

		template <Precision PR>
		void GenerateTwiddleTable(std::string &twStr)
		{
			const double TWO_PI = -6.283185307179586476925286766559;

			// Generate the table
			size_t nt = 0;
			double phi = TWO_PI / double (N);
			for (size_t iY = 0; iY < Y; ++iY)
			{
				size_t i = size_t(1) << (iY * ARBITRARY::TWIDDLE_DEE);
				for (size_t iX = 0; iX < X; ++iX)
				{
					size_t j = i * iX;

					double c = cos(phi * (double)j);
					double s = sin(phi * (double)j);

					//if (fabs(c) < 1.0E-12)	c = 0.0;
					//if (fabs(s) < 1.0E-12)	s = 0.0;

					wc[nt]   = c;
					ws[nt++] = s;
				}
			}

			std::string sfx = FloatSuffix<PR>();

			// Stringize the table
			std::stringstream ss;
			nt = 0;

			ss << "\n __constant ";
			ss << RegBaseType<PR>(2);
			ss << " " << TwTableLargeName();
			ss << "[" << Y << "][" << X << "] = {\n";
			for (size_t iY = 0; iY < Y; ++iY)
			{
				ss << "{ ";
				for (size_t iX = 0; iX < X; ++iX)
				{
					char cv[64], sv[64];
					sprintf(cv, "%036.34lf", wc[nt]);
					sprintf(sv, "%036.34lf", ws[nt++]);
					ss << "("; ss << RegBaseType<PR>(2); ss << ")(";
					ss << cv; ss << sfx; ss << ", ";
					ss << sv; ss << sfx; ss << ")";
					ss << ", ";
				}
				ss << " },\n";
			}
			ss << "};\n\n";


			// Twiddle calc function
			ss << "__attribute__((always_inline)) ";
			ss << RegBaseType<PR>(2);
			ss << "\n" << TwTableLargeFunc() << "(uint u)\n{\n";

			ss << "\t" "uint j = u & " << unsigned(X-1) << ";\n";
			ss << "\t" ; ss << RegBaseType<PR>(2); ss << " result = ";
			ss << TwTableLargeName();
			ss << "[0][j];\n";

			for (size_t iY = 1; iY < Y; ++iY)
			{
				std::string phasor = TwTableLargeName();
				phasor += "[";
				phasor += SztToStr(iY);
				phasor += "][j]";

				stringpair product = ComplexMul((RegBaseType<PR>(2)).c_str(), "result", phasor.c_str());

				ss << "\t" "u >>= " << unsigned (ARBITRARY::TWIDDLE_DEE) << ";\n";
				ss << "\t" "j = u & " << unsigned(X-1) << ";\n";
				ss << "\t" "result = " << product.first << "\n";
				ss << "\t" "\t" << product.second <<";\n";
			}
			ss << "\t" "return result;\n}\n\n";

			twStr += ss.str();
		}
    };




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
			case 7:
				{
					static const char *C7SFR = "\
					/*FFT7 Forward Real */ \n\
					\n\
						pr0 = *R1 + *R6; \n\
						pi0 = *I1 + *I6; \n\
						pr1 = *R1 - *R6; \n\
						pi1 = *I1 - *I6; \n\
						pr2 = *R2 + *R5; \n\
						pi2 = *I2 + *I5; \n\
						pr3 = *R2 - *R5; \n\
						pi3 = *I2 - *I5; \n\
						pr4 = *R4 + *R3; \n\
						pi4 = *I4 + *I3; \n\
						pr5 = *R4 - *R3; \n\
						pi5 = *I4 - *I3; \n\
					\n\
						pr6 = pr2 + pr0; \n\
						pi6 = pi2 + pi0; \n\
						qr4 = pr2 - pr0; \n\
						qi4 = pi2 - pi0; \n\
						qr2 = pr0 - pr4; \n\
						qi2 = pi0 - pi4; \n\
						qr3 = pr4 - pr2; \n\
						qi3 = pi4 - pi2; \n\
						pr7 = pr5 + pr3; \n\
						pi7 = pi5 + pi3; \n\
						qr7 = pr5 - pr3; \n\
						qi7 = pi5 - pi3; \n\
						qr6 = pr1 - pr5; \n\
						qi6 = pi1 - pi5; \n\
						qr8 = pr3 - pr1; \n\
						qi8 = pi3 - pi1; \n\
						qr1 = pr6 + pr4; \n\
						qi1 = pi6 + pi4; \n\
						qr5 = pr7 + pr1; \n\
						qi5 = pi7 + pi1; \n\
						qr0 = *R0 + qr1; \n\
						qi0 = *I0 + qi1; \n\
					\n\
						qr1 *= C7Q1; \n\
						qi1 *= C7Q1; \n\
						qr2 *= C7Q2; \n\
						qi2 *= C7Q2; \n\
						qr3 *= C7Q3; \n\
						qi3 *= C7Q3; \n\
						qr4 *= C7Q4; \n\
						qi4 *= C7Q4; \n\
					\n\
						qr5 *= (C7Q5); \n\
						qi5 *= (C7Q5); \n\
						qr6 *= (C7Q6); \n\
						qi6 *= (C7Q6); \n\
						qr7 *= (C7Q7); \n\
						qi7 *= (C7Q7); \n\
						qr8 *= (C7Q8); \n\
						qi8 *= (C7Q8); \n\
					\n\
						pr0 =  qr0 + qr1; \n\
						pi0 =  qi0 + qi1; \n\
						pr1 =  qr2 + qr3; \n\
						pi1 =  qi2 + qi3; \n\
						pr2 =  qr4 - qr3; \n\
						pi2 =  qi4 - qi3; \n\
						pr3 = -qr2 - qr4; \n\
						pi3 = -qi2 - qi4; \n\
						pr4 =  qr6 + qr7; \n\
						pi4 =  qi6 + qi7; \n\
						pr5 =  qr8 - qr7; \n\
						pi5 =  qi8 - qi7; \n\
						pr6 = -qr8 - qr6; \n\
						pi6 = -qi8 - qi6; \n\
						pr7 =  pr0 + pr1; \n\
						pi7 =  pi0 + pi1; \n\
						pr8 =  pr0 + pr2; \n\
						pi8 =  pi0 + pi2; \n\
						pr9 =  pr0 + pr3; \n\
						pi9 =  pi0 + pi3; \n\
						qr6 =  pr4 + qr5; \n\
						qi6 =  pi4 + qi5; \n\
						qr7 =  pr5 + qr5; \n\
						qi7 =  pi5 + qi5; \n\
						qr8 =  pr6 + qr5; \n\
						qi8 =  pi6 + qi5; \n\
					\n\
						TR0 = qr0; TI0 = qi0; \n\
						TR1 = pr7 + qi6; \n\
						TI1 = pi7 - qr6; \n\
						TR2 = pr9 + qi8; \n\
						TI2 = pi9 - qr8; \n\
						TR3 = pr8 - qi7; \n\
						TI3 = pi8 + qr7; \n\
						TR4 = pr8 + qi7; \n\
						TI4 = pi8 - qr7; \n\
						TR5 = pr9 - qi8; \n\
						TI5 = pi9 + qr8; \n\
						TR6 = pr7 - qi6; \n\
						TI6 = pi7 + qr6; \n\
					";

					static const char *C7SBR = "\
					/*FFT7 Backward Real */ \n\
					\n\
						pr0 = *R1 + *R6; \n\
						pi0 = *I1 + *I6; \n\
						pr1 = *R1 - *R6; \n\
						pi1 = *I1 - *I6; \n\
						pr2 = *R2 + *R5; \n\
						pi2 = *I2 + *I5; \n\
						pr3 = *R2 - *R5; \n\
						pi3 = *I2 - *I5; \n\
						pr4 = *R4 + *R3; \n\
						pi4 = *I4 + *I3; \n\
						pr5 = *R4 - *R3; \n\
						pi5 = *I4 - *I3; \n\
					\n\
						pr6 = pr2 + pr0; \n\
						pi6 = pi2 + pi0; \n\
						qr4 = pr2 - pr0; \n\
						qi4 = pi2 - pi0; \n\
						qr2 = pr0 - pr4; \n\
						qi2 = pi0 - pi4; \n\
						qr3 = pr4 - pr2; \n\
						qi3 = pi4 - pi2; \n\
						pr7 = pr5 + pr3; \n\
						pi7 = pi5 + pi3; \n\
						qr7 = pr5 - pr3; \n\
						qi7 = pi5 - pi3; \n\
						qr6 = pr1 - pr5; \n\
						qi6 = pi1 - pi5; \n\
						qr8 = pr3 - pr1; \n\
						qi8 = pi3 - pi1; \n\
						qr1 = pr6 + pr4; \n\
						qi1 = pi6 + pi4; \n\
						qr5 = pr7 + pr1; \n\
						qi5 = pi7 + pi1; \n\
						qr0 = *R0 + qr1; \n\
						qi0 = *I0 + qi1; \n\
					\n\
						qr1 *= C7Q1; \n\
						qi1 *= C7Q1; \n\
						qr2 *= C7Q2; \n\
						qi2 *= C7Q2; \n\
						qr3 *= C7Q3; \n\
						qi3 *= C7Q3; \n\
						qr4 *= C7Q4; \n\
						qi4 *= C7Q4; \n\
					\n\
						qr5 *= -(C7Q5); \n\
						qi5 *= -(C7Q5); \n\
						qr6 *= -(C7Q6); \n\
						qi6 *= -(C7Q6); \n\
						qr7 *= -(C7Q7); \n\
						qi7 *= -(C7Q7); \n\
						qr8 *= -(C7Q8); \n\
						qi8 *= -(C7Q8); \n\
					\n\
						pr0 =  qr0 + qr1; \n\
						pi0 =  qi0 + qi1; \n\
						pr1 =  qr2 + qr3; \n\
						pi1 =  qi2 + qi3; \n\
						pr2 =  qr4 - qr3; \n\
						pi2 =  qi4 - qi3; \n\
						pr3 = -qr2 - qr4; \n\
						pi3 = -qi2 - qi4; \n\
						pr4 =  qr6 + qr7; \n\
						pi4 =  qi6 + qi7; \n\
						pr5 =  qr8 - qr7; \n\
						pi5 =  qi8 - qi7; \n\
						pr6 = -qr8 - qr6; \n\
						pi6 = -qi8 - qi6; \n\
						pr7 =  pr0 + pr1; \n\
						pi7 =  pi0 + pi1; \n\
						pr8 =  pr0 + pr2; \n\
						pi8 =  pi0 + pi2; \n\
						pr9 =  pr0 + pr3; \n\
						pi9 =  pi0 + pi3; \n\
						qr6 =  pr4 + qr5; \n\
						qi6 =  pi4 + qi5; \n\
						qr7 =  pr5 + qr5; \n\
						qi7 =  pi5 + qi5; \n\
						qr8 =  pr6 + qr5; \n\
						qi8 =  pi6 + qi5; \n\
					\n\
						TR0 = qr0; TI0 = qi0; \n\
						TR1 = pr7 + qi6; \n\
						TI1 = pi7 - qr6; \n\
						TR2 = pr9 + qi8; \n\
						TI2 = pi9 - qr8; \n\
						TR3 = pr8 - qi7; \n\
						TI3 = pi8 + qr7; \n\
						TR4 = pr8 + qi7; \n\
						TI4 = pi8 - qr7; \n\
						TR5 = pr9 - qi8; \n\
						TI5 = pi9 + qr8; \n\
						TR6 = pr7 - qi6; \n\
						TI6 = pi7 + qr6; \n\
					";

					static const char *C7SFC = "\
					/*FFT7 Forward Complex */ \n\
					\n\
						p0 = *R1 + *R6; \n\
						p1 = *R1 - *R6; \n\
						p2 = *R2 + *R5; \n\
						p3 = *R2 - *R5; \n\
						p4 = *R4 + *R3; \n\
						p5 = *R4 - *R3; \n\
					\n\
						p6 = p2 + p0; \n\
						q4 = p2 - p0; \n\
						q2 = p0 - p4; \n\
						q3 = p4 - p2; \n\
						p7 = p5 + p3; \n\
						q7 = p5 - p3; \n\
						q6 = p1 - p5; \n\
						q8 = p3 - p1; \n\
						q1 = p6 + p4; \n\
						q5 = p7 + p1; \n\
						q0 = *R0 + q1; \n\
					\n\
						q1 *= C7Q1; \n\
						q2 *= C7Q2; \n\
						q3 *= C7Q3; \n\
						q4 *= C7Q4; \n\
					\n\
						q5 *= (C7Q5); \n\
						q6 *= (C7Q6); \n\
						q7 *= (C7Q7); \n\
						q8 *= (C7Q8); \n\
					\n\
						p0 = q0 + q1; \n\
						p1 = q2 + q3; \n\
						p2 = q4 - q3; \n\
						p3 = -q2 - q4; \n\
						p4 = q6 + q7; \n\
						p5 = q8 - q7; \n\
						p6 = -q8 - q6; \n\
						p7 = p0 + p1; \n\
						p8 = p0 + p2; \n\
						p9 = p0 + p3; \n\
						q6 = p4 + q5; \n\
						q7 = p5 + q5; \n\
						q8 = p6 + q5; \n\
					\n\
						*R0 = q0; \n\
						(*R1).x = p7.x + q6.y; \n\
						(*R1).y = p7.y - q6.x; \n\
						(*R2).x = p9.x + q8.y; \n\
						(*R2).y = p9.y - q8.x; \n\
						(*R3).x = p8.x - q7.y; \n\
						(*R3).y = p8.y + q7.x; \n\
						(*R4).x = p8.x + q7.y; \n\
						(*R4).y = p8.y - q7.x; \n\
						(*R5).x = p9.x - q8.y; \n\
						(*R5).y = p9.y + q8.x; \n\
						(*R6).x = p7.x - q6.y; \n\
						(*R6).y = p7.y + q6.x; \n\
					";

					static const char *C7SBC = "\
					/*FFT7 Backward Complex */ \n\
					\n\
						p0 = *R1 + *R6; \n\
						p1 = *R1 - *R6; \n\
						p2 = *R2 + *R5; \n\
						p3 = *R2 - *R5; \n\
						p4 = *R4 + *R3; \n\
						p5 = *R4 - *R3; \n\
					\n\
						p6 = p2 + p0; \n\
						q4 = p2 - p0; \n\
						q2 = p0 - p4; \n\
						q3 = p4 - p2; \n\
						p7 = p5 + p3; \n\
						q7 = p5 - p3; \n\
						q6 = p1 - p5; \n\
						q8 = p3 - p1; \n\
						q1 = p6 + p4; \n\
						q5 = p7 + p1; \n\
						q0 = *R0 + q1; \n\
					\n\
						q1 *= C7Q1; \n\
						q2 *= C7Q2; \n\
						q3 *= C7Q3; \n\
						q4 *= C7Q4; \n\
					\n\
						q5 *= -(C7Q5); \n\
						q6 *= -(C7Q6); \n\
						q7 *= -(C7Q7); \n\
						q8 *= -(C7Q8); \n\
					\n\
						p0 = q0 + q1; \n\
						p1 = q2 + q3; \n\
						p2 = q4 - q3; \n\
						p3 = -q2 - q4; \n\
						p4 = q6 + q7; \n\
						p5 = q8 - q7; \n\
						p6 = -q8 - q6; \n\
						p7 = p0 + p1; \n\
						p8 = p0 + p2; \n\
						p9 = p0 + p3; \n\
						q6 = p4 + q5; \n\
						q7 = p5 + q5; \n\
						q8 = p6 + q5; \n\
					\n\
						*R0 = q0; \n\
						(*R1).x = p7.x + q6.y; \n\
						(*R1).y = p7.y - q6.x; \n\
						(*R2).x = p9.x + q8.y; \n\
						(*R2).y = p9.y - q8.x; \n\
						(*R3).x = p8.x - q7.y; \n\
						(*R3).y = p8.y + q7.x; \n\
						(*R4).x = p8.x + q7.y; \n\
						(*R4).y = p8.y - q7.x; \n\
						(*R5).x = p9.x - q8.y; \n\
						(*R5).y = p9.y + q8.x; \n\
						(*R6).x = p7.x - q6.y; \n\
						(*R6).y = p7.y + q6.x; \n\
					";



					if (!cReg) {
						for (size_t i = 0; i < 10; i++)
							bflyStr += regType + " pr" + SztToStr(i) + ", pi" + SztToStr(i) + ";\n\t";
						for (size_t i = 0; i < 9; i++)
							bflyStr += regType + " qr" + SztToStr(i) + ", qi" + SztToStr(i) + ";\n\t";

						if (fwd)
							bflyStr += C7SFR;
						else
							bflyStr += C7SBR;
					} else {
						for (size_t i = 0; i < 10; i++)
							bflyStr += regType + " p" + SztToStr(i) + ";\n\t";
						for (size_t i = 0; i < 9; i++)
							bflyStr += regType + " q" + SztToStr(i) + ";\n\t";
						if (fwd)
							bflyStr += C7SFC;
						else
							bflyStr += C7SBC;
					}
				}
				break;

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
						if (radix !=7) 
						{
						bflyStr += "((*R"; bflyStr += SztToStr(i); bflyStr += ").x) = TR"; bflyStr += SztToStr(i); bflyStr += "; ";
						bflyStr += "((*R"; bflyStr += SztToStr(i); bflyStr += ").y) = TI"; bflyStr += SztToStr(i); bflyStr += ";\n\t";
						}
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

