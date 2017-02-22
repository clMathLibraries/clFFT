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
#if !defined( AMD_CLFFT_generator_twiddle_H )
#define AMD_CLFFT_generator_twiddle_H

#include "stdafx.h"
#include <math.h>
#include "generator.stockham.h"
#include <list>



namespace StockhamGenerator
{
	// Twiddle factors table
    class TwiddleTable
    {
        size_t N; // length
		double *wc, *ws; // cosine, sine arrays

	public:
		TwiddleTable(size_t length) : N(length)
		{
			// Allocate memory for the tables
			// We compute twiddle factors in double precision for both P_SINGLE and P_DOUBLE
			wc = new double[N];
			ws = new double[N];
		}

		~TwiddleTable()
		{
			// Free
			delete[] wc;
			delete[] ws;
		}

		template <Precision PR>
		void GenerateTwiddleTable(const std::vector<size_t> &radices, std::string &twStr)
		{
			const double TWO_PI = -6.283185307179586476925286766559;

			// Make sure the radices vector sums up to N
			size_t sz = 1;
			for(std::vector<size_t>::const_iterator i = radices.begin();
				i != radices.end(); i++)
			{
				sz *= (*i);
			}
			assert(sz == N);

			// Generate the table
			size_t L = 1;
			size_t nt = 0;
			for(std::vector<size_t>::const_iterator i = radices.begin();
				i != radices.end(); i++)
			{
				size_t radix = *i;

				L *= radix;

				// Twiddle factors
				for(size_t k=0; k<(L/radix); k++)
				{
					double theta = TWO_PI * ((double)k)/((double)L);

					for(size_t j=1; j<radix; j++)
					{
						double c = cos(((double)j) * theta);
						double s = sin(((double)j) * theta);

						//if (fabs(c) < 1.0E-12)	c = 0.0;
						//if (fabs(s) < 1.0E-12)	s = 0.0;

						wc[nt]   = c;
						ws[nt++] = s;
					}
				}
			}

			std::string sfx = FloatSuffix<PR>();

			// Stringize the table
			std::stringstream ss;
			ss.imbue(std::locale("C"));
			ss.precision(34);
			for(size_t i = 0; i < (N-1); i++)
			{
				ss << "("; ss << RegBaseType<PR>(2); ss << ")(";
				ss << std::scientific << wc[i] << sfx << ", ";
				ss << std::scientific << ws[i] << sfx << "),\n";
			}
			twStr += ss.str();
		}
    };

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
            Y = DivRoundingUp<size_t>(CeilPo2(N), ARBITRARY::TWIDDLE_DEE);
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
            double phi = TWO_PI / double(N);
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

                    wc[nt] = c;
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
            ss << "\n" << TwTableLargeFunc() << "(size_t u)\n{\n";

            ss << "\t" "size_t j = u & " << unsigned(X - 1) << ";\n";
            ss << "\t"; ss << RegBaseType<PR>(2); ss << " result = ";
            ss << TwTableLargeName();
            ss << "[0][j];\n";

            for (size_t iY = 1; iY < Y; ++iY)
            {
                std::string phasor = TwTableLargeName();
                phasor += "[";
                phasor += SztToStr(iY);
                phasor += "][j]";

                stringpair product = ComplexMul((RegBaseType<PR>(2)).c_str(), "result", phasor.c_str());

                ss << "\t" "u >>= " << unsigned(ARBITRARY::TWIDDLE_DEE) << ";\n";
                ss << "\t" "j = u & " << unsigned(X - 1) << ";\n";
                ss << "\t" "result = " << product.first << "\n";
                ss << "\t" "\t" << product.second << ";\n";
            }
            ss << "\t" "return result;\n}\n\n";

            twStr += ss.str();
        }
    };

};

#endif