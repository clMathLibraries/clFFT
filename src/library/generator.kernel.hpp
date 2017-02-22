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
#if !defined( AMD_CLFFT_generator_kernel_H )
#define AMD_CLFFT_generator_kernel_H
#include <stdio.h>
#include "private.h"
#include "repo.h"
#include "plan.h"
#include "generator.stockham.h"
#include "generator.butterfly.hpp"
#include "generator.pass.hpp"


 // FFT Stockham Autosort Method
 //
 //   Each pass does one digit reverse in essence. Hence by the time all passes are done, complete
 //   digit reversal is done and output FFT is in correct order. Intermediate FFTs are stored in natural order,
 //   which is not the case with basic Cooley-Tukey algorithm. Natural order in intermediate data makes it
 //   convenient for stitching together passes with different radices.
 //
 //  Basic FFT algorithm:
 //
 //        Pass loop
 //        {
 //            Outer loop
 //            {
 //                Inner loop
 //                {
 //                }
 //            }
 //        }
 //
 //  The sweeps of the outer and inner loop resemble matrix indexing, this matrix changes shape with every pass as noted below
 //
 //   FFT pass diagram (radix 2)
 //
 //                k            k+R                                    k
 //            * * * * * * * * * * * * * * * *                     * * * * * * * *
 //            *   |             |           *                     *   |         *
 //            *   |             |           *                     *   |         *
 //            *   |             |           * LS        -->       *   |         *
 //            *   |             |           *                     *   |         *
 //            *   |             |           *                     *   |         *
 //            * * * * * * * * * * * * * * * *                     *   |         *
 //                         RS                                     *   |         * L
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                * * * * * * * *
 //                                                                       R
 //
 //
 //    With every pass, the matrix doubles in height and halves in length
 //
 //
 //  N = 2^T = Length of FFT
 //  q = pass loop index
 //  k = outer loop index = (0 ... R-1)
 //  j = inner loop index = (0 ... LS-1)
 //
 //  Tables shows how values change as we go through the passes
 //
 //    q | LS   |  R   |  L  | RS
 //   ___|______|______|_____|___
 //    0 |  1   | N/2  |  2  | N
 //    1 |  2   | N/4  |  4  | N/2
 //    2 |  4   | N/8  |  8  | N/4
 //    . |  .   | .    |  .  | .
 //  T-1 |  N/2 | 1    |  N  | 2
 //
 //
 //   Data Read Order
 //     Radix 2: k*LS + j, (k+R)*LS + j
 //     Radix 3: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j
 //     Radix 4: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j, (k+3R)*LS + j
 //     Radix 5: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j, (k+3R)*LS + j, (k+4R)*LS + j
 //
 //   Data Write Order
 //       Radix 2: k*L + j, k*L + j + LS
 //       Radix 3: k*L + j, k*L + j + LS, k*L + j + 2*LS
 //       Radix 4: k*L + j, k*L + j + LS, k*L + j + 2*LS, k*L + j + 3*LS
 //       Radix 5: k*L + j, k*L + j + LS, k*L + j + 2*LS, k*L + j + 3*LS, k*L + j + 4*LS
 //




namespace StockhamGenerator
{

    //Table used by class KernelCoreSpecs
#define RADIX_TABLE_COMMON 	{     2048,           256,             1,         4,     8, 8, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{      512,            64,             1,         3,     8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{      256,            64,             1,         4,     4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{       64,            64,             4,         3,     4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{       32,            64,            16,         2,     8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{       16,            64,            16,         2,     4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{        4,            64,            32,         2,     2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },	\
							{        2,            64,            64,         1,     2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

    template <Precision PR>
    class KernelCoreSpecs
    {
        struct SpecRecord
        {
            size_t length;
            size_t workGroupSize;
            size_t numTransforms;
            size_t numPasses;
            size_t radices[12]; // Setting upper limit of number of passes to 12
        };

        typedef typename std::map<size_t, SpecRecord> SpecTable;
        SpecTable specTable;

    public:
        KernelCoreSpecs()
        {
            switch (PR)
            {
            case P_SINGLE:
            {
                SpecRecord specRecord[] = {

                    RADIX_TABLE_COMMON

                    //  Length, WorkGroupSize, NumTransforms, NumPasses,  Radices
                { 4096,           256,             1,         4,     8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 1024,           128,             1,         4,     8, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 128,            64,             4,         3,     8, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 8,            64,            32,         2,     4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

                };

                size_t tableLength = sizeof(specRecord) / sizeof(specRecord[0]);
                for (size_t i = 0; i<tableLength; i++) specTable[specRecord[i].length] = specRecord[i];

            } break;

            case P_DOUBLE:
            {
                SpecRecord specRecord[] = {

                    RADIX_TABLE_COMMON

                    //  Length, WorkGroupSize, NumTransforms, NumPasses,  Radices
                { 1024,           128,             1,         4,     8, 8, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 128,            64,             4,         3,     8, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 8,            64,            16,         3,     2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

                };

                size_t tableLength = sizeof(specRecord) / sizeof(specRecord[0]);
                for (size_t i = 0; i<tableLength; i++) specTable[specRecord[i].length] = specRecord[i];
            } break;

            default:
                assert(false);
            }
        }

        void GetRadices(size_t length, size_t &numPasses, const size_t * &pRadices) const
        {
            pRadices = NULL;
            numPasses = 0;

            typename SpecTable::const_iterator it = specTable.find(length);
            if (it != specTable.end())
            {
                pRadices = it->second.radices;
                numPasses = it->second.numPasses;
            }
        }

        void GetWGSAndNT(size_t length, size_t &workGroupSize, size_t &numTransforms) const
        {
            workGroupSize = 0;
            numTransforms = 0;

            typename SpecTable::const_iterator it = specTable.find(length);
            if (it != specTable.end())
            {
                workGroupSize = it->second.workGroupSize;
                numTransforms = it->second.numTransforms;
            }
        }
    };


    // FFT kernel generator
    // Kernel calls butterfly and pass
    template <Precision PR>
    class Kernel
    {
        size_t length;							// Length of FFT
        size_t workGroupSize;					// Work group size
        size_t cnPerWI;							// complex numbers per work-item

        size_t numTrans;						// Number of transforms per work-group
        size_t workGroupSizePerTrans;			// Work group subdivision per transform
        size_t numPasses;						// Number of FFT passes
        std::vector<size_t> radices;			// Base radix at each pass
        std::vector<Pass<PR> > passes;			// Array of pass objects

        bool halfLds;							// LDS used to store one component (either real or imaginary) at a time
                                                // for passing intermediate data between the passes, if this is set
                                                // then each pass-function should accept same set of registers

        bool linearRegs;						// scalar registers

                                                // Future optimization ideas
                                                // bool limitRegs;							// TODO: Incrementally write to LDS, thereby using same set of registers for more than 1 butterflies
                                                // bool combineReadTwMul;					// TODO: Combine reading into registers and Twiddle multiply

        bool r2c2r;								// real to complex or complex to real transform
        bool r2c, c2r;
        bool rcFull;
        bool rcSimple;

        bool blockCompute;						// When we have to compute FFT in blocks (either read or write is along columns)
        BlockComputeType blockComputeType;
        size_t blockWidth, blockWGS, blockLDS;

        bool realSpecial;

        const FFTKernelGenKeyParams params;		// key params


        inline std::string IterRegs(const std::string &pfx, bool initComma = true)
        {
            std::string str = "";

            if (linearRegs)
            {
                if (initComma) str += ", ";

                for (size_t i = 0; i<cnPerWI; i++)
                {
                    if (i != 0) str += ", ";
                    str += pfx; str += "R";
                    str += SztToStr(i);
                }
            }

            return str;
        }

        inline bool IsGroupedReadWritePossible()
        {
            bool possible = true;
            const size_t *iStride, *oStride;

            if (r2c2r)
                return false;

            if (realSpecial)
                return false;

            if (params.fft_placeness == CLFFT_INPLACE)
            {
                iStride = oStride = params.fft_inStride;
            }
            else
            {
                iStride = params.fft_inStride;
                oStride = params.fft_outStride;
            }

            for (size_t i = 1; i < params.fft_DataDim; i++)
            {
                if (iStride[i] % 2) { possible = false; break; }
                if (oStride[i] % 2) { possible = false; break; }
            }

            return possible;
        }

        inline std::string OffsetCalcBlock(const std::string &off, bool input = true)
        {
            std::string str;

            const size_t *pStride = input ? params.fft_inStride : params.fft_outStride;

            str += "\t"; str += off; str += " = ";
            std::string nextBatch = "batch";
            for (size_t i = (params.fft_DataDim - 1); i>2; i--)
            {
                size_t currentLength = 1;
                for (int j = 2; j<i; j++) currentLength *= params.fft_N[j];
                currentLength *= (params.fft_N[1] / blockWidth);

                str += "("; str += nextBatch; str += "/"; str += SztToStr(currentLength);
                str += ")*"; str += SztToStr(pStride[i]); str += " + ";

                nextBatch = "(" + nextBatch + "%" + SztToStr(currentLength) + ")";
            }

            str += "("; str += nextBatch; str += "/"; str += SztToStr(params.fft_N[1] / blockWidth);
            str += ")*"; str += SztToStr(pStride[2]); str += " + ("; str += nextBatch;
            str += "%"; str += SztToStr(params.fft_N[1] / blockWidth); str += ")*";
            if ((input && (blockComputeType == BCT_R2C)) || (!input && (blockComputeType == BCT_C2R)))
                str += SztToStr(blockWidth*length);
            else
                str += SztToStr(blockWidth);
            str += ";\n";

            return str;
        }

        inline std::string OffsetCalc(const std::string &off, bool input = true, bool rc_second_index = false)
        {
            std::string str;

            const size_t *pStride = input ? params.fft_inStride : params.fft_outStride;

            std::string batch;
            if (r2c2r && !rcSimple)
            {
                batch += "(batch*"; batch += SztToStr(2 * numTrans);
                if (rc_second_index) batch += " + 1";
                else				batch += " + 0";

                if (numTrans != 1) { batch += " + 2*(me/"; batch += SztToStr(workGroupSizePerTrans); batch += "))"; }
                else { batch += ")"; }
            }
            else
            {
                if (numTrans == 1) { batch += "batch"; }
                else {
                    batch += "(batch*"; batch += SztToStr(numTrans);
                    batch += " + (me/"; batch += SztToStr(workGroupSizePerTrans); batch += "))";
                }
            }

            str += "\t"; str += off; str += " = ";
            std::string nextBatch = batch;
            for (size_t i = (params.fft_DataDim - 1); i>1; i--)
            {
                size_t currentLength = 1;
                for (int j = 1; j<i; j++) currentLength *= params.fft_N[j];

                str += "("; str += nextBatch; str += "/"; str += SztToStr(currentLength);
                str += ")*"; str += SztToStr(pStride[i]); str += " + ";

                nextBatch = "(" + nextBatch + "%" + SztToStr(currentLength) + ")";
            }

            str += nextBatch; str += "*"; str += SztToStr(pStride[1]); str += ";\n";

            return str;
        }

    public:
        Kernel(const FFTKernelGenKeyParams &paramsVal) :
            params(paramsVal), r2c2r(false)

        {
            length = params.fft_N[0];
            workGroupSize = params.fft_SIMD;
            numTrans = (workGroupSize * params.fft_R) / length;

            r2c = false;
            c2r = false;
            // Check if it is R2C or C2R transform
            if (params.fft_inputLayout == CLFFT_REAL)  r2c = true;
            if (params.fft_outputLayout == CLFFT_REAL) c2r = true;
            r2c2r = (r2c || c2r);

            if (r2c)
            {
                rcFull = ((params.fft_outputLayout == CLFFT_COMPLEX_INTERLEAVED) ||
                    (params.fft_outputLayout == CLFFT_COMPLEX_PLANAR)) ? true : false;
            }
            if (c2r)
            {
                rcFull = ((params.fft_inputLayout == CLFFT_COMPLEX_INTERLEAVED) ||
                    (params.fft_inputLayout == CLFFT_COMPLEX_PLANAR)) ? true : false;
            }

            rcSimple = params.fft_RCsimple;

            halfLds = true;
            linearRegs = true;

            realSpecial = params.fft_realSpecial;

            blockCompute = params.blockCompute;
            blockComputeType = params.blockComputeType;
            // Make sure we can utilize all Lds if we are going to
            // use blocked columns to compute FFTs
            if (blockCompute)
            {
                assert(length <= 256);  // 256 parameter comes from prototype experiments
                                        // largest length at which block column possible given 32KB LDS limit
                                        // if LDS limit is different this number need to be changed appropriately
                halfLds = false;
                linearRegs = true;
            }

            assert(((length*numTrans) % workGroupSize) == 0);
            cnPerWI = (numTrans * length) / workGroupSize;
            workGroupSizePerTrans = workGroupSize / numTrans;

            // !!!! IMPORTANT !!!! Keep these assertions unchanged, algorithm depend on these to be true
            assert((cnPerWI * workGroupSize) == (numTrans * length));
            assert(cnPerWI <= length); // Don't do more than 1 fft per work-item

                                       // Breakdown into passes

            size_t LS = 1;
            size_t L;
            size_t R = length;
            size_t pid = 0;

            // See if we can get radices from the lookup table
            const size_t *pRadices = NULL;
            size_t nPasses;
            KernelCoreSpecs<PR> kcs;
            kcs.GetRadices(length, nPasses, pRadices);
            if ((params.fft_MaxWorkGroupSize >= 256) && (pRadices != NULL))
            {
                for (size_t i = 0; i<nPasses; i++)
                {
                    size_t rad = pRadices[i];
                    L = LS * rad;
                    R /= rad;

                    radices.push_back(rad);
                    passes.push_back(Pass<PR>(i, length, rad, cnPerWI, L, LS, R, linearRegs, halfLds, r2c, c2r, rcFull, rcSimple, realSpecial));

                    //Pass precallback information to Pass object if its the first pass. 
                    //This will be used in single kernel transforms
                    if (params.fft_hasPreCallback && i == 0 && !params.blockCompute)
                    {
                        passes[0].SetPrecallback(params.fft_hasPreCallback, params.fft_preCallback);
                    }

                    //Pass post-callback information to Pass object if its the last pass. 
                    //This will be used in single kernel transforms
                    if (params.fft_hasPostCallback && i == (nPasses - 1) && !params.blockCompute)
                    {
                        passes[i].SetPostcallback(params.fft_hasPostCallback, params.fft_postCallback);
                    }

                    LS *= rad;
                }
                assert(R == 1); // this has to be true for correct radix composition of the length
                numPasses = nPasses;
            }
            else
            {
                // Possible radices
                size_t cRad[] = { 13,11,10,8,7,6,5,4,3,2,1 }; // Must be in descending order
                size_t cRadSize = (sizeof(cRad) / sizeof(cRad[0]));

                // Generate the radix and pass objects
                while (true)
                {
                    size_t rad;

                    assert(cRadSize >= 1);

                    // Picks the radices in descending order (biggest radix first)
                    for (size_t r = 0; r<cRadSize; r++)
                    {
                        rad = cRad[r];

                        if ((rad > cnPerWI) || (cnPerWI%rad))
                            continue;

                        if (!(R % rad))
                            break;
                    }

                    assert((cnPerWI%rad) == 0);

                    L = LS * rad;
                    R /= rad;

                    radices.push_back(rad);
                    passes.push_back(Pass<PR>(pid, length, rad, cnPerWI, L, LS, R, linearRegs, halfLds, r2c, c2r, rcFull, rcSimple, realSpecial));

                    //Pass precallback information to Pass object if its the first pass. 
                    //This will be used in single kernel transforms
                    if (pid == 0 && params.fft_hasPreCallback)
                    {
                        passes[0].SetPrecallback(params.fft_hasPreCallback, params.fft_preCallback);
                    }

                    pid++;
                    LS *= rad;

                    assert(R >= 1);
                    if (R == 1)
                        break;
                }
                numPasses = pid;

                //Pass post-callback information to Pass object if its the last pass. 
                //This will be used in single kernel transforms
                if (params.fft_hasPostCallback)
                {
                    passes[numPasses - 1].SetPostcallback(params.fft_hasPostCallback, params.fft_postCallback);
                }
            }

            assert(numPasses == passes.size());
            assert(numPasses == radices.size());

#ifdef PARMETERS_TO_BE_READ

            ParamRead pr;
            ReadParameterFile(pr);

            radices.clear();
            passes.clear();

            radices = pr.radices;
            numPasses = radices.size();

            LS = 1;
            R = length;
            for (size_t i = 0; i<numPasses; i++)
            {
                size_t rad = radices[i];
                L = LS * rad;
                R /= rad;

                passes.push_back(Pass<PR>(i, length, rad, cnPerWI, L, LS, R, linearRegs));

                LS *= rad;
            }
            assert(R == 1);
#endif

            // Grouping read/writes ok?
            bool grp = IsGroupedReadWritePossible();
            for (size_t i = 0; i < numPasses; i++)
                passes[i].SetGrouping(grp);

            // Store the next pass-object pointers
            if (numPasses > 1)
                for (size_t i = 0; i < (numPasses - 1); i++)
                    passes[i].SetNextPass(&passes[i + 1]);


            if (blockCompute)
            {
                blockWidth = BlockSizes::BlockWidth(length);
                blockWGS = BlockSizes::BlockWorkGroupSize(length);
                blockLDS = BlockSizes::BlockLdsSize(length);
            }
            else
            {
                blockWidth = blockWGS = blockLDS = 0;
            }
        }

        class BlockSizes
        {
        public:
            enum ValType
            {
                BS_VT_WGS,
                BS_VT_BWD,
                BS_VT_LDS,
            };

            static size_t BlockLdsSize(size_t N) { return GetValue(N, BS_VT_LDS); }
            static size_t BlockWidth(size_t N) { return GetValue(N, BS_VT_BWD); }
            static size_t BlockWorkGroupSize(size_t N) { return GetValue(N, BS_VT_WGS); }

        private:

            static size_t GetValue(size_t N, ValType vt)
            {
                size_t wgs; // preferred work group size
                size_t bwd; // block width to be used
                size_t lds; // LDS size to be used for the block


                KernelCoreSpecs<PR> kcs;
                size_t t_wgs, t_nt;
                kcs.GetWGSAndNT(N, t_wgs, t_nt);

                switch (N)
                {
                case 256:	bwd = 8 / PrecisionWidth<PR>();   wgs = (bwd > t_nt) ? 256 : t_wgs; break;
                case 128:	bwd = 8 / PrecisionWidth<PR>();   wgs = (bwd > t_nt) ? 128 : t_wgs; break;
                case 64:	bwd = 16 / PrecisionWidth<PR>();  wgs = (bwd > t_nt) ? 128 : t_wgs; break;
                case 32:	bwd = 32 / PrecisionWidth<PR>();  wgs = (bwd > t_nt) ? 64 : t_wgs; break;
                case 16:	bwd = 64 / PrecisionWidth<PR>();  wgs = (bwd > t_nt) ? 64 : t_wgs; break;
                case 8:		bwd = 128 / PrecisionWidth<PR>(); wgs = (bwd > t_nt) ? 64 : t_wgs; break;
                default:	assert(false);
                }

                // block width cannot be less than numTrans, math in other parts of code depend on this assumption
                assert(bwd >= t_nt);

                lds = N*bwd;

                switch (vt)
                {
                case BS_VT_WGS: return wgs;
                case BS_VT_BWD: return bwd;
                case BS_VT_LDS: return lds;
                default: assert(false); return 0;
                }
            }
        };

        void GenerateKernel(std::string &str, cl_device_id Dev_ID)
        {
            std::string twType = RegBaseType<PR>(2);
            std::string rType = RegBaseType<PR>(1);
            std::string r2Type = RegBaseType<PR>(2);

            bool inInterleaved;	 // Input is interleaved format
            bool outInterleaved; // Output is interleaved format
            inInterleaved = ((params.fft_inputLayout == CLFFT_COMPLEX_INTERLEAVED) ||
                (params.fft_inputLayout == CLFFT_HERMITIAN_INTERLEAVED)) ? true : false;
            outInterleaved = ((params.fft_outputLayout == CLFFT_COMPLEX_INTERLEAVED) ||
                (params.fft_outputLayout == CLFFT_HERMITIAN_INTERLEAVED)) ? true : false;

            // use interleaved LDS when halfLds constraint absent
            bool ldsInterleaved = inInterleaved || outInterleaved;
            ldsInterleaved = halfLds ? false : ldsInterleaved;
            ldsInterleaved = blockCompute ? true : ldsInterleaved;

            bool inReal;  // Input is real format
            bool outReal; // Output is real format
            inReal = (params.fft_inputLayout == CLFFT_REAL) ? true : false;
            outReal = (params.fft_outputLayout == CLFFT_REAL) ? true : false;

            size_t large1D = 0;
            if (params.fft_realSpecial)
                large1D = params.fft_N[0] * params.fft_realSpecial_Nr;
            else
                large1D = params.fft_N[0] * params.fft_N[1];

            // Pragma
            str += ClPragma<PR>();

            // Twiddle table
            if (length > 1)
            {
                TwiddleTable twTable(length);

                str += "\n__constant ";
                str += twType; str += " ";
                str += TwTableName();
                str += "["; str += SztToStr(length - 1); str += "] = {\n";
                twTable.GenerateTwiddleTable<PR>(radices, str);
                str += "};\n\n";
            }
            str += "\n";

            // twiddle factors for 1d-large 3-step algorithm
            if (params.fft_3StepTwiddle)
            {
                TwiddleTableLarge twLarge(large1D);
                twLarge.GenerateTwiddleTable<PR>(str);
            }

            std::string sfx = FloatSuffix<PR>();

            // Base type
            str += "#define fptype "; str += RegBaseType<PR>(1); str += "\n\n";

            // Vector type
            str += "#define fvect2 "; str += RegBaseType<PR>(2); str += "\n\n";

            bool cReg = linearRegs ? true : false;

            // Generate butterflies for all unique radices
            std::list<size_t> uradices;
            for (std::vector<size_t>::const_iterator r = radices.begin(); r != radices.end(); r++)
                uradices.push_back(*r);

            uradices.sort();
            uradices.unique();


            //constants
            if (length % 8 == 0)
            {
                str += "#define C8Q  0.70710678118654752440084436210485"; str += sfx; str += "\n";
            }

            if (length % 5 == 0)
            {
                str += "#define C5QA 0.30901699437494742410229341718282"; str += sfx; str += "\n";
                str += "#define C5QB 0.95105651629515357211643933337938"; str += sfx; str += "\n";
                str += "#define C5QC 0.50000000000000000000000000000000"; str += sfx; str += "\n";
                str += "#define C5QD 0.58778525229247312916870595463907"; str += sfx; str += "\n";
                str += "#define C5QE 0.80901699437494742410229341718282"; str += sfx; str += "\n";
            }

            if (length % 3 == 0)
            {
                str += "#define C3QA 0.50000000000000000000000000000000"; str += sfx; str += "\n";
                str += "#define C3QB 0.86602540378443864676372317075294"; str += sfx; str += "\n";
            }

            if (length % 7 == 0)
            {
                str += "#define C7Q1 -1.16666666666666651863693004997913"; str += sfx; str += "\n";
                str += "#define C7Q2  0.79015646852540022404554065360571"; str += sfx; str += "\n";
                str += "#define C7Q3  0.05585426728964774240049351305970"; str += sfx; str += "\n";
                str += "#define C7Q4  0.73430220123575240531721419756650"; str += sfx; str += "\n";
                str += "#define C7Q5  0.44095855184409837868031445395900"; str += sfx; str += "\n";
                str += "#define C7Q6  0.34087293062393136944265847887436"; str += sfx; str += "\n";
                str += "#define C7Q7 -0.53396936033772524066165487965918"; str += sfx; str += "\n";
                str += "#define C7Q8  0.87484229096165666561546458979137"; str += sfx; str += "\n";
            }

            if (length % 11 == 0)
            {
                str += "#define b11_0 0.9898214418809327"; str += sfx; str += "\n";
                str += "#define b11_1 0.9594929736144973"; str += sfx; str += "\n";
                str += "#define b11_2 0.9189859472289947"; str += sfx; str += "\n";
                str += "#define b11_3 0.8767688310025893"; str += sfx; str += "\n";
                str += "#define b11_4 0.8308300260037728"; str += sfx; str += "\n";
                str += "#define b11_5 0.7784344533346518"; str += sfx; str += "\n";
                str += "#define b11_6 0.7153703234534297"; str += sfx; str += "\n";
                str += "#define b11_7 0.6343562706824244"; str += sfx; str += "\n";
                str += "#define b11_8 0.3425847256816375"; str += sfx; str += "\n";
                str += "#define b11_9 0.5211085581132027"; str += sfx; str += "\n";
            }

            if (length % 13 == 0)
            {
                str += "#define b13_0  0.9682872443619840"; str += sfx; str += "\n";
                str += "#define b13_1  0.9578059925946651"; str += sfx; str += "\n";
                str += "#define b13_2  0.8755023024091479"; str += sfx; str += "\n";
                str += "#define b13_3  0.8660254037844386"; str += sfx; str += "\n";
                str += "#define b13_4  0.8595425350987748"; str += sfx; str += "\n";
                str += "#define b13_5  0.8534800018598239"; str += sfx; str += "\n";
                str += "#define b13_6  0.7693388175729806"; str += sfx; str += "\n";
                str += "#define b13_7  0.6865583707817543"; str += sfx; str += "\n";
                str += "#define b13_8  0.6122646503767565"; str += sfx; str += "\n";
                str += "#define b13_9  0.6004772719326652"; str += sfx; str += "\n";
                str += "#define b13_10 0.5817047785105157"; str += sfx; str += "\n";
                str += "#define b13_11 0.5751407294740031"; str += sfx; str += "\n";
                str += "#define b13_12 0.5220263851612750"; str += sfx; str += "\n";
                str += "#define b13_13 0.5200285718888646"; str += sfx; str += "\n";
                str += "#define b13_14 0.5165207806234897"; str += sfx; str += "\n";
                str += "#define b13_15 0.5149187780863157"; str += sfx; str += "\n";
                str += "#define b13_16 0.5035370328637666"; str += sfx; str += "\n";
                str += "#define b13_17 0.5000000000000000"; str += sfx; str += "\n";
                str += "#define b13_18 0.3027756377319946"; str += sfx; str += "\n";
                str += "#define b13_19 0.3014792600477098"; str += sfx; str += "\n";
                str += "#define b13_20 0.3004626062886657"; str += sfx; str += "\n";
                str += "#define b13_21 0.2517685164318833"; str += sfx; str += "\n";
                str += "#define b13_22 0.2261094450357824"; str += sfx; str += "\n";
                str += "#define b13_23 0.0833333333333333"; str += sfx; str += "\n";
                str += "#define b13_24 0.0386329546443481"; str += sfx; str += "\n";
            }

            str += "\n";


            //If pre-callback is set for the plan
            std::string callbackstr;
            if (params.fft_hasPreCallback)
            {
                //Insert pre-callback function code at the beginning 
                callbackstr += params.fft_preCallback.funcstring;
                callbackstr += "\n\n";

                str += callbackstr;
            }

            //If post-callback is set for the plan
            if (params.fft_hasPostCallback)
            {
                //Insert post-callback function code 
                str += params.fft_postCallback.funcstring;
                str += "\n\n";
            }

            typename std::vector< Pass<PR> >::const_iterator p;
            if (length > 1)
            {
                for (std::list<size_t>::const_iterator r = uradices.begin(); r != uradices.end(); r++)
                {
                    size_t rad = *r;
                    p = passes.begin();
                    while (p->GetRadix() != rad) p++;

                    for (size_t d = 0; d<2; d++)
                    {
                        bool fwd = d ? false : true;

                        if (p->GetNumB1()) { Butterfly<PR> bfly(rad, 1, fwd, cReg); bfly.GenerateButterfly(str); str += "\n"; }
                        if (p->GetNumB2()) { Butterfly<PR> bfly(rad, 2, fwd, cReg); bfly.GenerateButterfly(str); str += "\n"; }
                        if (p->GetNumB4()) { Butterfly<PR> bfly(rad, 4, fwd, cReg); bfly.GenerateButterfly(str); str += "\n"; }
                    }
                }
            }

            // Generate passes
            for (size_t d = 0; d<2; d++)
            {
                bool fwd;

                if (r2c2r)
                {
                    fwd = r2c;
                }
                else
                {
                    fwd = d ? false : true;
                }

                double scale = fwd ? params.fft_fwdScale : params.fft_backScale;

                for (p = passes.begin(); p != passes.end(); p++)
                {
                    double s = 1.0;
                    size_t ins = 1, outs = 1;
                    bool gIn = false, gOut = false;
                    bool inIlvd = false, outIlvd = false;
                    bool inRl = false, outRl = false;
                    bool tw3Step = false;


                    if (p == passes.begin() && params.fft_twiddleFront) { tw3Step = params.fft_3StepTwiddle; }
                    if ((p + 1) == passes.end()) { s = scale; if (!params.fft_twiddleFront) tw3Step = params.fft_3StepTwiddle; }

                    if (blockCompute && !r2c2r)
                    {
                        inIlvd = ldsInterleaved;
                        outIlvd = ldsInterleaved;
                    }
                    else
                    {
                        if (p == passes.begin()) { inIlvd = inInterleaved;  inRl = inReal;  gIn = true; ins = params.fft_inStride[0]; }
                        if ((p + 1) == passes.end()) { outIlvd = outInterleaved; outRl = outReal; gOut = true; outs = params.fft_outStride[0]; }

                        if (p != passes.begin()) { inIlvd = ldsInterleaved; }
                        if ((p + 1) != passes.end()) { outIlvd = ldsInterleaved; }
                    }

                    p->GeneratePass(fwd, str, tw3Step, params.fft_twiddleFront, inIlvd, outIlvd, inRl, outRl, ins, outs, s, gIn, gOut);
                }

                // if real transform we do only 1 direction
                if (r2c2r)
                    break;
            }



            // TODO : address this kludge
            str += " typedef union  { uint u; int i; } cb_t;\n\n";

            for (size_t d = 0; d<2; d++)
            {
                bool fwd;

                if (r2c2r)
                {
                    fwd = inReal ? true : false;
                }
                else
                {
                    fwd = d ? false : true;
                }

                // FFT kernel begin
                // Function attribute
                str += "__kernel __attribute__((reqd_work_group_size (";
                if (blockCompute)	str += SztToStr(blockWGS);
                else				str += SztToStr(workGroupSize);
                str += ",1,1)))\nvoid ";

                // Function name
                if (fwd) str += "fft_fwd";
                else	str += "fft_back";
                str += "(";

                // TODO : address this kludge
                size_t SizeParam_ret = 0;
                clGetDeviceInfo(Dev_ID, CL_DEVICE_VENDOR, 0, NULL, &SizeParam_ret);
                char* nameVendor = new char[SizeParam_ret];
                clGetDeviceInfo(Dev_ID, CL_DEVICE_VENDOR, SizeParam_ret, nameVendor, NULL);

                //nv compiler doesn't support __constant kernel argument
                if (strncmp(nameVendor, "NVIDIA", 6) != 0)
                    str += "__constant cb_t *cb __attribute__((max_constant_size(32))), ";
                else
                    str += "__global cb_t *cb, ";

                delete[] nameVendor;

                //If plan has pre/post callback
                callbackstr.clear();
                bool hasCallback = params.fft_hasPreCallback || params.fft_hasPostCallback;

                if (hasCallback)
                {
                    if (params.fft_hasPreCallback)
                    {
                        callbackstr += ", __global void* pre_userdata";
                    }
                    if (params.fft_hasPostCallback)
                    {
                        callbackstr += ", __global void* post_userdata";
                    }

                    if (params.fft_preCallback.localMemSize > 0 || params.fft_postCallback.localMemSize > 0)
                    {
                        callbackstr += ", __local void* localmem";
                    }
                }

                // Function attributes
                if (params.fft_placeness == CLFFT_INPLACE)
                {
                    if (r2c2r)
                    {
                        if (outInterleaved)
                        {
                            str += "__global "; str += r2Type; str += " * restrict gb";
                        }
                        else
                        {
                            str += "__global "; str += rType; str += " * restrict gb";
                        }

                        //If plan has callback
                        if (hasCallback)
                        {
                            str += callbackstr;
                        }

                        str += ")\n";
                    }
                    else
                    {
                        assert(inInterleaved == outInterleaved);
                        assert(params.fft_inStride[1] == params.fft_outStride[1]);
                        assert(params.fft_inStride[0] == params.fft_outStride[0]);

                        if (inInterleaved)
                        {
                            str += "__global "; str += r2Type; str += " * restrict gb";

                            //If plan has callback
                            if (hasCallback)
                            {
                                str += callbackstr;
                            }

                            str += ")\n";
                        }
                        else
                        {
                            str += "__global "; str += rType; str += " * restrict gbRe, ";
                            str += "__global "; str += rType; str += " * restrict gbIm";

                            //If plan has callback
                            if (hasCallback)
                            {
                                str += callbackstr;
                            }

                            str += ")\n";
                        }
                    }
                }
                else
                {
                    if (r2c2r)
                    {
                        if (inInterleaved)
                        {
                            str += "__global "; str += r2Type; str += " * restrict gbIn, ";
                        }
                        else if (inReal)
                        {
                            str += "__global "; str += rType; str += " * restrict gbIn, ";
                        }
                        else
                        {
                            str += "__global const "; str += rType; str += " * restrict gbInRe, ";
                            str += "__global const "; str += rType; str += " * restrict gbInIm, ";
                        }

                        if (outInterleaved)
                        {
                            str += "__global "; str += r2Type; str += " * restrict gbOut";
                        }
                        else if (outReal)
                        {
                            str += "__global "; str += rType; str += " * restrict gbOut";
                        }
                        else
                        {
                            str += "__global "; str += rType; str += " * restrict gbOutRe, ";
                            str += "__global "; str += rType; str += " * restrict gbOutIm";
                        }

                        //If plan has callback
                        if (hasCallback)
                        {
                            str += callbackstr;
                        }

                        str += ")\n";
                    }
                    else
                    {
                        if (inInterleaved)
                        {
                            str += "__global const "; str += r2Type; str += " * restrict gbIn, ";
                        }
                        else
                        {
                            str += "__global const "; str += rType; str += " * restrict gbInRe, ";
                            str += "__global const "; str += rType; str += " * restrict gbInIm, ";
                        }

                        if (outInterleaved)
                        {
                            str += "__global "; str += r2Type; str += " * restrict gbOut";
                        }
                        else
                        {
                            str += "__global "; str += rType; str += " * restrict gbOutRe, ";
                            str += "__global "; str += rType; str += " * restrict gbOutIm";
                        }

                        //If plan has callback
                        if (hasCallback)
                        {
                            str += callbackstr;
                        }

                        str += ")\n";
                    }
                }

                str += "{\n";

                // Initialize
                str += "\t";
                str += "uint me = get_local_id(0);\n\t";
                str += "uint batch = get_group_id(0);";
                str += "\n";



                // Allocate LDS
                if (blockCompute)
                {
                    str += "\n\t"; str += "__local "; str += r2Type; str += " lds[";
                    str += SztToStr(blockLDS); str += "];\n";
                }
                else
                {
                    size_t ldsSize = halfLds ? length*numTrans : 2 * length*numTrans;
                    ldsSize = ldsInterleaved ? ldsSize / 2 : ldsSize;

                    if (numPasses > 1)
                    {
                        str += "\n\t";
                        str += "__local "; str += ldsInterleaved ? r2Type : rType; str += " lds[";
                        str += SztToStr(ldsSize); str += "];\n";
                    }
                }

                // Declare memory pointers
                str += "\n\t";
                if (r2c2r)
                {
                    str += "uint iOffset;\n\t";
                    str += "uint oOffset;\n\n\t";
                    if (!rcSimple)
                    {
                        str += "uint iOffset2;\n\t";
                        str += "uint oOffset2;\n\n\t";
                    }

                    if (!params.fft_hasPreCallback)
                    {
                        if (inInterleaved)
                        {
                            if (!rcSimple) { str += "__global "; str += r2Type; str += " *lwbIn2;\n\t"; }
                            str += "__global "; str += r2Type; str += " *lwbIn;\n\t";
                        }
                        else if (inReal)
                        {
                            if (!rcSimple) { str += "__global "; str += rType; str += " *lwbIn2;\n\t"; }
                            str += "__global "; str += rType; str += " *lwbIn;\n\t";

                        }
                        else
                        {
                            if (!rcSimple) { str += "__global "; str += rType; str += " *lwbInRe2;\n\t"; }
                            if (!rcSimple) { str += "__global "; str += rType; str += " *lwbInIm2;\n\t"; }

                            str += "__global "; str += rType; str += " *lwbInRe;\n\t";
                            str += "__global "; str += rType; str += " *lwbInIm;\n\t";

                        }
                    }

                    if (outInterleaved)
                    {
                        if (!params.fft_hasPostCallback)
                        {
                            if (!rcSimple) { str += "__global "; str += r2Type; str += " *lwbOut2;\n\t"; }
                            str += "__global "; str += r2Type; str += " *lwbOut;\n";
                        }
                    }
                    else if (outReal)
                    {
                        if (!params.fft_hasPostCallback)
                        {
                            if (!rcSimple) { str += "__global "; str += rType; str += " *lwbOut2;\n\t"; }
                            str += "__global "; str += rType; str += " *lwbOut;\n";
                        }
                    }
                    else
                    {
                        if (!params.fft_hasPostCallback)
                        {
                            if (!rcSimple) { str += "__global "; str += rType; str += " *lwbOutRe2;\n\t"; }
                            if (!rcSimple) { str += "__global "; str += rType; str += " *lwbOutIm2;\n\t"; }
                            str += "__global "; str += rType; str += " *lwbOutRe;\n\t";
                            str += "__global "; str += rType; str += " *lwbOutIm;\n";
                        }
                    }
                    str += "\n";
                }
                else
                {
                    if (params.fft_placeness == CLFFT_INPLACE)
                    {
                        str += "uint ioOffset;\n\t";

                        //Skip if callback is set 
                        if (!params.fft_hasPreCallback || !params.fft_hasPostCallback)
                        {
                            if (inInterleaved)
                            {
                                str += "__global "; str += r2Type; str += " *lwb;\n";
                            }
                            else
                            {
                                str += "__global "; str += rType; str += " *lwbRe;\n\t";
                                str += "__global "; str += rType; str += " *lwbIm;\n";
                            }
                        }
                        str += "\n";
                    }
                    else
                    {
                        str += "uint iOffset;\n\t";
                        str += "uint oOffset;\n\t";

                        //Skip if precallback is set 
                        if (!(params.fft_hasPreCallback))
                        {
                            if (inInterleaved)
                            {
                                str += "__global "; str += r2Type; str += " *lwbIn;\n\t";
                            }
                            else
                            {
                                str += "__global "; str += rType; str += " *lwbInRe;\n\t";
                                str += "__global "; str += rType; str += " *lwbInIm;\n\t";
                            }
                        }

                        //Skip if postcallback is set 
                        if (!params.fft_hasPostCallback)
                        {
                            if (outInterleaved)
                            {
                                str += "__global "; str += r2Type; str += " *lwbOut;\n";
                            }
                            else
                            {
                                str += "__global "; str += rType; str += " *lwbOutRe;\n\t";
                                str += "__global "; str += rType; str += " *lwbOutIm;\n";
                            }
                        }
                        str += "\n";
                    }
                }

                // Setup registers if needed
                if (linearRegs)
                {
                    str += "\t"; str += RegBaseType<PR>(2);
                    str += " "; str += IterRegs("", false);
                    str += ";\n\n";
                }

                // Calculate total transform count
                std::string totalBatch = "(";
                size_t i = 0;
                while (i < (params.fft_DataDim - 2))
                {
                    totalBatch += SztToStr(params.fft_N[i + 1]); totalBatch += " * ";
                    i++;
                }
                totalBatch += "cb[0].u)";

                // Conditional read-write ('rw') for arbitrary batch number
                if (r2c2r && !rcSimple)
                {
                    str += "\tuint this = "; str += totalBatch; str += " - batch*";
                    str += SztToStr(2 * numTrans); str += ";\n";
                    str += "\tuint rw = (me < ((this+1)/2)*"; str += SztToStr(workGroupSizePerTrans);
                    str += ") ? (this - 2*(me/"; str += SztToStr(workGroupSizePerTrans); str += ")) : 0;\n\n";
                }
                else
                {
                    if ((numTrans > 1) && !blockCompute)
                    {
                        str += "\tuint rw = (me < ("; str += totalBatch;
                        str += " - batch*"; str += SztToStr(numTrans); str += ")*";
                        str += SztToStr(workGroupSizePerTrans); str += ") ? 1 : 0;\n\n";
                    }
                    else
                    {
                        str += "\tuint rw = 1;\n\n";
                    }
                }

                // Transform index for 3-step twiddles
                if (params.fft_3StepTwiddle && !blockCompute)
                {
                    if (numTrans == 1)
                    {
                        str += "\tuint b = batch%";
                    }
                    else
                    {
                        str += "\tuint b = (batch*"; str += SztToStr(numTrans); str += " + (me/";
                        str += SztToStr(workGroupSizePerTrans); str += "))%";
                    }

                    str += SztToStr(params.fft_N[1]); str += ";\n\n";

                    if (params.fft_realSpecial)
                    {
                        str += "\tuint bt = b;\n\n";
                    }
                }
                else
                {
                    str += "\tuint b = 0;\n\n";
                }

                // Setup memory pointers
                if (r2c2r)
                {
                    str += OffsetCalc("iOffset", true);
                    str += OffsetCalc("oOffset", false);
                    if (!rcSimple) { str += OffsetCalc("iOffset2", true, true); }
                    if (!rcSimple) { str += OffsetCalc("oOffset2", false, true); }

                    str += "\n\t";
                    if (params.fft_placeness == CLFFT_INPLACE)
                    {
                        if (!params.fft_hasPreCallback)
                        {
                            if (inInterleaved)
                            {
                                if (!rcSimple) { str += "lwbIn2 = (__global "; str += r2Type; str += " *)gb + iOffset2;\n\t"; }
                                str += "lwbIn  = (__global "; str += r2Type; str += " *)gb + iOffset;\n\t";
                            }
                            else
                            {
                                if (!rcSimple) { str += "lwbIn2 = (__global "; str += rType; str += " *)gb + iOffset2;\n\t"; }
                                str += "lwbIn  = (__global "; str += rType; str += " *)gb + iOffset;\n\t";

                            }
                        }

                        if (!params.fft_hasPostCallback)
                        {
                            if (!rcSimple) { str += "lwbOut2 = gb + oOffset2;\n\t"; }
                            str += "lwbOut = gb + oOffset;\n";
                        }
                        str += "\n";
                    }
                    else
                    {
                        if (!params.fft_hasPreCallback)
                        {
                            if (inInterleaved || inReal)
                            {
                                if (!rcSimple) { str += "lwbIn2 = gbIn + iOffset2;\n\t"; }
                                str += "lwbIn = gbIn + iOffset;\n\t";
                            }
                            else
                            {
                                if (!rcSimple) { str += "lwbInRe2 = gbInRe + iOffset2;\n\t"; }
                                if (!rcSimple) { str += "lwbInIm2 = gbInIm + iOffset2;\n\t"; }
                                str += "lwbInRe = gbInRe + iOffset;\n\t";
                                str += "lwbInIm = gbInIm + iOffset;\n\t";
                            }
                        }

                        if (!params.fft_hasPostCallback)
                        {
                            if (outInterleaved || outReal)
                            {
                                if (!rcSimple) { str += "lwbOut2 = gbOut + oOffset2;\n\t"; }
                                str += "lwbOut = gbOut + oOffset;\n";
                            }
                            else
                            {

                                if (!rcSimple) { str += "lwbOutRe2 = gbOutRe + oOffset2;\n\t"; }
                                if (!rcSimple) { str += "lwbOutIm2 = gbOutIm + oOffset2;\n\t"; }
                                str += "lwbOutRe = gbOutRe + oOffset;\n\t";
                                str += "lwbOutIm = gbOutIm + oOffset;\n";
                            }
                        }
                        str += "\n";
                    }
                }
                else
                {
                    if (params.fft_placeness == CLFFT_INPLACE)
                    {
                        if (blockCompute)
                            str += OffsetCalcBlock("ioOffset", true);
                        else
                            str += OffsetCalc("ioOffset", true);

                        str += "\t";

                        //Skip if callback is set 
                        if (!params.fft_hasPreCallback || !params.fft_hasPostCallback)
                        {
                            if (inInterleaved)
                            {
                                str += "lwb = gb + ioOffset;\n";
                            }
                            else
                            {
                                str += "lwbRe = gbRe + ioOffset;\n\t";
                                str += "lwbIm = gbIm + ioOffset;\n";
                            }
                        }
                        str += "\n";
                    }
                    else
                    {
                        if (blockCompute)
                        {
                            str += OffsetCalcBlock("iOffset", true);
                            str += OffsetCalcBlock("oOffset", false);
                        }
                        else
                        {
                            str += OffsetCalc("iOffset", true);
                            str += OffsetCalc("oOffset", false);
                        }

                        str += "\t";

                        //Skip if precallback is set 
                        if (!(params.fft_hasPreCallback))
                        {
                            if (inInterleaved)
                            {
                                str += "lwbIn = gbIn + iOffset;\n\t";
                            }
                            else
                            {
                                str += "lwbInRe = gbInRe + iOffset;\n\t";
                                str += "lwbInIm = gbInIm + iOffset;\n\t";
                            }
                        }

                        //Skip if postcallback is set 
                        if (!params.fft_hasPostCallback)
                        {
                            if (outInterleaved)
                            {
                                str += "lwbOut = gbOut + oOffset;\n";
                            }
                            else
                            {
                                str += "lwbOutRe = gbOutRe + oOffset;\n\t";
                                str += "lwbOutIm = gbOutIm + oOffset;\n";
                            }
                        }
                        str += "\n";
                    }
                }

                std::string inOffset;
                std::string outOffset;
                if (params.fft_placeness == CLFFT_INPLACE && !r2c2r)
                {
                    inOffset += "ioOffset";
                    outOffset += "ioOffset";
                }
                else
                {
                    inOffset += "iOffset";
                    outOffset += "oOffset";
                }

                // Read data into LDS for blocked access
                if (blockCompute)
                {

                    size_t loopCount = (length * blockWidth) / blockWGS;

                    if ((blockComputeType == BCT_C2C) && params.fft_hasPreCallback)
                    {
                        str += "\n\t"; str += r2Type; str += " retCallback;";
                    }

                    str += "\n\tfor(uint t=0; t<"; str += SztToStr(loopCount);
                    str += "; t++)\n\t{\n";

                    //get offset 
                    std::string bufOffset;

                    for (size_t c = 0; c<2; c++)
                    {
                        std::string comp = "";
                        std::string readBuf = (params.fft_placeness == CLFFT_INPLACE) ? "lwb" : "lwbIn";
                        if (!inInterleaved) comp = c ? ".y" : ".x";
                        if (!inInterleaved)
                            readBuf = (params.fft_placeness == CLFFT_INPLACE) ? (c ? "lwbIm" : "lwbRe") : (c ? "lwbInIm" : "lwbInRe");

                        if ((blockComputeType == BCT_C2C) || (blockComputeType == BCT_C2R))
                        {
                            bufOffset.clear();
                            bufOffset += "(me%"; bufOffset += SztToStr(blockWidth); bufOffset += ") + ";
                            bufOffset += "(me/"; bufOffset += SztToStr(blockWidth); bufOffset += ")*"; bufOffset += SztToStr(params.fft_inStride[0]);
                            bufOffset += " + t*"; bufOffset += SztToStr(params.fft_inStride[0] * blockWGS / blockWidth);

                            if ((blockComputeType == BCT_C2C) && params.fft_hasPreCallback)
                            {
                                if (c == 0)
                                {
                                    str += "\t\tretCallback = "; str += params.fft_preCallback.funcname; str += "(";

                                    if (inInterleaved)
                                    {
                                        str += (params.fft_placeness == CLFFT_INPLACE) ? "gb, " : "gbIn, ";
                                    }
                                    else
                                    {
                                        str += (params.fft_placeness == CLFFT_INPLACE) ? "gbRe, gbIm, " : "gbInRe, gbInIm, ";
                                    }

                                    str += inOffset; str += " + "; str += bufOffset; str += ", pre_userdata";
                                    str += (params.fft_preCallback.localMemSize > 0) ? str += ", localmem);\n" : ");\n";
                                }

                                str += "\t\tR0"; str += comp; str += " = retCallback"; str += comp; str += ";\n";
                            }
                            else
                            {
                                str += "\t\tR0"; str += comp; str += " = ";
                                str += readBuf; str += "[";	str += bufOffset; str += "];\n";
                            }
                        }
                        else
                        {
                            str += "\t\tR0"; str += comp; str += " = "; str += readBuf; str += "[me + t*"; str += SztToStr(blockWGS); str += "];\n";
                        }


                        if (inInterleaved) break;
                    }

                    if ((blockComputeType == BCT_C2C) || (blockComputeType == BCT_C2R))
                    {
                        str += "\t\tlds[t*"; str += SztToStr(blockWGS / blockWidth); str += " + ";
                        str += "(me%"; str += SztToStr(blockWidth); str += ")*"; str += SztToStr(length); str += " + ";
                        str += "(me/"; str += SztToStr(blockWidth); str += ")] = R0;"; str += "\n";
                    }
                    else
                    {
                        str += "\t\tlds[t*"; str += SztToStr(blockWGS); str += " + me] = R0;"; str += "\n";
                    }

                    str += "\t}\n\n";
                    str += "\tbarrier(CLK_LOCAL_MEM_FENCE);\n\n";
                }


                // Set rw and 'me' per transform
                // rw string also contains 'b'
                std::string rw, me;

                if (r2c2r && !rcSimple)	rw = "rw, b, ";
                else					rw = ((numTrans > 1) || realSpecial) ? "rw, b, " : "1, b, ";

                if (numTrans > 1) { me += "me%"; me += SztToStr(workGroupSizePerTrans); me += ", "; }
                else { me += "me, "; }

                if (blockCompute) { me = "me%"; me += SztToStr(workGroupSizePerTrans); me += ", "; }

                // Buffer strings
                std::string inBuf, outBuf;
                if (r2c2r)
                {
                    if (rcSimple)
                    {
                        if (inInterleaved || inReal)		inBuf = params.fft_hasPreCallback ? "gbIn, " : "lwbIn, ";
                        else							inBuf = "lwbInRe, lwbInIm, ";
                        if (outInterleaved || outReal)	outBuf = params.fft_hasPostCallback ? "gbOut" : "lwbOut";
                        else							outBuf = "lwbOutRe, lwbOutIm";
                    }
                    else
                    {
                        if (inInterleaved || inReal)
                        {
                            if (!params.fft_hasPreCallback)
                            {
                                inBuf = "lwbIn, lwbIn2, ";
                            }
                            else
                            {
                                if (params.fft_placeness == CLFFT_INPLACE)
                                {
                                    inBuf = "(__global "; inBuf += r2c ? rType : r2Type; inBuf += "*) gb, ";
                                    inBuf += "(__global "; inBuf += r2c ? rType : r2Type; inBuf += "*) gb, ";
                                }
                                else
                                {
                                    inBuf = "gbIn, gbIn, ";
                                }
                            }
                        }
                        else							inBuf = (params.fft_hasPreCallback) ? "gbInRe, gbInRe, gbInIm, gbInIm, " : "lwbInRe, lwbInRe2, lwbInIm, lwbInIm2, ";

                        if (outInterleaved || outReal)	outBuf = params.fft_hasPostCallback ? ((params.fft_placeness == CLFFT_INPLACE) ? "gb, gb" : "gbOut, gbOut") : "lwbOut, lwbOut2";
                        else							outBuf = params.fft_hasPostCallback ? "gbOutRe, gbOutRe, gbOutIm, gbOutIm" : "lwbOutRe, lwbOutRe2, lwbOutIm, lwbOutIm2";
                    }
                }
                else
                {
                    if (params.fft_placeness == CLFFT_INPLACE)
                    {
                        if (inInterleaved)
                        {
                            inBuf = params.fft_hasPreCallback ? "gb, " : "lwb, ";
                            outBuf = params.fft_hasPostCallback ? "gb" : "lwb";
                        }
                        else
                        {
                            inBuf = params.fft_hasPreCallback ? "gbRe, gbIm, " : "lwbRe, lwbIm, ";
                            outBuf = params.fft_hasPostCallback ? "gbRe, gbIm" : "lwbRe, lwbIm";
                        }
                    }
                    else
                    {
                        if (inInterleaved)	inBuf = params.fft_hasPreCallback ? "gbIn, " : "lwbIn, ";
                        else				inBuf = params.fft_hasPreCallback ? "gbInRe, gbInIm, " : "lwbInRe, lwbInIm, ";
                        if (outInterleaved)	outBuf = params.fft_hasPostCallback ? "gbOut" : "lwbOut";
                        else				outBuf = params.fft_hasPostCallback ? "gbOutRe, gbOutIm" : "lwbOutRe, lwbOutIm";
                    }
                }


                if (blockCompute)
                {
                    str += "\n\tfor(uint t=0; t<"; str += SztToStr(blockWidth / (blockWGS / workGroupSizePerTrans));
                    str += "; t++)\n\t{\n\n";

                    inBuf = "lds, ";
                    outBuf = "lds";

                    if (params.fft_3StepTwiddle)
                    {
                        str += "\t\tb = (batch%"; str += SztToStr(params.fft_N[1] / blockWidth); str += ")*";
                        str += SztToStr(blockWidth); str += " + t*"; str += SztToStr(blockWGS / workGroupSizePerTrans);
                        str += " + (me/"; str += SztToStr(workGroupSizePerTrans); str += ");\n\n";
                    }
                }

                if (realSpecial)
                {
                    str += "\n\tfor(uint t=0; t<2; t++)\n\t{\n\n";
                }

                // Call passes
                if (numPasses == 1)
                {
                    str += "\t";
                    str += PassName(0, fwd);
                    str += "("; str += rw; str += me;

                    str += (params.fft_hasPreCallback) ? inOffset : "0";

                    if (params.fft_hasPostCallback)
                    {
                        str += ", "; str += outOffset; str += ", ";
                    }
                    else
                    {
                        str += ", 0, ";
                    }

                    str += inBuf; str += outBuf;
                    str += IterRegs("&");

                    //If callback is set
                    if (hasCallback)
                    {
                        //if pre-calback set 
                        if (params.fft_hasPreCallback)
                        {
                            str += (r2c2r && !rcSimple) ? ", iOffset2, pre_userdata" : ", pre_userdata";
                        }

                        //if post-calback set 
                        if (params.fft_hasPostCallback)
                        {
                            if ((r2c || c2r) && !rcSimple) { str += ", "; str += outOffset; str += "2"; }

                            str += ", post_userdata";
                        }

                        if (params.fft_preCallback.localMemSize > 0)
                        {
                            str += ", localmem";
                        }
                        if (params.fft_postCallback.localMemSize > 0)
                        {
                            //if precallback localmem also requested, send the localmem with the right offset
                            if (params.fft_hasPreCallback && params.fft_preCallback.localMemSize > 0)
                            {
                                str += ", ((__local char *)localmem + "; str += SztToStr(params.fft_preCallback.localMemSize); str += ")";
                            }
                            else
                            {
                                str += ", localmem";
                            }
                        }
                    }

                    str += ");\n";
                }
                else
                {
                    for (typename std::vector<Pass<PR> >::const_iterator p = passes.begin(); p != passes.end(); p++)
                    {
                        std::string exTab = "";
                        if (blockCompute || realSpecial) exTab = "\t";

                        str += exTab;
                        str += "\t";
                        str += PassName(p->GetPosition(), fwd);
                        str += "(";

                        std::string ldsOff;
                        if (blockCompute)
                        {
                            ldsOff += "t*"; ldsOff += SztToStr(length*(blockWGS / workGroupSizePerTrans)); ldsOff += " + (me/";
                            ldsOff += SztToStr(workGroupSizePerTrans); ldsOff += ")*"; ldsOff += SztToStr(length);
                        }
                        else
                        {
                            if (numTrans > 1)
                            {
                                ldsOff += "(me/"; ldsOff += SztToStr(workGroupSizePerTrans);
                                ldsOff += ")*"; ldsOff += SztToStr(length);
                            }
                            else
                            {
                                ldsOff += "0";
                            }
                        }

                        std::string ldsArgs;
                        if (halfLds) { ldsArgs += "lds, lds"; }
                        else {
                            if (ldsInterleaved) { ldsArgs += "lds"; }
                            else { ldsArgs += "lds, lds + "; ldsArgs += SztToStr(length*numTrans); }
                        }

                        str += rw;
                        if (params.fft_realSpecial) str += "t, ";
                        str += me;
                        if (p == passes.begin()) // beginning pass
                        {
                            if (blockCompute)
                            {
                                str += ldsOff;
                            }
                            else
                            {
                                str += (params.fft_hasPreCallback) ? inOffset : "0";
                            }
                            str += ", ";
                            str += ldsOff;
                            str += ", ";
                            str += inBuf;
                            str += ldsArgs; str += IterRegs("&");

                            //if precalback set, append additional arguments
                            if (!blockCompute && params.fft_hasPreCallback)
                            {
                                str += (r2c2r && !rcSimple) ? ", iOffset2, pre_userdata" : ", pre_userdata";

                                if (params.fft_preCallback.localMemSize > 0)
                                {
                                    str += ", localmem";
                                }
                            }

                            str += ");\n";
                            if (!halfLds) { str += exTab; str += "\tbarrier(CLK_LOCAL_MEM_FENCE);\n"; }
                        }
                        else if ((p + 1) == passes.end()) // ending pass
                        {
                            str += ldsOff;
                            str += ", ";
                            if (blockCompute)
                            {
                                str += ldsOff;
                            }
                            else
                            {
                                str += (params.fft_hasPostCallback) ? outOffset : "0";
                            }
                            str += ", ";
                            str += ldsArgs; str += ", ";
                            str += outBuf;

                            str += IterRegs("&");

                            if (!blockCompute && params.fft_hasPostCallback)
                            {
                                if ((c2r || r2c) && !rcSimple) { str += ", "; str += outOffset; str += "2"; }

                                str += ", post_userdata";

                                if (params.fft_postCallback.localMemSize > 0)
                                {
                                    //if precallback localmem also requested, send the localmem with the right offset
                                    if (params.fft_hasPreCallback && params.fft_preCallback.localMemSize > 0)
                                    {
                                        str += ", ((__local char *)localmem + "; str += SztToStr(params.fft_preCallback.localMemSize); str += ")";
                                    }
                                    else
                                    {
                                        str += ", localmem";
                                    }
                                }
                            }
                            str += ");\n";

                            if (!halfLds) { str += exTab; str += "\tbarrier(CLK_LOCAL_MEM_FENCE);\n"; }
                        }
                        else // intermediate pass
                        {
                            str += ldsOff;
                            str += ", ";
                            str += ldsOff;
                            str += ", ";
                            str += ldsArgs; str += ", ";
                            str += ldsArgs; str += IterRegs("&"); str += ");\n";
                            if (!halfLds) { str += exTab; str += "\tbarrier(CLK_LOCAL_MEM_FENCE);\n"; }
                        }
                    }
                }

                if (realSpecial)
                {
                    size_t Nt = 1 + length / 2;
                    str += "\n\t\tif( (bt == 0) || (2*bt == ";
                    str += SztToStr(params.fft_realSpecial_Nr); str += ") ) { rw = 0; }\n";

                    str += "\t\tlwbOut += ("; str += SztToStr(params.fft_realSpecial_Nr);
                    str += " - 2*bt)*"; str += SztToStr(Nt); str += ";\n";
                    str += "\t\tb = "; str += SztToStr(params.fft_realSpecial_Nr);
                    str += " - b;\n\n";
                }

                if (blockCompute || realSpecial)
                {
                    str += "\n\t}\n\n";
                }



                // Write data from LDS for blocked access
                if (blockCompute)
                {

                    size_t loopCount = (length * blockWidth) / blockWGS;

                    str += "\tbarrier(CLK_LOCAL_MEM_FENCE);\n\n";
                    str += "\n\tfor(uint t=0; t<"; str += SztToStr(loopCount);
                    str += "; t++)\n\t{\n";

                    if ((blockComputeType == BCT_C2C) || (blockComputeType == BCT_R2C))
                    {
                        str += "\t\tR0 = lds[t*"; str += SztToStr(blockWGS / blockWidth); str += " + ";
                        str += "(me%"; str += SztToStr(blockWidth); str += ")*"; str += SztToStr(length); str += " + ";
                        str += "(me/"; str += SztToStr(blockWidth); str += ")];"; str += "\n";
                    }
                    else
                    {
                        str += "\t\tR0 = lds[t*"; str += SztToStr(blockWGS); str += " + me];"; str += "\n";
                    }

                    for (size_t c = 0; c<2; c++)
                    {
                        std::string comp = "";
                        std::string writeBuf = (params.fft_placeness == CLFFT_INPLACE) ? "lwb" : "lwbOut";
                        if (!outInterleaved) comp = c ? ".y" : ".x";
                        if (!outInterleaved)
                            writeBuf = (params.fft_placeness == CLFFT_INPLACE) ? (c ? "lwbIm" : "lwbRe") : (c ? "lwbOutIm" : "lwbOutRe");

                        if ((blockComputeType == BCT_C2C) || (blockComputeType == BCT_R2C))
                        {
                            if (blockComputeType == BCT_R2C && params.fft_hasPostCallback)
                            {
                                if (outInterleaved)
                                    writeBuf = (params.fft_placeness == CLFFT_INPLACE) ? "gb" : "gbOut";
                                else
                                    writeBuf = (params.fft_placeness == CLFFT_INPLACE) ? "gbRe, gbIm" : "gbOutRe, gbOutIm";

                                str += "\t\t"; str += params.fft_postCallback.funcname; str += "("; str += writeBuf; str += ", (";
                                str += outOffset; str += " + (me%"; str += SztToStr(blockWidth); str += ") + ";
                                str += "(me/"; str += SztToStr(blockWidth); str += ")*"; str += SztToStr(params.fft_outStride[0]);
                                str += " + t*"; str += SztToStr(params.fft_outStride[0] * blockWGS / blockWidth);
                                str += "), post_userdata, R0";
                                if (!outInterleaved) str += ".x, R0.y";

                                if (params.fft_postCallback.localMemSize > 0)
                                {
                                    if (params.fft_hasPreCallback && params.fft_preCallback.localMemSize > 0)
                                    {
                                        str += ", (char *)(localmem + "; str += SztToStr(params.fft_preCallback.localMemSize); str += ")";
                                    }
                                    else
                                    {
                                        str += ", localmem";
                                    }
                                }
                                str += ");\n";

                                //in the planar case, break from for loop since both real and imag components are handled
                                //together in post-callback
                                if (!outInterleaved) break;
                            }
                            else
                            {
                                str += "\t\t"; str += writeBuf; str += "[(me%"; str += SztToStr(blockWidth); str += ") + ";
                                str += "(me/"; str += SztToStr(blockWidth); str += ")*"; str += SztToStr(params.fft_outStride[0]);
                                str += " + t*"; str += SztToStr(params.fft_outStride[0] * blockWGS / blockWidth); str += "] = R0"; str += comp; str += ";\n";
                            }
                        }
                        else
                        {
                            str += "\t\t"; str += writeBuf; str += "[me + t*"; str += SztToStr(blockWGS); str += "] = R0"; str += comp; str += ";\n";
                        }

                        if (outInterleaved) break;
                    }

                    str += "\t}\n\n";
                }



                str += "}\n\n";

                if (r2c2r)
                    break;
            }
        }
    };

};

#endif

