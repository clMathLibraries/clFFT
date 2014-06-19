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


#include "stdafx.h"
#include <math.h>
#include <list>
#include "generator.stockham.h"

using namespace StockhamGenerator;

namespace CopyGenerator
{
    // Copy kernel
    template <Precision PR>
    class CopyKernel
    {
        size_t N;
		size_t Nt;
		const FFTKernelGenKeyParams params;
		bool h2c, c2h;

		inline std::string OffsetCalc(const std::string &off, bool input = true)
		{
			std::string str;

			const size_t *pStride = input ? params.fft_inStride : params.fft_outStride;

			std::string batch = "batch";

			switch(params.fft_DataDim)
			{
			case 5:
				{
					str += "\t{\n\tuint ocalc1 = ";
					str += batch; str += "%"; str += SztToStr(params.fft_N[1] * params.fft_N[2] * params.fft_N[3]);
					str += ";\n";

					str += "\tuint ocalc0 = ";
					str += "ocalc1"; str += "%"; str += SztToStr(params.fft_N[1] * params.fft_N[2]);
					str += ";\n";

					str += "\t"; str += off; str += " = ";
					str += "("; str += batch; str += "/"; str += SztToStr(params.fft_N[1] * params.fft_N[2] * params.fft_N[3]);
					str += ")*"; str += SztToStr(pStride[4]); str += " + ";

					str += "(ocalc1"; str += "/"; str += SztToStr(params.fft_N[1] * params.fft_N[2]); str += ")*";
					str += SztToStr(pStride[3]); str += " + ";

					str += "(ocalc0"; str += "/"; str += SztToStr(params.fft_N[1]); str += ")*";
					str += SztToStr(pStride[2]); str += " + ";
					str += "(ocalc0"; str += "%"; str += SztToStr(params.fft_N[1]); str += ")*";
					str += SztToStr(pStride[1]); str += ";\n";

					str += "\t}\n";
				}
				break;
			case 4:
				{
					str += "\t{\n\tuint ocalc0 = ";
					str += batch; str += "%"; str += SztToStr(params.fft_N[1] * params.fft_N[2]);
					str += ";\n";

					str += "\t"; str += off; str += " = ";
					str += "("; str += batch; str += "/"; str += SztToStr(params.fft_N[1] * params.fft_N[2]); str += ")*";
					str += SztToStr(pStride[3]); str += " + ";

					str += "(ocalc0"; str += "/"; str += SztToStr(params.fft_N[1]); str += ")*";
					str += SztToStr(pStride[2]); str += " + ";
					str += "(ocalc0"; str += "%"; str += SztToStr(params.fft_N[1]); str += ")*";
					str += SztToStr(pStride[1]); str += ";\n";

					str += "\t}\n";
				}
				break;
			case 3:
				{
					str += "\t"; str += off; str += " = ";
					str += "("; str += batch; str += "/"; str += SztToStr(params.fft_N[1]); str += ")*";
					str += SztToStr(pStride[2]); str += " + ";
					str += "("; str += batch; str += "%"; str += SztToStr(params.fft_N[1]); str += ")*";
					str += SztToStr(pStride[1]); str += ";\n";
				}
				break;
			case 2:
				{
					str += "\t"; str += off; str += " = ";
					str += batch; str += "*"; str += SztToStr(pStride[1]); str += ";\n";
				}
				break;
			default:
				assert(false);
			}

			return str;
		}

    public:
        CopyKernel( const FFTKernelGenKeyParams &paramsVal) :
					params(paramsVal)

        {
			N = params.fft_N[0];
			Nt = 1 + N/2;

			h2c = (	(params.fft_inputLayout == CLFFT_HERMITIAN_PLANAR) ||
					(params.fft_inputLayout == CLFFT_HERMITIAN_INTERLEAVED) ) ? true : false;
			c2h = (	(params.fft_outputLayout == CLFFT_HERMITIAN_PLANAR) ||
					(params.fft_outputLayout == CLFFT_HERMITIAN_INTERLEAVED) ) ? true : false;

			// We only do out-of-place copies at this point
			assert(params.fft_placeness == CLFFT_OUTOFPLACE);
		}

        void GenerateKernel(std::string &str)
		{
			std::string rType  = RegBaseType<PR>(1);
			std::string r2Type  = RegBaseType<PR>(2);

			bool inIlvd; // Input is interleaved format
			bool outIlvd; // Output is interleaved format
			inIlvd  = (	(params.fft_inputLayout == CLFFT_COMPLEX_INTERLEAVED) ||
						(params.fft_inputLayout == CLFFT_HERMITIAN_INTERLEAVED) ) ? true : false;
			outIlvd = (	(params.fft_outputLayout == CLFFT_COMPLEX_INTERLEAVED) ||
						(params.fft_outputLayout == CLFFT_HERMITIAN_INTERLEAVED) ) ? true : false;



			// Pragma
			str += ClPragma<PR>();

			std::string sfx = FloatSuffix<PR>();

			// Copy kernel begin
			str += "__kernel void ";

			// Function name
			if(h2c)	str += "copy_h2c";
			else	str += "copy_c2h";

			str += "(";

			if(inIlvd)
			{
				str += "__global const "; str += r2Type; str += " * restrict gbIn, ";
			}
			else
			{
				str += "__global const "; str += rType; str += " * restrict gbInRe, ";
				str += "__global const "; str += rType; str += " * restrict gbInIm, ";
			}

			if(outIlvd)
			{
				str += "__global "; str += r2Type; str += " * restrict gbOut)\n";
			}
			else
			{
				str += "__global "; str += rType; str += " * restrict gbOutRe, ";
				str += "__global "; str += rType; str += " * restrict gbOutIm)\n";
			}


			str += "{\n";

			// Initialize
			str += "\tuint me = get_global_id(0);\n\t";

			// Declare memory pointers
			str += "\n\t";
			str += "uint iOffset;\n\t";
			str += "uint oOffset;\n\t";

			// input
			if(inIlvd)
			{
				str += "__global "; str += r2Type; str += " *lwbIn;\n\t";
			}
			else
			{
				str += "__global "; str += rType; str += " *lwbInRe;\n\t";
				str += "__global "; str += rType; str += " *lwbInIm;\n\t";
			}

			// output
			if(outIlvd)
			{
					str += "__global "; str += r2Type; str += " *lwbOut;\n";
				if(h2c)
				{
					str += "\t";
					str += "__global "; str += r2Type; str += " *lwbOut2;\n\n";
				}
			}
			else
			{
					str += "__global "; str += rType; str += " *lwbOutRe;\n\t";
					str += "__global "; str += rType; str += " *lwbOutIm;\n";
				if(h2c)
				{
					str += "\t";
					str += "__global "; str += rType; str += " *lwbOutRe2;\n\t";
					str += "__global "; str += rType; str += " *lwbOutIm2;\n\n";
				}
			}



			// Setup registers
			str += "\t"; str += RegBaseType<PR>(2); str += " R;\n\n";

			// Setup variables
			str += "\tuint batch, mel, mel2;\n\t";
			str += "batch = me/"; str += SztToStr(Nt); str += ";\n\t";
			str += "mel = me%"; str += SztToStr(Nt); str += ";\n\t";
			str += "mel2 = ("; str += SztToStr(N); str += " - mel)%"; str += SztToStr(N); str += ";\n\n";


			// Setup memory pointers
			str += OffsetCalc("iOffset", true);
			str += OffsetCalc("oOffset", false);

			// offset strings
			std::string inF, inF2, outF, outF2;
			inF   = "(mel*";  inF   += SztToStr(params.fft_inStride[0]);  inF   += ")";
			inF2  = "(mel2*"; inF2  += SztToStr(params.fft_inStride[0]);  inF2  += ")";
			outF  = "(mel*";  outF  += SztToStr(params.fft_outStride[0]); outF  += ")";
			outF2 = "(mel2*"; outF2 += SztToStr(params.fft_outStride[0]); outF2 += ")";

			str += "\n\t";

			// inputs
			if(inIlvd)
			{
				str += "lwbIn = gbIn + iOffset + "; str += inF; str += ";\n\t";
			}
			else
			{
				str += "lwbInRe = gbInRe + iOffset + "; str += inF; str += ";\n\t";
				str += "lwbInIm = gbInIm + iOffset + "; str += inF; str += ";\n\t";
			}

			// outputs
			if(outIlvd)
			{
					str += "lwbOut = gbOut + oOffset + "; str += outF; str += ";\n";
				if(h2c)
				{
					str += "\t";
					str += "lwbOut2 = gbOut + oOffset + "; str += outF2; str += ";\n";
				}
			}
			else
			{
					str += "lwbOutRe = gbOutRe + oOffset + "; str += outF; str += ";\n\t";
					str += "lwbOutIm = gbOutIm + oOffset + "; str += outF; str += ";\n";
				if(h2c)
				{
					str += "\t";
					str += "lwbOutRe2 = gbOutRe + oOffset + "; str += outF2; str += ";\n\t";
					str += "lwbOutIm2 = gbOutIm + oOffset + "; str += outF2; str += ";\n";
				}
			}

			str += "\n\t";

			// Do the copy
			if(c2h)
			{
				if(inIlvd)
				{
					str += "R = lwbIn[0];\n\t";
				}
				else
				{
					str += "R.x = lwbInRe[0];\n\t";
					str += "R.y = lwbInIm[0];\n\t";
				}

				if(outIlvd)
				{
					str += "lwbOut[0] = R;\n\n";
				}
				else
				{
					str += "lwbOutRe[0] = R.x;\n\t";
					str += "lwbOutIm[0] = R.y;\n\t";
				}
			}
			else
			{
				if(inIlvd)
				{
					str += "R = lwbIn[0];\n\t";
				}
				else
				{
					str += "R.x = lwbInRe[0];\n\t";
					str += "R.y = lwbInIm[0];\n\t";
				}

				if(outIlvd)
				{
					str += "lwbOut[0] = R;\n\t";
					str += "R.y = -R.y;\n\t";
					str += "lwbOut2[0] = R;\n\n";
				}
				else
				{
					str += "lwbOutRe[0] = R.x;\n\t";
					str += "lwbOutIm[0] = R.y;\n\t";
					str += "R.y = -R.y;\n\t";
					str += "lwbOutRe2[0] = R.x;\n\t";
					str += "lwbOutIm2[0] = R.y;\n\n";
				}
			}

			str += "}\n";
		}
    };
};


template<>
clfftStatus FFTPlan::GetKernelGenKeyPvt<Copy> (FFTKernelGenKeyParams & params) const
{

    //    Query the devices in this context for their local memory sizes
    //    How we generate a kernel depends on the *minimum* LDS size for all devices.
    //
    const FFTEnvelope * pEnvelope = NULL;
    OPENCL_V(const_cast<FFTPlan*>(this)->GetEnvelope (& pEnvelope), _T("GetEnvelope failed"));
    BUG_CHECK (NULL != pEnvelope);

    ::memset( &params, 0, sizeof( params ) );
    params.fft_precision    = this->precision;
    params.fft_placeness    = this->placeness;
    params.fft_inputLayout  = this->inputLayout;
	params.fft_MaxWorkGroupSize = this->envelope.limit_WorkGroupSize;

    ARG_CHECK (this->inStride.size() == this->outStride.size())

    params.fft_outputLayout = this->outputLayout;

    switch (this->inStride.size()) {
        //    1-D array is a 2-D data structure.
        //    1-D unit is a special case of 1-D array.
    case 1:
        ARG_CHECK(this->length   .size() > 0);
        ARG_CHECK(this->outStride.size() > 0);
        params.fft_DataDim      = 2;
        params.fft_N[0]         = this->length[0];
        params.fft_inStride[0]  = this->inStride[0];
        params.fft_inStride[1]  = this->iDist;
        params.fft_outStride[0] = this->outStride[0];
        params.fft_outStride[1] = this->oDist;
        break;

        //    2-D array is a 3-D data structure
        //    2-D unit is a speical case of 2-D array.
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

        //    3-D array is a 4-D data structure
        //    3-D unit is a special case of 3-D array.
    case 3:
        ARG_CHECK(this->length   .size() > 2);
        ARG_CHECK(this->outStride.size() > 2);
        params.fft_DataDim      = 4;
        params.fft_N[0]         = this->length[0];
        params.fft_N[1]         = this->length[1];
        params.fft_N[2]         = this->length[2];
        params.fft_inStride[0]  = this->inStride[0];
        params.fft_inStride[1]  = this->inStride[1];
        params.fft_inStride[2]  = this->inStride[2];
        params.fft_inStride[3]  = this->iDist;
        params.fft_outStride[0] = this->outStride[0];
        params.fft_outStride[1] = this->outStride[1];
        params.fft_outStride[2] = this->outStride[2];
        params.fft_outStride[3] = this->oDist;
        break;

    default:
        ARG_CHECK (false);
    }

    params.fft_fwdScale  = this->forwardScale;
    params.fft_backScale = this->backwardScale;

    return CLFFT_SUCCESS;
}

template<>
clfftStatus FFTPlan::GetWorkSizesPvt<Copy> (std::vector<size_t> & globalWS, std::vector<size_t> & localWS) const
{
    FFTKernelGenKeyParams fftParams;
	OPENCL_V( this->GetKernelGenKeyPvt<Copy>( fftParams ), _T("GetKernelGenKey() failed!") );

	size_t count = this->batchsize;
	switch(fftParams.fft_DataDim)
	{
	case 5: assert(false);
	case 4: count *= fftParams.fft_N[2];
	case 3: count *= fftParams.fft_N[1];
	case 2: count *= (1 + fftParams.fft_N[0]/2); break;
	case 1: assert(false);
	}

	globalWS.push_back( count );
    localWS.push_back( 64 );

    return    CLFFT_SUCCESS;
}

template<>
clfftStatus FFTPlan::GetMax1DLengthPvt<Copy> (size_t * longest) const
{
	return FFTPlan::GetMax1DLengthPvt<Stockham>(longest);
}

using namespace CopyGenerator;

template<>
clfftStatus FFTPlan::GenerateKernelPvt<Copy>(FFTRepo& fftRepo, const cl_command_queue commQueueFFT ) const
{
  FFTKernelGenKeyParams params;
  OPENCL_V( this->GetKernelGenKeyPvt<Copy> (params), _T("GetKernelGenKey() failed!") );

  std::string programCode;
  Precision pr = (params.fft_precision == CLFFT_SINGLE) ? P_SINGLE : P_DOUBLE;
  switch(pr)
  {
  case P_SINGLE:
    {
      CopyKernel<P_SINGLE> kernel(params);
      kernel.GenerateKernel(programCode);
    } break;
  case P_DOUBLE:
    {
      CopyKernel<P_DOUBLE> kernel(params);
      kernel.GenerateKernel(programCode);
    } break;
  }

  cl_int status = CL_SUCCESS;
  cl_context QueueContext = NULL;
  status = clGetCommandQueueInfo(commQueueFFT, CL_QUEUE_CONTEXT, sizeof(cl_context), &QueueContext, NULL);

  OPENCL_V( status, _T( "clGetCommandQueueInfo failed" ) );

  OPENCL_V( fftRepo.setProgramCode( Copy, params, programCode, QueueContext ), _T( "fftRepo.setclString() failed!" ) );
  OPENCL_V( fftRepo.setProgramEntryPoints( Copy, params, "copy_c2h", "copy_h2c", QueueContext ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );

  return CLFFT_SUCCESS;
}
