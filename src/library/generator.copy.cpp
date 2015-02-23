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
		bool general;

		inline std::string OffsetCalc(const std::string &off, bool input = true)
		{
			std::string str;

			const size_t *pStride = input ? params.fft_inStride : params.fft_outStride;

			str += "\t"; str += off; str += " = ";
			std::string nextBatch = "batch";
			for(size_t i=(params.fft_DataDim - 1); i>1; i--)
			{
				size_t currentLength = 1;
				for(int j=1; j<i; j++) currentLength *= params.fft_N[j];

				str += "("; str += nextBatch; str += "/"; str += SztToStr(currentLength);
				str += ")*"; str += SztToStr(pStride[i]); str += " + ";

				nextBatch = "(" + nextBatch + "%" + SztToStr(currentLength) + ")";
			}

			str += nextBatch; str += "*"; str += SztToStr(pStride[1]); str += ";\n";

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

			general = !(h2c || c2h);

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
			if(general)
					str += "copy_general";
			else
			{
				if(h2c)	str += "copy_h2c";
				else	str += "copy_c2h";
			}

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
			if(general)
			{
				str += "\tuint me = get_local_id(0);\n\t";
				str += "uint batch = get_group_id(0);\n\t";
			}
			else
			{
				str += "\tuint me = get_global_id(0);\n\t";
			}

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

			if(!general)
			{
				// Setup variables
				str += "\tuint batch, mel, mel2;\n\t";
				str += "batch = me/"; str += SztToStr(Nt); str += ";\n\t";
				str += "mel = me%"; str += SztToStr(Nt); str += ";\n\t";
				str += "mel2 = ("; str += SztToStr(N); str += " - mel)%"; str += SztToStr(N); str += ";\n\n";
			}


			// Setup memory pointers
			str += OffsetCalc("iOffset", true);
			str += OffsetCalc("oOffset", false);

			// offset strings
			std::string inF, inF2, outF, outF2;
			if(general)
			{
				inF = inF2 = outF = outF2 = "";
			}
			else
			{
				inF   = " + (mel*";  inF   += SztToStr(params.fft_inStride[0]);  inF   += ")";
				inF2  = " + (mel2*"; inF2  += SztToStr(params.fft_inStride[0]);  inF2  += ")";
				outF  = " + (mel*";  outF  += SztToStr(params.fft_outStride[0]); outF  += ")";
				outF2 = " + (mel2*"; outF2 += SztToStr(params.fft_outStride[0]); outF2 += ")";
			}

			str += "\n\t";

			// inputs
			if(inIlvd)
			{
				str += "lwbIn = gbIn + iOffset"; str += inF; str += ";\n\t";
			}
			else
			{
				str += "lwbInRe = gbInRe + iOffset"; str += inF; str += ";\n\t";
				str += "lwbInIm = gbInIm + iOffset"; str += inF; str += ";\n\t";
			}

			// outputs
			if(outIlvd)
			{
					str += "lwbOut = gbOut + oOffset"; str += outF; str += ";\n";
				if(h2c)
				{
					str += "\t";
					str += "lwbOut2 = gbOut + oOffset"; str += outF2; str += ";\n";
				}
			}
			else
			{
					str += "lwbOutRe = gbOutRe + oOffset"; str += outF; str += ";\n\t";
					str += "lwbOutIm = gbOutIm + oOffset"; str += outF; str += ";\n";
				if(h2c)
				{
					str += "\t";
					str += "lwbOutRe2 = gbOutRe + oOffset"; str += outF2; str += ";\n\t";
					str += "lwbOutIm2 = gbOutIm + oOffset"; str += outF2; str += ";\n";
				}
			}

			str += "\n\t";

			// Do the copy
			if(general)
			{
				str += "for(uint t=0; t<"; str += SztToStr(N/64); str += "; t++)\n\t{\n\t\t";
				
				if(inIlvd)
				{
					str += "R = lwbIn[me + t*64];\n\t\t";
				}
				else
				{
					str += "R.x = lwbInRe[me + t*64];\n\t\t";
					str += "R.y = lwbInIm[me + t*64];\n\t\t";
				}

				if(outIlvd)
				{
					str += "lwbOut[me + t*64] = R;\n";
				}
				else
				{
					str += "lwbOutRe[me + t*64] = R.x;\n\t\t";
					str += "lwbOutIm[me + t*64] = R.y;\n";
				}

				str += "\t}\n\n";
			}
			else
			{
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

	params.fft_DataDim = this->length.size() + 1;
	int i = 0;
	for(i = 0; i < (params.fft_DataDim - 1); i++)
	{
        params.fft_N[i]         = this->length[i];
        params.fft_inStride[i]  = this->inStride[i];
        params.fft_outStride[i] = this->outStride[i];

	}
    params.fft_inStride[i]  = this->iDist;
    params.fft_outStride[i] = this->oDist;

    params.fft_fwdScale  = this->forwardScale;
    params.fft_backScale = this->backwardScale;

    return CLFFT_SUCCESS;
}

template<>
clfftStatus FFTPlan::GetWorkSizesPvt<Copy> (std::vector<size_t> & globalWS, std::vector<size_t> & localWS) const
{
    FFTKernelGenKeyParams fftParams;
	OPENCL_V( this->GetKernelGenKeyPvt<Copy>( fftParams ), _T("GetKernelGenKey() failed!") );

	bool h2c, c2h;
	h2c = (	(fftParams.fft_inputLayout == CLFFT_HERMITIAN_PLANAR) ||
			(fftParams.fft_inputLayout == CLFFT_HERMITIAN_INTERLEAVED) ) ? true : false;
	c2h = (	(fftParams.fft_outputLayout == CLFFT_HERMITIAN_PLANAR) ||
			(fftParams.fft_outputLayout == CLFFT_HERMITIAN_INTERLEAVED) ) ? true : false;

	bool general = !(h2c || c2h);

	size_t count = this->batchsize;

	switch(fftParams.fft_DataDim)
	{
	case 5: assert(false);
	case 4: count *= fftParams.fft_N[2];
	case 3: count *= fftParams.fft_N[1];
	case 2:
			{
				if(general)
				{
					count *= 64;
				}
				else
				{
					count *= (1 + fftParams.fft_N[0]/2); 
				}
			}
			break;
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
clfftStatus FFTPlan::GenerateKernelPvt<Copy>(FFTRepo& fftRepo, const cl_command_queue& commQueueFFT ) const
{
  FFTKernelGenKeyParams params;
  OPENCL_V( this->GetKernelGenKeyPvt<Copy> (params), _T("GetKernelGenKey() failed!") );

  bool h2c, c2h;
  h2c = (	(params.fft_inputLayout == CLFFT_HERMITIAN_PLANAR) ||
  			(params.fft_inputLayout == CLFFT_HERMITIAN_INTERLEAVED) ) ? true : false;
  c2h = (	(params.fft_outputLayout == CLFFT_HERMITIAN_PLANAR) ||
  			(params.fft_outputLayout == CLFFT_HERMITIAN_INTERLEAVED) ) ? true : false;
  
  bool general = !(h2c || c2h);

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
	cl_device_id Device = NULL;
	status = clGetCommandQueueInfo(commQueueFFT, CL_QUEUE_DEVICE, sizeof(cl_device_id), &Device, NULL);
	OPENCL_V( status, _T( "clGetCommandQueueInfo failed" ) );

    cl_context QueueContext = NULL;
    status = clGetCommandQueueInfo(commQueueFFT, CL_QUEUE_CONTEXT, sizeof(cl_context), &QueueContext, NULL);
    OPENCL_V( status, _T( "clGetCommandQueueInfo failed" ) );

  OPENCL_V( fftRepo.setProgramCode( Copy, params, programCode, Device, QueueContext ), _T( "fftRepo.setclString() failed!" ) );

  if(general)
  {
  OPENCL_V( fftRepo.setProgramEntryPoints( Copy, params, "copy_general", "copy_general", Device, QueueContext ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );
  }
  else
  {
  OPENCL_V( fftRepo.setProgramEntryPoints( Copy, params, "copy_c2h", "copy_h2c", Device, QueueContext ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );
  }

  return CLFFT_SUCCESS;
}
