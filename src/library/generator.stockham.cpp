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
#include "action.h"
#include "generator.stockham.h"
#include "generator.twiddle.hpp"
#include "generator.butterfly.hpp"
#include "generator.pass.hpp"
#include "generator.kernel.hpp"


/* Classes used for generating clFFT kernels are in StockhamGenerator namespace. See the above included 4 *.hpp files */

namespace StockhamGenerator
{
	// Experimnetal Start =========================================
	// Kernel Generator Parameterization ==========================

	// Uncomment this directive to activate parameter reads from file
//#define PARMETERS_TO_BE_READ

	// Parameters to read
	struct ParamRead
	{
		size_t	workGroupSize;
		size_t	numTransformsPerWg;
		std::vector<size_t> radices;
		bool	halfLds;
	};

	// File format

	// WorkGroupSize:
	// TransformsPerWorkGroup:
	// Radices:
	// LdsUse:

	void ReadParameterFile(ParamRead &readParam)
	{
		const char *fileName = "parameters.txt";
		std::ifstream file(fileName);

		if(!file.is_open())
		{
			std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
			exit(-1);
		}

		std::string strWgs = "WorkGroupSize:";
		std::string strNtw = "TransformsPerWorkGroup:";
		std::string strRad = "Radices:";
		std::string strLds = "LdsUse:";
		std::string numbers = "0123456789";

		std::string line;
		while(std::getline(file, line))
		{

			size_t pos;

			pos = line.find(strWgs);
			if(pos != std::string::npos)
			{
				line.erase(pos, strWgs.length());
				size_t numStart	= line.find_first_of(numbers);
				size_t numEnd	= line.find_first_not_of(numbers, numStart);
				std::string val = line.substr(numStart, numEnd-numStart);
				readParam.workGroupSize = strtol(val.c_str(), NULL, 10);
				continue;
			}

			pos = line.find(strNtw);
			if(pos != std::string::npos)
			{
				line.erase(pos, strNtw.length());
				size_t numStart	= line.find_first_of(numbers);
				size_t numEnd	= line.find_first_not_of(numbers, numStart);
				std::string val = line.substr(numStart, numEnd-numStart);
				readParam.numTransformsPerWg = strtol(val.c_str(), NULL, 10);
				continue;
			}

			pos = line.find(strRad);
			if(pos != std::string::npos)
			{
				line.erase(pos, strRad.length());
				while(std::string::npos != line.find_first_of(numbers))
				{
					size_t numStart	= line.find_first_of(numbers);
					size_t numEnd	= line.find_first_not_of(numbers, numStart);
					std::string val = line.substr(numStart, numEnd-numStart);
					readParam.radices.push_back(strtol(val.c_str(), NULL, 10));
					line.erase(0, numEnd);
				}
				continue;
			}
		}

		//std::cout << std::endl;
		//std::cout << "File Parameters" << std::endl;
		//std::cout << strWgs << " " << readParam.workGroupSize << std::endl;
		//std::cout << strNtw << " " << readParam.numTransformsPerWg << std::endl;
		//std::cout << strRad << " "; for(size_t i=0; i<readParam.radices.size(); i++) std::cout << readParam.radices[i] << " "; std::cout << std::endl;
		//std::cout << std::endl;
	}

	// Uncomment this directive to experiment with kernels read from file
//#define KERNEL_INTERJECT

	void ReadKernelFromFile(std::string &str)
	{
		const char *fileName = "fft_kernel.cl";
		std::ifstream file(fileName);

		if(!file.is_open())
		{
			std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
			exit(-1);
		}

		str.clear();

		std::string line;
		while(std::getline(file, line))
		{
			str += line;
			str += '\n';
		}
	}

	// Experimental End ===========================================


	// Given the length of 1d fft, this function determines the appropriate work group size
	// and the number of transforms per work group
	// TODO for optimizations - experiment with different possibilities for work group sizes and num transforms for improving performance
	void DetermineSizes(const size_t &MAX_WGS, const size_t &length, size_t &workGroupSize, size_t &numTrans, Precision &pr)
	{
		assert(MAX_WGS >= 64);

		if(length == 1) // special case
		{
			workGroupSize = 64;
			numTrans = 64;
			return;
		}

		size_t baseRadix[] = {13,11,7,5,3,2}; // list only supported primes
		size_t baseRadixSize = sizeof(baseRadix)/sizeof(baseRadix[0]);

		size_t l = length;
		std::map<size_t, size_t> primeFactorsExpanded;
		for(size_t r=0; r<baseRadixSize; r++)
		{
			size_t rad = baseRadix[r];
			size_t e = 1;
			while(!(l%rad))
			{
				l /= rad;
				e *= rad;
			}

			primeFactorsExpanded[rad] = e;
		}

		assert(l == 1); // Makes sure the number is composed of only supported primes

		if		(primeFactorsExpanded[2] == length)	// Length is pure power of 2
		{
			if		(length >= 1024)	{ workGroupSize = (MAX_WGS >= 256) ? 256 : MAX_WGS; numTrans = 1; }
			else if (length == 512)		{ workGroupSize = 64; numTrans = 1; }
			else if	(length >= 16)		{ workGroupSize = 64;  numTrans = 256/length; }
			else						{ workGroupSize = 64;  numTrans = 128/length; }
		}
		else if	(primeFactorsExpanded[3] == length) // Length is pure power of 3
		{
			workGroupSize = (MAX_WGS >= 256) ? 243 : 27;
			numTrans = length >= 3*workGroupSize ? 1 : (3*workGroupSize)/length;
		}
		else if	(primeFactorsExpanded[5] == length) // Length is pure power of 5
		{
			workGroupSize = (MAX_WGS >= 128) ? 125 : 25;
			numTrans = length >= 5*workGroupSize ? 1 : (5*workGroupSize)/length;
		}
		else if	(primeFactorsExpanded[7] == length) // Length is pure power of 7
		{
			workGroupSize = 49;
			numTrans = length >= 7*workGroupSize ? 1 : (7*workGroupSize)/length;
		}
		else if (primeFactorsExpanded[11] == length) // Length is pure power of 11
		{
			workGroupSize = 121;
			numTrans = length >= 11 * workGroupSize ? 1 : (11 * workGroupSize) / length;
		}
		else if (primeFactorsExpanded[13] == length) // Length is pure power of 13
		{
			workGroupSize = 169;
			numTrans = length >= 13 * workGroupSize ? 1 : (13 * workGroupSize) / length;
		}
		else
		{
			size_t leastNumPerWI = 1; // least number of elements in one work item
			size_t maxWorkGroupSize = MAX_WGS; // maximum work group size desired


			if        (primeFactorsExpanded[2] * primeFactorsExpanded[3] == length) { 
				if (length % 12 == 0) { 
					leastNumPerWI = 12; maxWorkGroupSize = 128;
				} else { 
					leastNumPerWI =  6; maxWorkGroupSize = 256;
				}
			} else if (primeFactorsExpanded[2] * primeFactorsExpanded[5] == length) { 
				if (length % 20 == 0) { 
					leastNumPerWI = 20; maxWorkGroupSize = 64;
				} else { 
					leastNumPerWI = 10; maxWorkGroupSize = 128;
				}
			} else if (primeFactorsExpanded[2] * primeFactorsExpanded[7] == length) { 
					leastNumPerWI = 14; maxWorkGroupSize = 64;
			} else if (primeFactorsExpanded[3] * primeFactorsExpanded[5] == length) { 
				    leastNumPerWI = 15; maxWorkGroupSize = 128;
			} else if (primeFactorsExpanded[3] * primeFactorsExpanded[7] == length) { 
				    leastNumPerWI = 21; maxWorkGroupSize = 128;
			} else if (primeFactorsExpanded[5] * primeFactorsExpanded[7] == length) { 
				    leastNumPerWI = 35; maxWorkGroupSize = 64;
			} else if (primeFactorsExpanded[2] * primeFactorsExpanded[3] * primeFactorsExpanded[5] == length) { 
				    leastNumPerWI = 30; maxWorkGroupSize = 64;
			} else if (primeFactorsExpanded[2] * primeFactorsExpanded[3] * primeFactorsExpanded[7] == length) { 
				    leastNumPerWI = 42; maxWorkGroupSize = 60;
			} else if (primeFactorsExpanded[2] * primeFactorsExpanded[5] * primeFactorsExpanded[7] == length) { 
				    leastNumPerWI = 70; maxWorkGroupSize = 36;
			} else if (primeFactorsExpanded[3] * primeFactorsExpanded[5] * primeFactorsExpanded[7] == length) { 
				    leastNumPerWI =105; maxWorkGroupSize = 24;
			}
			else if (primeFactorsExpanded[2] * primeFactorsExpanded[11] == length) {
				leastNumPerWI = 22; maxWorkGroupSize = 128;
			}
			else if (primeFactorsExpanded[2] * primeFactorsExpanded[13] == length) {
				leastNumPerWI = 26; maxWorkGroupSize = 128;
			}
			else {
				    leastNumPerWI =210; maxWorkGroupSize = 12;
			}
			if (pr==P_DOUBLE)
			{
				//leastNumPerWI /= 2; 
				maxWorkGroupSize /= 2;
			}
			

			if (maxWorkGroupSize > MAX_WGS)
				maxWorkGroupSize = MAX_WGS;
			assert (leastNumPerWI > 0 && length % leastNumPerWI == 0);

			for (size_t lnpi = leastNumPerWI; lnpi <= length; lnpi += leastNumPerWI) {
				if (length % lnpi != 0) continue;

				if (length / lnpi <= MAX_WGS) {
					leastNumPerWI = lnpi;
					break;
				}
			}

			numTrans = maxWorkGroupSize / (length / leastNumPerWI);
			numTrans = numTrans < 1 ? 1 : numTrans;
			workGroupSize = numTrans * (length / leastNumPerWI);
		}

		assert(workGroupSize <= MAX_WGS);
	}


};


/*=====================The below is instantiation of StockhamGenerator classes==============================*/


using namespace StockhamGenerator;

FFTGeneratedStockhamAction::FFTGeneratedStockhamAction(clfftPlanHandle plHandle, FFTPlan * plan, cl_command_queue queue, clfftStatus & err)
    : FFTStockhamAction(plHandle, plan, queue, err)
{
    if (err != CLFFT_SUCCESS)
    {
        // FFTAction() failed, exit
        fprintf(stderr, "FFTStockhamAction() failed!\n");
        return;
    }

    // Initialize the FFTAction::FFTKernelGenKeyParams member
    err = this->initParams();

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedStockhamAction::initParams() failed!\n");
        return;
    }

    FFTRepo &fftRepo = FFTRepo::getInstance();

    err = this->generateKernel(fftRepo, queue);

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedStockhamAction::generateKernel failed\n");
        return;
    }

    err = compileKernels(queue, plHandle, plan);

    if (err != CLFFT_SUCCESS)
    {
        fprintf(stderr, "FFTGeneratedStockhamAction::compileKernels failed\n");
        return;
    }

    err = CLFFT_SUCCESS;
}

bool FFTGeneratedStockhamAction::buildForwardKernel()
{
    clfftLayout inputLayout = this->getSignatureData()->fft_inputLayout;
    clfftLayout outputLayout = this->getSignatureData()->fft_outputLayout;

    bool r2c_transform = (inputLayout == CLFFT_REAL);
    bool c2r_transform = (outputLayout == CLFFT_REAL);
    bool real_transform = (r2c_transform || c2r_transform);

    return (!real_transform) || r2c_transform;
}

bool FFTGeneratedStockhamAction::buildBackwardKernel()
{
    clfftLayout inputLayout = this->getSignatureData()->fft_inputLayout;
    clfftLayout outputLayout = this->getSignatureData()->fft_outputLayout;

    bool r2c_transform = (inputLayout == CLFFT_REAL);
    bool c2r_transform = (outputLayout == CLFFT_REAL);
    bool real_transform = (r2c_transform || c2r_transform);

    return (!real_transform) || c2r_transform;
}

clfftStatus FFTGeneratedStockhamAction::initParams ()
{

    //    Query the devices in this context for their local memory sizes
    //    How we generate a kernel depends on the *minimum* LDS size for all devices.
    //
    const FFTEnvelope * pEnvelope = NULL;
    OPENCL_V(this->plan->GetEnvelope (& pEnvelope), _T("GetEnvelope failed"));
    BUG_CHECK (NULL != pEnvelope);

    // Remainder: params was properly cleared by its constructor
    //            clearing it again would destroy datasize and id!!
    this->signature.fft_precision    = this->plan->precision;
    this->signature.fft_placeness    = this->plan->placeness;
    this->signature.fft_inputLayout  = this->plan->inputLayout;
	this->signature.fft_MaxWorkGroupSize = this->plan->envelope.limit_WorkGroupSize;

    ARG_CHECK(this->plan->length.size()    > 0);
	ARG_CHECK(this->plan->inStride.size()  > 0);
    ARG_CHECK(this->plan->outStride.size() > 0);

    ARG_CHECK (this->plan->inStride.size() == this->plan->outStride.size())

	bool real_transform = ((this->plan->inputLayout == CLFFT_REAL) || (this->plan->outputLayout == CLFFT_REAL));

    if ( (CLFFT_INPLACE == this->plan->placeness) && (!real_transform) ) {
        //    If this is an in-place transform the
        //    input and output layout, dimensions and strides
        //    *MUST* be the same.
        //
        ARG_CHECK (this->plan->inputLayout == this->plan->outputLayout)
        this->signature.fft_outputLayout = this->plan->inputLayout;
        for (size_t u = this->plan->inStride.size(); u-- > 0; ) {
            ARG_CHECK (this->plan->inStride[u] == this->plan->outStride[u]);
        }
    } else {
        this->signature.fft_outputLayout = this->plan->outputLayout;
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


	this->signature.fft_RCsimple = this->plan->RCsimple;

	this->signature.fft_realSpecial = this->plan->realSpecial;
	this->signature.fft_realSpecial_Nr = this->plan->realSpecial_Nr;

	this->signature.blockCompute = this->plan->blockCompute;
	this->signature.blockComputeType = this->plan->blockComputeType;

	this->signature.fft_twiddleFront = this->plan->twiddleFront;

	size_t wgs, nt;
#ifdef PARMETERS_TO_BE_READ
	ParamRead pr;
	ReadParameterFile(pr);
	wgs = pr.workGroupSize;
	nt = pr.numTransformsPerWg;
#else
	size_t t_wgs, t_nt;
	Precision pr = (this->signature.fft_precision == CLFFT_SINGLE) ? P_SINGLE : P_DOUBLE;
	switch(pr)
	{
	case P_SINGLE:
		{
			KernelCoreSpecs<P_SINGLE> kcs;
			kcs.GetWGSAndNT(this->signature.fft_N[0], t_wgs, t_nt);
			if(this->signature.blockCompute)
			{
				this->signature.blockSIMD = Kernel<P_SINGLE>::BlockSizes::BlockWorkGroupSize(this->signature.fft_N[0]);
				this->signature.blockLDS  = Kernel<P_SINGLE>::BlockSizes::BlockLdsSize(this->signature.fft_N[0]);
			}
		} break;
	case P_DOUBLE:
		{
			KernelCoreSpecs<P_DOUBLE> kcs;
			kcs.GetWGSAndNT(this->signature.fft_N[0], t_wgs, t_nt);
			if(this->signature.blockCompute)
			{
				this->signature.blockSIMD = Kernel<P_DOUBLE>::BlockSizes::BlockWorkGroupSize(this->signature.fft_N[0]);
				this->signature.blockLDS  = Kernel<P_DOUBLE>::BlockSizes::BlockLdsSize(this->signature.fft_N[0]);
			}
		} break;
	}

	if((t_wgs != 0) && (t_nt != 0) && (this->plan->envelope.limit_WorkGroupSize >= 256))
	{
		wgs = t_wgs;
		nt = t_nt;
	}
	else
		DetermineSizes(this->plan->envelope.limit_WorkGroupSize, this->signature.fft_N[0], wgs, nt, pr);
#endif

	assert((nt * this->signature.fft_N[0]) >= wgs);
	assert((nt * this->signature.fft_N[0])%wgs == 0);

	this->signature.fft_R = (nt * this->signature.fft_N[0])/wgs;
	this->signature.fft_SIMD = wgs;

	//Set pre-callback if specified
	if (this->plan->hasPreCallback)
	{
		this->signature.fft_hasPreCallback = true;
		this->signature.fft_preCallback = this->plan->preCallback;
	}

	//Set post-callback if specified
	if (this->plan->hasPostCallback)
	{
		this->signature.fft_hasPostCallback = true;
		this->signature.fft_postCallback = this->plan->postCallbackParam;
	}
	this->signature.limit_LocalMemSize = this->plan->envelope.limit_LocalMemSize;

    if (this->plan->large1D != 0) {
        ARG_CHECK (this->signature.fft_N[0] != 0)
        ARG_CHECK ((this->plan->large1D % this->signature.fft_N[0]) == 0)
        this->signature.fft_3StepTwiddle = true;
		if(!(this->plan->realSpecial))
			ARG_CHECK ( this->plan->large1D  == (this->signature.fft_N[1] * this->signature.fft_N[0]) );
    }

    this->signature.fft_fwdScale  = this->plan->forwardScale;
    this->signature.fft_backScale = this->plan->backwardScale;

    return CLFFT_SUCCESS;
}

clfftStatus FFTGeneratedStockhamAction::getWorkSizes (std::vector<size_t> & globalWS, std::vector<size_t> & localWS)
{
    //    How many complex numbers in the input mutl-dimensional array?
    //
    unsigned long long count = 1;
    for (unsigned u = 0; u < this->plan->length.size(); ++u) {
        count *= std::max<size_t> (1, this->plan->length[ u ]);
    }
    count *= this->plan->batchsize;


	if(this->signature.blockCompute)
	{
		count = DivRoundingUp<unsigned long long> (count, this->signature.blockLDS); 
		count = count * this->signature.blockSIMD; 

		globalWS.push_back( static_cast< size_t >( count ) );
		localWS.push_back( this->signature.blockSIMD );

		return    CLFFT_SUCCESS;
	}

    count = DivRoundingUp<unsigned long long> (count, this->signature.fft_R);      // count of WorkItems
    count = DivRoundingUp<unsigned long long> (count, this->signature.fft_SIMD);   // count of WorkGroups

	// for real transforms we only need half the work groups since we do twice the work in 1 work group
	if( !(this->signature.fft_RCsimple) && ((this->signature.fft_inputLayout == CLFFT_REAL) || (this->signature.fft_outputLayout == CLFFT_REAL)) )
		count = DivRoundingUp<unsigned long long> (count, 2);

    count = std::max<unsigned long long> (count, 1) * this->signature.fft_SIMD;
        // .. count of WorkItems, rounded up to next multiple of fft_SIMD.

	// 1 dimension work group size
	globalWS.push_back( static_cast< size_t >( count ) );

    localWS.push_back( this->signature.fft_SIMD );

    return    CLFFT_SUCCESS;
}

clfftStatus FFTPlan::GetMax1DLengthStockham (size_t * longest) const
{
	// TODO  The caller has already acquired the lock on *this
	//	However, we shouldn't depend on it.

	//	Query the devices in this context for their local memory sizes
	//	How large a kernel we can generate depends on the *minimum* LDS
	//	size for all devices.
	//
	const FFTEnvelope * pEnvelope = NULL;
	OPENCL_V(this->GetEnvelope (& pEnvelope), _T("GetEnvelope failed"));
	BUG_CHECK (NULL != pEnvelope);

	ARG_CHECK (NULL != longest)
	size_t LdsperElement = this->ElementSize();
	size_t result = pEnvelope->limit_LocalMemSize /
		(1 * LdsperElement);
	result = FloorPo2 (result);
	*longest = result;
	return CLFFT_SUCCESS;
}

clfftStatus FFTGeneratedStockhamAction::generateKernel(FFTRepo& fftRepo, const cl_command_queue commQueueFFT )
{
    cl_int status = CL_SUCCESS;
    cl_device_id Device = NULL;
    status = clGetCommandQueueInfo(commQueueFFT, CL_QUEUE_DEVICE, sizeof(cl_device_id), &Device, NULL);
    OPENCL_V( status, _T( "clGetCommandQueueInfo failed" ) );

    cl_context QueueContext = NULL;
    status = clGetCommandQueueInfo(commQueueFFT, CL_QUEUE_CONTEXT, sizeof(cl_context), &QueueContext, NULL);
    OPENCL_V( status, _T( "clGetCommandQueueInfo failed" ) );

	std::string programCode;
	Precision pr = (this->signature.fft_precision == CLFFT_SINGLE) ? P_SINGLE : P_DOUBLE;
	switch(pr)
	{
	case P_SINGLE:
		{
			Kernel<P_SINGLE> kernel(this->signature);
			kernel.GenerateKernel(programCode, Device);
		} break;
	case P_DOUBLE:
		{
			Kernel<P_DOUBLE> kernel(this->signature);
			kernel.GenerateKernel(programCode, Device);
		} break;
	}

	//Requested local memory size by callback must not exceed the device LDS limits after factoring the LDS size required by main FFT kernel
	if ((this->signature.fft_hasPreCallback && this->signature.fft_preCallback.localMemSize > 0) || 
		(this->signature.fft_hasPostCallback && this->signature.fft_postCallback.localMemSize > 0))
	{
		bool validLDSSize = false;
		size_t requestedCallbackLDS = 0;

		if (this->signature.fft_hasPreCallback && this->signature.fft_preCallback.localMemSize > 0)
			requestedCallbackLDS = this->signature.fft_preCallback.localMemSize;
		if (this->signature.fft_hasPostCallback && this->signature.fft_postCallback.localMemSize > 0)
			requestedCallbackLDS += this->signature.fft_postCallback.localMemSize;

		if (this->plan->blockCompute)
		{
			validLDSSize = ((this->signature.blockLDS * this->plan->ElementSize()) +  requestedCallbackLDS) < this->plan->envelope.limit_LocalMemSize;
		}
		else
		{
			size_t length = this->signature.fft_N[0];
			size_t workGroupSize = this->signature.fft_SIMD;
			size_t numTrans = (workGroupSize * this->signature.fft_R) / length;

			//TODO - Need to abstract this out. Repeating the same compute as in GenerateKernel. 
			// Set half lds only for power-of-2 problem sizes & interleaved data
			bool halfLds = ( (this->signature.fft_inputLayout == CLFFT_COMPLEX_INTERLEAVED) &&
						(this->signature.fft_outputLayout == CLFFT_COMPLEX_INTERLEAVED) ) ? true : false;
			halfLds = halfLds ? ((length & (length-1)) ? false : true) : false;

			// Set half lds for real transforms
			halfLds = ( (this->signature.fft_inputLayout == CLFFT_REAL) &&
						(this->signature.fft_outputLayout == CLFFT_REAL) ) ? true : halfLds;

			size_t ldsSize = halfLds ? length*numTrans : 2*length*numTrans;
			size_t elementSize = ((this->signature.fft_precision == CLFFT_DOUBLE) || (this->signature.fft_precision == CLFFT_DOUBLE_FAST)) ? sizeof(double) : sizeof(float);

			validLDSSize = ((ldsSize * elementSize) + requestedCallbackLDS) < this->plan->envelope.limit_LocalMemSize;
		}

		if(!validLDSSize)
		{
			fprintf(stderr, "Requested local memory size not available\n");
			return CLFFT_INVALID_ARG_VALUE;
		}
	}

#ifdef KERNEL_INTERJECT
	ReadKernelFromFile(programCode);
#endif

    OPENCL_V( fftRepo.setProgramCode( this->getGenerator(), this->getSignatureData(), programCode, Device, QueueContext ), _T( "fftRepo.setclString() failed!" ) );
    OPENCL_V( fftRepo.setProgramEntryPoints( this->getGenerator(), this->getSignatureData(), "fft_fwd", "fft_back", Device, QueueContext ), _T( "fftRepo.setProgramEntryPoint() failed!" ) );

    return CLFFT_SUCCESS;
}
