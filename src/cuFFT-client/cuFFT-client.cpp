// ########################################################################
// Copyright 2015 Advanced Micro Devices, Inc.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ########################################################################

/* Example showing the use of CUFFT  */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "statisticalTimer.extern.h"
#include "client.h"

#ifdef __linux
#define strcmpi strcasecmp
#endif

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

template<typename T>
bool runC2CFFT(int rank, size_t lengthX, size_t lengthY, size_t lengthZ, int direction,
			   cufftType type, int profile_count, int batch_size, bool outPlace,
			   int iDist, int oDist, int iStride, int oStride);

template<typename In, typename Out>
bool runR2CFFT(int rank, size_t lengthX, size_t lengthY, size_t lengthZ,
			   cufftType type, int profile_count, int batch_size, bool outPlace,
			   int iDist, int oDist, int iStride, int oStride);

template<typename In, typename Out>
bool runC2RFFT(int rank, size_t lengthX, size_t lengthY, size_t lengthZ,
			   cufftType type, int profile_count, int batch_size, bool outPlace,
			   int iDist, int oDist, int iStride, int oStride);

template<typename T>
bool runFFTTransformC2CAdv(size_t fftLength, cufftType type, int rank, int* n, size_t inmem_size, size_t outmem_size, int* inembed, int* outembed,    
						  int direction, int profile_count, int batch_size, bool outPlace, int istride, int ostride, int idist, int odist);

template<typename In, typename Out>
bool runFFTTransformR2CAdv(size_t fftLength, cufftType type, size_t outLength, int rank, int* n, size_t inmem_size, size_t outmem_size, int* inembed, int* outembed,    
						  int profile_count, int batch_size, bool outPlace, int istride, int ostride, int idist, int odist);

template<typename In, typename Out>
bool runFFTTransformC2RAdv(size_t fftLength, cufftType type, size_t outLength, size_t innerDimLength, int rank, int* n, size_t inmem_size, size_t outmem_size,    
						  int* inembed, int* outembed, int profile_count, int batch_size, bool outPlace, int istride, int ostride, int idist, int odist);

template<typename T>
bool run1DFFTTransformC2C(size_t lengthX, int direction, cufftType type, int profile_count, int batch_size, bool outPlace);

template<typename In, typename Out>
bool run1DFFTTransformR2C(size_t lengthX, cufftType type, int profile_count, int batch_size, bool outPlace);

template<typename In, typename Out>
bool run1DFFTTransformC2R(size_t lengthX, cufftType type, int profile_count, int batch_size, bool outPlace);

template<typename T>
bool run2DFFTTransformC2C(size_t lengthX, size_t lengthY, int direction, cufftType type, int profile_count, int batch_size, bool outPlace);

template<typename In, typename Out>
bool run2DFFTTransformR2C(size_t lengthX, size_t lengthY, cufftType type, int profile_count, int batch_size, bool outPlace);

template<typename In, typename Out>
bool run2DFFTTransformC2R(size_t lengthX, size_t lengthY, cufftType type, int profile_count, int batch_size, bool outPlace);

template<typename T>
bool run3DFFTTransformC2C(size_t lengthX, size_t lengthY, size_t lengthZ, int direction, cufftType type, int profile_count, int batch_size, bool outPlace);

template<typename In, typename Out>
bool run3DFFTTransformR2C(size_t lengthX, size_t lengthY, size_t lengthZ, cufftType type, int profile_count, int batch_size, bool outPlace);

template<typename In, typename Out>
bool run3DFFTTransformC2R(size_t lengthX, size_t lengthY, size_t lengthZ, cufftType type, int profile_count, int batch_size, bool outPlace);

template<typename In, typename Out>
void cleanup(In *h_in_signal, In *d_in_signal, Out *h_output_signal, Out *d_output_signal, bool outplace);

#define SIGNAL_SIZE        1024

// Global variables

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA FFT
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
	bool isInverse = false;
	cufftType fftType = CUFFT_C2C;
	int intType = 1;
	size_t lengthX = SIGNAL_SIZE;
	size_t lengthY = 1;
	size_t lengthZ = 1;
	int profile_count = 5;
	int batch_size = 1;
	bool outPlace = false;
	int iStride = 1;
	int oStride = 1;
	int iDist = 0;
	int oDist = 0;
	int rank = 1;
	bool bTestResult = false;
	bool isDouble = false;

	//Parse command-line options
	while (argv[1] && argv[1][0] == '-') {
    if (strcmpi(argv[1], "-inv") == 0) {
      argv++; argc--;
      isInverse = atoi(argv[1]) ? true : false;
    } else if (strcmpi(argv[1], "-type") == 0) {
      argv++; argc--;
	  intType = atoi(argv[1]);
	} 
	else if (strcmpi(argv[1], "-b") == 0) {
      argv++; argc--;
	  batch_size = atoi(argv[1]);
	} else if (strcmpi(argv[1], "-d") == 0) {
      argv++; argc--;
	  isDouble = atoi(argv[1]) ? true : false;
	} else if (strcmpi(argv[1], "-x") == 0) {
      argv++; argc--;
	  lengthX = atoi(argv[1]);
	} else if (strcmpi(argv[1], "-y") == 0) {
      argv++; argc--;
	  lengthY = atoi(argv[1]);
	} else if (strcmpi(argv[1], "-z") == 0) {
      argv++; argc--;
	  lengthZ = atoi(argv[1]);
	} else if (strcmpi(argv[1], "-p") == 0) {
      argv++; argc--;
	  profile_count = atoi(argv[1]);
	} else if (strcmpi(argv[1], "-o") == 0) {
      argv++; argc--;
	  outPlace = atoi(argv[1]) ? true : false;
	} else if (strcmpi(argv[1], "-istr") == 0) {
      argv++; argc--;
	  iStride = atoi(argv[1]);
	} else if (strcmpi(argv[1], "-ostr") == 0) {
      argv++; argc--;
	  oStride = atoi(argv[1]);
	} else if (strcmpi(argv[1], "-idist") == 0) {
      argv++; argc--;
	  iDist = atoi(argv[1]);
	} else if (strcmpi(argv[1], "-odist") == 0) {
      argv++; argc--;
	  oDist = atoi(argv[1]);
	} else {
	  if (strcmpi(argv[1], "-h") != 0)
		fprintf(stderr, "Illegal option %s ignored\n", argv[1]);

		printf("Usage:\n %s\n[-inv <0|1 Inverse transform (default: forward 1)>]\n"
			"[-type <Type of FFT Transform:\n	1) C2C (default)\n	2) R2C\n	3) C2R>]\n"
			  "[-b <batch size>]\n"
			 "[-d <0|1 Use Double Precision (default: 0 i.e. Single Precision)]\n"
			 "[-x <Specify the length of the 1st dimension of a test array>]\n"
			 "[-y <Specify the length of the 2nd dimension of a test array>]\n"
			 "[-z <Specify the length of the 3rd dimension of a test array>]\n"
			 "[-p <Specify number of iterations for profiling measurements. default 5>]\n"
			 "[-o <0|1 Out of place FFT transform (default: 0 i.e. in place)]\n"
			 "[-istr <Specify the input stride of the innermost dimension of a test array]\n"
			 "[-ostr <Specify the output stride of the innermost dimension of a test array]\n"
			 "[-idist <Specify the distance between first element of two consecutive batches of input ]\n"
			 "[-odist <Specify the distance between first element of two consecutive batches of output ]\n"
			 "[-h (usage)]\n", argv[0]);
		  exit(1);
		}
		argv++;
		argc--;
	  }

	if (lengthX < 1 || lengthY < 1 || lengthZ < 1)
	{
		printf("Length of test array must be >= 1 in each dimension. Exiting..\n"); 
		exit(EXIT_FAILURE);
	}

	switch (intType)
	{
	case 1: 
		
		fftType = isDouble ? CUFFT_Z2Z : CUFFT_C2C; 
		break;
	case 2: 
		fftType = isDouble ? CUFFT_D2Z : CUFFT_R2C; 
		break;
	case 3: 
		fftType = isDouble ? CUFFT_Z2D : CUFFT_C2R; 
		break;
	default: printf("Input Arguments ERROR!! Invalid FFT type. Use -h to check the correct options. Exiting..\n"); exit(EXIT_FAILURE);
	}

    printf("[cuFFT-ClientApp] is starting...\n");

    findCudaDevice(argc, (const char **)argv);

	//Find the rank
	if (lengthY == 1 && lengthZ == 1)
	{
		rank = 1;
	}
	else if (lengthY > 1 && lengthZ == 1)
	{
		rank = 2;
	}
	else if (lengthY > 1 && lengthZ > 1)
	{
		rank = 3;
	}

	int direction = isInverse ? CUFFT_INVERSE : CUFFT_FORWARD;

	//Switch based on the type of transform
	switch (fftType)
	{
		case CUFFT_C2C: 
		{
			//Single C2C FFT
			bTestResult = runC2CFFT<cufftComplex>(rank, lengthX, lengthY, lengthZ, direction, fftType , profile_count, batch_size, outPlace, iDist, oDist, iStride, oStride);
			
			break;
		}	
		case CUFFT_Z2Z: 
		{
			//Double C2C FFT
			bTestResult = runC2CFFT<cufftDoubleComplex>(rank, lengthX, lengthY, lengthZ, direction, fftType , profile_count, batch_size, outPlace, iDist, oDist, iStride, oStride);
			
			break;
		}	
		case CUFFT_R2C: 
		{
			//R2C FFT
			bTestResult = runR2CFFT<cufftReal, cufftComplex>(rank, lengthX, lengthY, lengthZ, fftType, profile_count, batch_size, outPlace, iDist, oDist, iStride, oStride);
			break;
		}
		case CUFFT_D2Z: 
		{
			//Double R2C FFT
			bTestResult = runR2CFFT<cufftDoubleReal, cufftDoubleComplex>(rank, lengthX, lengthY, lengthZ, fftType, profile_count, batch_size, outPlace, iDist, oDist, iStride, oStride);
			break;
		}
		case CUFFT_C2R: 
		{
			//C2R FFT
			bTestResult = runC2RFFT<cufftComplex, cufftReal>(rank, lengthX, lengthY, lengthZ, fftType, profile_count, batch_size, outPlace, iDist, oDist, iStride, oStride);
			break;
		}
		case CUFFT_Z2D: 
		{
			//C2R FFT
			bTestResult = runC2RFFT<cufftDoubleComplex, cufftDoubleReal>(rank, lengthX, lengthY, lengthZ, fftType, profile_count, batch_size, outPlace, iDist, oDist, iStride, oStride);
			break;
		}
		default: printf("Invalid FFT type. Exiting..\n");
			break;
	}

	if (bTestResult)
	{
		printf("FFT Transformation PASSED!\n");
	}
	else
	{
		printf("FFT Transformation FAILED!!\n");
	}
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

/*
* Run C2R FFT for 1D/2D/3D
*/
template<typename In, typename Out>
bool runC2RFFT(int rank, size_t lengthX, size_t lengthY, size_t lengthZ,
			   cufftType type, int profile_count, int batch_size, bool outPlace,
			   int iDist, int oDist, int iStride, int oStride)
{
	bool bTestResult = false;

	if (!outPlace && iStride < oStride)
	{
		printf("Input Arguments ERROR!! For in-place transform, output stride must be less than or equal to input stride. Exiting..\n\n");

		exit(EXIT_FAILURE);
	}

	if (type == CUFFT_Z2D)
		printf("Transforming signal Double Complex to Double Real \n");
	else
		printf("Transforming signal Complex to Real \n");

	switch (rank)
	{
		case 1:
		{
			if (iStride == 1 && oStride == 1 && iDist == 0 && oDist == 0)
			{
				//Basic Data layout
				printf("Dimension : 1\n");
				printf("Data Layout : Basic\n");

				bTestResult = run1DFFTTransformC2R<In, Out>(lengthX, type, profile_count, batch_size, outPlace); 
			}
			else
			{
				//Advanced Data layout

				size_t fftLength = lengthX;
				size_t innerDimLength = outPlace ? lengthX  : (lengthX + 2) ;
				size_t inLength = (lengthX/2 + 1)*iStride;
				size_t outLength = innerDimLength * oStride;

				if (iDist > 0 && iDist < inLength)
				{
					printf("Input Arguments ERROR!! Value of idist cannot be less than product of lengths along each dimension and stride, ((x/2+1)*istr). Exiting..\n\n");
					exit(EXIT_FAILURE);
				}

				if (oDist > 0 && (oDist < outLength))
				{
					printf("Input Arguments ERROR!! Value of odist cannot be less than product of lengths along each dimension and stride, ((x+2)*ostr). Exiting..\n\n");
					exit(EXIT_FAILURE);
				}
								
				iDist = iDist? iDist : (int)inLength;
				oDist = oDist? oDist : (int)outLength;
				
				int n[1] = {(int)lengthX};

				int inembed[1] = {(int)(lengthX/2+1)};
				int outembed[1] = {(int)innerDimLength};

				size_t inmem_size = sizeof(In) * iDist * batch_size;
				size_t outmem_size = sizeof(Out) * oDist * batch_size;

				printf("Dimension : 1\n");
				printf("Data Layout : Advanced\n");

				bTestResult = runFFTTransformC2RAdv<In, Out>(fftLength, type, outLength, innerDimLength, rank, n, inmem_size, outmem_size, 
													inembed, outembed, profile_count, batch_size, outPlace, iStride, oStride,
													iDist, oDist);
			}
			break;
		}
		case 2:
		{
			if (iStride == 1 && oStride == 1 && iDist == 0 && oDist == 0)
			{
				//Basic Data layout
				printf("Dimension : 2\n");
				printf("Data Layout : Basic\n");

				bTestResult = run2DFFTTransformC2R<In, Out>(lengthX, lengthY, type, profile_count, batch_size, outPlace); 
			}
			else
			{
				//Advanced Data layout

				size_t fftLength = lengthX * lengthY;
				size_t innerDimLength = outPlace ? lengthY  : (lengthY + 2) ;
				size_t inLength = lengthX * (lengthY/2 + 1)*iStride;
				size_t outLength = lengthX * innerDimLength * oStride;

				if (iDist > 0 && iDist < inLength)
				{
					printf("Input Arguments ERROR!! Value of idist cannot be less than product of lengths along each dimension and stride, (x*(y/2+1)*istr). Exiting..\n\n");
					exit(EXIT_FAILURE);
				}

				if (oDist > 0 && (oDist < outLength))
				{
					printf("Input Arguments ERROR!! Value of odist cannot be less than product of lengths along each dimension and stride, (x*(y+2)*ostr). Exiting..\n\n");
					exit(EXIT_FAILURE);
				}
								
				iDist = iDist? iDist : (int)inLength;
				oDist = oDist? oDist : (int)outLength;
				
				int n[2] = {(int)lengthX, (int)lengthY};

				int inembed[2] = {(int)lengthX, (int)(lengthY/2+1)};
				int outembed[2] = {(int)lengthX, (int)innerDimLength};

				size_t inmem_size = sizeof(In) * iDist * batch_size;
				size_t outmem_size = sizeof(Out) * oDist * batch_size;

				printf("Dimension : 2\n");
				printf("Data Layout : Advanced\n");

				bTestResult = runFFTTransformC2RAdv<In, Out>(fftLength, type, outLength, innerDimLength, rank, n, inmem_size, outmem_size, 
													inembed, outembed, profile_count, batch_size, outPlace, iStride, oStride,
													iDist, oDist);
			}
			break;
		}
		case 3:
		{
			if (iStride == 1 && oStride == 1 && iDist == 0 && oDist == 0)
			{
				//Basic Data layout
				printf("Dimension : 3\n");
				printf("Data Layout : Basic\n");

				bTestResult = run3DFFTTransformC2R<In, Out>(lengthX, lengthY, lengthZ, type, profile_count, batch_size, outPlace); 
			}
			else
			{
				//Advanced Data layout

				size_t fftLength = lengthX * lengthY * lengthZ;
				size_t innerDimLength = outPlace ? lengthZ  : (lengthZ + 2) ;
				size_t inLength = lengthX * lengthY * (lengthZ/2 + 1)*iStride;
				size_t outLength = lengthX * lengthY * innerDimLength * oStride;

				if (iDist > 0 && iDist < inLength)
				{
					printf("Input Arguments ERROR!! Value of idist cannot be less than product of lengths along each dimension and stride, (x*y*(z/2+1)*istr). Exiting..\n\n");
					exit(EXIT_FAILURE);
				}

				if (oDist > 0 && (oDist < outLength))
				{
					printf("Input Arguments ERROR!! Value of odist cannot be less than product of lengths along each dimension and stride, (x*y*(z+2)*ostr). Exiting..\n\n");
					exit(EXIT_FAILURE);
				}
								
				iDist = iDist? iDist : (int)inLength;
				oDist = oDist? oDist : (int)outLength;
				
				int n[3] = {(int)lengthX, (int)lengthY, (int)lengthZ};

				int inembed[3] = {(int)lengthX, (int)lengthY, (int)(lengthZ/2+1)};
				int outembed[3] = {(int)lengthX, (int)lengthY, (int)innerDimLength};

				size_t inmem_size = sizeof(In) * iDist * batch_size;
				size_t outmem_size = sizeof(Out) * oDist * batch_size;

				printf("Dimension : 3\n");
				printf("Data Layout : Advanced\n");

				bTestResult = runFFTTransformC2RAdv<In, Out>(fftLength, type, outLength, innerDimLength, rank, n, inmem_size, outmem_size, 
													inembed, outembed, profile_count, batch_size, outPlace, iStride, oStride,
													iDist, oDist);
			}
			break;
		}
	}

	return bTestResult;
}

/*
* Run R2C FFT for 1D/2D/3D
*/
template<typename In, typename Out>
bool runR2CFFT(int rank, size_t lengthX, size_t lengthY, size_t lengthZ,
			   cufftType type, int profile_count, int batch_size, bool outPlace,
			   int iDist, int oDist, int iStride, int oStride)
{
	bool bTestResult = false;

	if (!outPlace && (iStride < oStride || iDist < oDist))
	{
		printf("Input Arguments ERROR!! For in-place transform, output stride and distance must be less than or equal to input stride and distance respectively. Exiting..\n\n");

		exit(EXIT_FAILURE);
	}

	if (type == CUFFT_D2Z)
		printf("Transforming signal Double Real to Double Complex \n");
	else
		printf("Transforming signal Real to Complex \n");
	
	switch (rank)
	{
		case 1:
		{
			//1D Transform
			if (iDist > 0 && iDist < ((lengthX+2)*iStride))
			{
				printf("Input Arguments ERROR!! Value of idist cannot be less than product of lengths along each dimension and stride, ((x+2)*istr). Exiting..\n\n");
				exit(EXIT_FAILURE);
			}

			if (oDist > 0 && (oDist < ((lengthX/2 + 1)*oStride)))
			{
				printf("Input Arguments ERROR!! Value of odist cannot be less than product of lengths along each dimension and stride, ((x/2+1)*ostr). Exiting..\n\n");
				exit(EXIT_FAILURE);
			}

			if (iStride == 1 && oStride == 1 && iDist == 0 && oDist == 0)
			{
				//Basic Data layout
				printf("Dimension : 1\n");
				printf("Data Layout : Basic\n");

				bTestResult = run1DFFTTransformR2C<In, Out>(lengthX, type, profile_count, batch_size, outPlace);
			}
			else
			{
				//Advanced Data layout
				iDist = iDist? iDist : (int)((lengthX+2)*iStride);
				oDist = oDist? oDist : (int)((lengthX/2 + 1)*oStride);

				if (!outPlace && oDist > (iDist/2))
				{
					printf("Input Arguments ERROR!! For in-place Real to Complex transform, output array distance must not be greater than half of input array distance. Exiting..\n\n");
						
					exit(EXIT_FAILURE);
				}

				int n[1] = {(int)lengthX};
				size_t fftLength = lengthX;
				size_t outLength = (lengthX/2 + 1)*oStride;

				int inembed[1] = {(int)(lengthX + 2)};
				int outembed[1] = {(int)(lengthX/2 + 1)};

				size_t inmem_size = sizeof(In) * iDist * batch_size;
				size_t outmem_size = sizeof(Out) * oDist * batch_size;
						
				printf("Dimension : 1\n");
				printf("Data Layout : Advanced\n");
	
				bTestResult = runFFTTransformR2CAdv<In, Out>(fftLength, type, outLength, rank, n, inmem_size, outmem_size, inembed, outembed,
													profile_count, batch_size, outPlace, iStride, oStride, iDist, oDist);
			}

			break;
		}
		case 2:
		{
			if (iDist > 0 && iDist < (lengthX *(lengthY+2)*iStride))
			{
				printf("Input Arguments ERROR!! Value of idist cannot be less than product of lengths along each dimension and stride, (x*(y+2)*istr). Exiting..\n\n");
				exit(EXIT_FAILURE);
			}

			if (oDist > 0 && (oDist < (lengthX *(lengthY/2 + 1)*oStride)))
			{
				printf("Input Arguments ERROR!! Value of odist cannot be less than product of lengths along each dimension and stride, (x*(y/2+1)*ostr). Exiting..\n\n");
				exit(EXIT_FAILURE);
			}

			if (iStride == 1 && oStride == 1 && iDist == 0 && oDist == 0)
			{
				// Basic Data layout
				printf("Dimension : 2\n");
				printf("Data Layout : Basic\n");

				bTestResult = run2DFFTTransformR2C<In, Out>(lengthX, lengthY, type, profile_count, batch_size, outPlace);
			}
			else
			{
				//Advanced Data layout
				iDist = iDist? iDist : (int)(lengthX * (lengthY+2)*iStride);
				oDist = oDist? oDist : (int)(lengthX * (lengthY/2 + 1)*oStride);

				if (!outPlace && oDist > (iDist/2))
				{
					printf("Input Arguments ERROR!! For in-place Real to Complex transform, output array distance must not be greater than half of input array distance. Exiting..\n\n");
						
					exit(EXIT_FAILURE);
				}

				int n[2] = {(int)lengthX, (int)lengthY};
				size_t fftLength = lengthX * lengthY;
				size_t outLength = lengthX * (lengthY/2 + 1)*oStride;

				int inembed[2] = {(int)lengthX, (int)(lengthY+2)};
				int outembed[2] = {(int)lengthX, (int)(lengthY/2+1)};

				size_t inmem_size = sizeof(In) * iDist * batch_size;
				size_t outmem_size = sizeof(Out) * oDist * batch_size;

				printf("Dimension : 2\n");
				printf("Data Layout : Advanced\n");
	
				bTestResult = runFFTTransformR2CAdv<In, Out>(fftLength, type, outLength, rank, n, inmem_size, outmem_size, inembed, outembed,
													profile_count, batch_size, outPlace, iStride, oStride, iDist, oDist);
			}

			break;
		}
		case 3:
		{
			if (iDist > 0 && iDist < (lengthX * lengthY *(lengthZ+2)*iStride))
			{
				printf("Input Arguments ERROR!! Value of idist cannot be less than product of lengths along each dimension and stride, (x*y*(z+2)*istr). Exiting..\n\n");
				exit(EXIT_FAILURE);
			}

			if (oDist > 0 && (oDist < (lengthX * lengthY *(lengthZ/2 +  1)*oStride)))
			{
				printf("Input Arguments ERROR!! Value of odist cannot be less than product of lengths along each dimension and stride, (x*y*(z/2+1)*ostr). Exiting..\n\n");
				exit(EXIT_FAILURE);
			}

			if (iStride == 1 && oStride == 1 && iDist == 0 && oDist == 0)
			{
				// Basic Data layout
				printf("Dimension : 3\n");
				printf("Data Layout : Basic\n");

				bTestResult = run3DFFTTransformR2C<In, Out>(lengthX, lengthY, lengthZ, type, profile_count, batch_size, outPlace);
			}
			else
			{
				//Advanced Data layout
				iDist = iDist? iDist : (int)(lengthX * lengthY * (lengthZ+2)*iStride);
				oDist = oDist? oDist : (int)(lengthX * lengthY * (lengthZ/2 + 1)*oStride);

				if (!outPlace && oDist > (iDist/2))
				{
					printf("Input Arguments ERROR!! For in-place Real to Complex transform, output array distance must not be greater than half of input array distance. Exiting..\n\n");
						
					exit(EXIT_FAILURE);
				}

				int n[3] = {(int)lengthX, (int)lengthY, (int)lengthZ};
				size_t fftLength = lengthX * lengthY * lengthZ;
				size_t outLength = lengthX * lengthY * (lengthZ/2 + 1)*oStride;

				int inembed[3] = {(int)lengthX, (int)lengthY, (int)(lengthZ+2)};
				int outembed[3] = {(int)lengthX, (int)lengthY, (int)(lengthZ/2+1)};

				size_t inmem_size = sizeof(In) * iDist * batch_size;
				size_t outmem_size = sizeof(Out) * oDist * batch_size;

				printf("Dimension : 3\n");
				printf("Data Layout : Advanced\n");
	
				bTestResult = runFFTTransformR2CAdv<In, Out>(fftLength, type, outLength, rank, n, inmem_size, outmem_size, inembed, outembed,
													profile_count, batch_size, outPlace, iStride, oStride, iDist, oDist);
			}

			break;
		}
	}
	return bTestResult;
}

/*
* Run C2C FFT for 1D/2D/3D
*/
template<typename T>
bool runC2CFFT(int rank, size_t lengthX, size_t lengthY, size_t lengthZ, int direction,
			   cufftType type, int profile_count, int batch_size, bool outPlace,
			   int iDist, int oDist, int iStride, int oStride)
{
	bool bTestResult = false;

	if (!outPlace && (iStride < oStride || iDist < oDist))
	{
		printf("Input Arguments ERROR!! For in-place transform, output stride and distance must be less than or equal to input stride and distance respectively. Exiting..\n\n");

		exit(EXIT_FAILURE);
	}

	if (iDist > 0 && iDist < (lengthX *lengthY*lengthZ*iStride))
	{
		printf("Input Arguments ERROR!! Value of idist cannot be less than product of lengths along each dimension and stride, (x*y*z*istr). Exiting..\n\n");
		exit(EXIT_FAILURE);
	}

	if (oDist > 0 && (oDist < (lengthX *lengthY*lengthZ*oStride)))
	{
		printf("Input Arguments ERROR!! Value of odist cannot be less than product of lengths along each dimension and stride, (x*y*z*ostr). Exiting..\n\n");
		exit(EXIT_FAILURE);
	}

	if (type == CUFFT_Z2Z)
		printf("Transforming signal Double Complex to Double Complex \n");
	else
		printf("Transforming signal Complex to Complex \n");

	switch (rank)
		{
			case 1:
			{
				//1D Transform

				if (iStride == 1 && oStride == 1 && iDist == 0 && oDist == 0)
				{
					//basic data layout
					
					printf("Dimension : 1\n");
					printf("Data Layout : Basic\n");

					bTestResult = run1DFFTTransformC2C<T>(lengthX, direction, type, profile_count, batch_size, outPlace);
				}
				else
				{
					//Advanced Data layout
					iDist = iDist? iDist : (int)(lengthX * iStride);
					oDist = oDist? oDist : (int)(lengthX * oStride);

					int n[1] = {(int)lengthX};
					size_t fftLength = lengthX;

					int inembed[1] = {(int)lengthX};
					int outembed[1] = {(int)lengthX};

					size_t inmem_size = sizeof(T) * iDist * batch_size;
					size_t outmem_size = sizeof(T) * oDist * batch_size;

					printf("Dimension : 1\n");
					printf("Data Layout : Advanced\n");
	
					bTestResult = runFFTTransformC2CAdv<T>(fftLength, type, rank, n, inmem_size, outmem_size, inembed, outembed,
														direction, profile_count, batch_size, outPlace, iStride, oStride, iDist, oDist);
				}
				break;
			}
			case 2:
			{
				//2D Transform

				if (iStride == 1 && oStride == 1 && iDist == 0 && oDist == 0)
				{
					//basic data layout
					printf("Dimension : 2\n");
					printf("Data Layout : Basic\n");

					bTestResult = run2DFFTTransformC2C<T>(lengthX, lengthY, direction, type, profile_count, batch_size, outPlace);
				}
				else
				{
					//Advanced Data layout
					iDist = iDist? iDist : (int)(lengthX * lengthY*iStride);
					oDist = oDist? oDist : (int)(lengthX * lengthY*oStride);

					int n[2] = {(int)lengthX, (int)lengthY};
					size_t fftLength = lengthX * lengthY;

					int inembed[2] = {(int)lengthX, (int)lengthY};
					int outembed[2] = {(int)lengthX, (int)lengthY};

					size_t inmem_size = sizeof(T) * iDist * batch_size;
					size_t outmem_size = sizeof(T) * oDist * batch_size;

					printf("Dimension : 2\n");
					printf("Data Layout : Advanced\n");
	
					bTestResult = runFFTTransformC2CAdv<T>(fftLength, type, rank, n, inmem_size, outmem_size, inembed, outembed,
														direction, profile_count, batch_size, outPlace, iStride, oStride, iDist, oDist);
				}
								
				break;
			}
			case 3:
			{				
				//3D Transform

				if (iStride == 1 && oStride == 1 && iDist == 0 && oDist == 0)
				{
					//basic data layout					
					printf("Dimension : 3\n");
					printf("Data Layout : Basic\n");

					bTestResult = run3DFFTTransformC2C<T>(lengthX, lengthY, lengthZ, direction, type, profile_count, batch_size, outPlace);
				}
				else
				{
					//Advanced Data layout
					iDist = iDist? iDist : (int)(lengthX * lengthY * lengthZ * iStride);
					oDist = oDist? oDist : (int)(lengthX * lengthY * lengthZ * oStride);

					int n[3] = {(int)lengthX, (int)lengthY, (int)lengthZ};
					size_t fftLength = lengthX * lengthY*lengthZ;

					int inembed[3] = {(int)lengthX, (int)lengthY, (int)lengthZ};
					int outembed[3] = {(int)lengthX, (int)lengthY, (int)lengthZ};

					size_t inmem_size = sizeof(T) * iDist * batch_size;
					size_t outmem_size = sizeof(T) * oDist * batch_size;
					
					printf("Dimension : 3\n");
					printf("Data Layout : Advanced\n");
	
					bTestResult = runFFTTransformC2CAdv<T>(fftLength, type, rank, n, inmem_size, outmem_size, inembed, outembed,
														direction, profile_count, batch_size, outPlace, iStride, oStride, iDist, oDist);
				}
				break;
			}
		}

	return bTestResult;
}

/*
* Run C2R FFT Advanced Data Layout
*/
template<typename In, typename Out>
bool runFFTTransformC2RAdv(size_t fftLength, cufftType type, size_t outLength, size_t innerDimLength, int rank, int* n, size_t inmem_size, size_t outmem_size,    
						  int* inembed, int* outembed, int profile_count, int batch_size, bool outPlace, int istride, int ostride, int idist, int odist)
{
	size_t invectorLength = idist;

	// Allocate host memory for the signal
    In *h_in_signal = (In *)malloc(inmem_size);
	
	Out *h_output_signal;
	In *d_in_signal;
	Out *d_output_signal;

    // Allocate device memory for signal
    checkCudaErrors(cudaMalloc((void **)&d_in_signal, inmem_size));

	if (outPlace)
	{
		h_output_signal = (Out *)malloc(outmem_size);

		// Allocate device memory for out signal
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, outmem_size));
	}
	else
	{
		h_output_signal = (Out*)h_in_signal;
		d_output_signal = (Out*)d_in_signal;
	}

    // CUFFT plan
    cufftHandle plan;
    checkCudaErrors(cufftPlanMany(&plan, rank, n,
						inembed, istride, idist, // *inembed, istride, idist
						outembed, ostride, odist,  // *onembed, ostride, odist
						type, batch_size));

	Timer tr;
	double wtime_t = 0.0;

	// Initalize the memory for the signal
	memset(h_in_signal,0,inmem_size);
	for (size_t j = 0; j < (invectorLength * batch_size); j+=invectorLength)
	{
		h_in_signal[j].x = (float)fftLength;
	}

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, inmem_size,
							   cudaMemcpyHostToDevice));

	for (int i = 0; i < profile_count; i++)
	{
		tr.Start();

		switch (type)
		{
		case CUFFT_Z2D:
			checkCudaErrors(cufftExecZ2D(plan, (cufftDoubleComplex*)d_in_signal, (cufftDoubleReal*)d_output_signal));
			break;
		case CUFFT_C2R:
			checkCudaErrors(cufftExecC2R(plan, (cufftComplex*)d_in_signal, (cufftReal*)d_output_signal));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n"); 
			cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}
		
		cudaDeviceSynchronize();
		wtime_t += tr.Sample();

		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = fftLength;
	double opsconst = 2.5 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));

    	// Copy device memory to host
    	checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, outmem_size,
                               cudaMemcpyDeviceToHost));

	bool bTestResult = true;
    	// check result only if profile count == 1 or out-place transform
        if (profile_count == 1 || outPlace)
	{
		// check result
		size_t idx = 0;
		size_t innerDim = 0;
		for (int i = 0; i < batch_size; i++)
		{
			idx = i * odist;

			for (int k = 0; k < outLength; k += ostride)
			{
				//For in-place transform, skip the 2 elements after iterating every inner dimension length
				if (!outPlace && (innerDim == (innerDimLength - 2)*ostride || innerDim == (innerDimLength - 1)*ostride))
				{
					if (innerDim == (innerDimLength - 1)*ostride)
						innerDim = 0;
					else
						innerDim += ostride;

					idx += ostride;
					continue;
				}

				/*****************************************************************************************
				* cuFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set 
				* followed by an inverse FFT on the resulting set yields data that is equal to the input, 
				* scaled by the number of elements. Scaling either transform by the reciprocal of the size 
				* of the data set is left for the user to perform as seen fit
				******************************************************************************************/
				if (h_output_signal[idx] != fftLength)
				{
					printf("failed at %d, value=%f\n", idx, h_output_signal[idx]);
					bTestResult = false;
					//break;
				}

				idx += ostride;
				innerDim += ostride;
			}
			innerDim = 0;
		}
	}
	
    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}

/*
* Run R2C FFT Advanced Data Layout
*/
template<typename In, typename Out>
bool runFFTTransformR2CAdv(size_t fftLength, cufftType type, size_t outLength, int rank, int* n, size_t inmem_size, size_t outmem_size, int* inembed, int* outembed,    
						  int profile_count, int batch_size, bool outPlace, int istride, int ostride, int idist, int odist)
{
	size_t invectorLength = idist;

	// Allocate host memory for the signal
    In *h_in_signal = (In *)malloc(inmem_size);
	
	Out *h_output_signal;
	In *d_in_signal;
	Out *d_output_signal;

    // Allocate device memory for signal
    checkCudaErrors(cudaMalloc((void **)&d_in_signal, inmem_size));

	if (outPlace)
	{
		h_output_signal = (Out *)malloc(outmem_size);

		// Allocate device memory for out signal
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, outmem_size));
	}
	else
	{
		h_output_signal = (Out*)h_in_signal;
		d_output_signal = (Out*)d_in_signal;
	}

    // CUFFT plan
    cufftHandle plan;
    checkCudaErrors(cufftPlanMany(&plan, rank, n,
						inembed, istride, idist, // *inembed, istride, idist
						outembed, ostride, odist,  // *onembed, ostride, odist
						type, batch_size));

	// Initalize the memory for the signal
	for (unsigned int j = 0; j < (invectorLength * batch_size); ++j)
	{
		h_in_signal[j] = 1;
	}

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, inmem_size,
							   cudaMemcpyHostToDevice));
	Timer tr;
	double wtime_t = 0.0;

	for (int i = 0; i < profile_count; i++)
	{
		tr.Start();

		switch (type)
		{
		case CUFFT_D2Z:
			checkCudaErrors(cufftExecD2Z(plan, (cufftDoubleReal*)d_in_signal, (cufftDoubleComplex*)d_output_signal));
			break;
		case CUFFT_R2C:
			checkCudaErrors(cufftExecR2C(plan, (cufftReal*)d_in_signal, (cufftComplex*)d_output_signal));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n"); 
			cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}
		
		cudaDeviceSynchronize();
		wtime_t += tr.Sample();

		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = fftLength;
	double opsconst = 2.5 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));

	// Copy device memory to host
    	checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, outmem_size,
                               cudaMemcpyDeviceToHost));

    	// check result only if profile count == 1 or outplace transform
	bool bTestResult = true;
	if (profile_count == 1 || outPlace)
	{
		int idx = 0;
		for (int k = 0; k < batch_size; k++)
		{
			idx = k * odist;

			for( int i = idx; i < (idx + outLength); i += ostride)
			{
				if (0 == (i % odist))
				{
					if (h_output_signal[i].x != fftLength)
					{
						printf("fail at %d, value %f\n", i, h_output_signal[i].x);
						bTestResult = false;
						break;
					}
				}
				else
				{
					if (h_output_signal[i].x != 0)
					{
						printf("fail at %d, value %f\n", i, h_output_signal[i].x);
						bTestResult = false;
						break;
					}
				}

				if (h_output_signal[ i ].y != 0)
				{
					printf("fail at %d, value %f\n", i, h_output_signal[i].y);
					bTestResult = false;
					break;
				}
			}
		}
	}
	
    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}

/*
* Run C2C FFT Advanced Data Layout
*/
template<typename T>
bool runFFTTransformC2CAdv(size_t fftLength, cufftType type, int rank, int* n, size_t inmem_size, size_t outmem_size, int* inembed, int* outembed,    
						  int direction, int profile_count, int batch_size, bool outPlace, int istride, int ostride, int idist, int odist)
{
	size_t invectorLength = idist;
	size_t outvectorLength = odist; 

	// Allocate host memory for the signal
    T *h_in_signal = (T *)malloc(inmem_size);
	
	T *h_output_signal;
	T *d_in_signal;
	T *d_output_signal;

    // Allocate device memory for signal
    checkCudaErrors(cudaMalloc((void **)&d_in_signal, inmem_size));

	if (outPlace)
	{
		h_output_signal = (T *)malloc(outmem_size);

		// Allocate device memory for out signal
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, outmem_size));
	}
	else
	{
		h_output_signal = h_in_signal;
		d_output_signal = d_in_signal;
	}

    // CUFFT plan
    cufftHandle plan;
    checkCudaErrors(cufftPlanMany(&plan, rank, n,
						inembed, istride, idist, // *inembed, istride, idist
						outembed, ostride, odist,  // *onembed, ostride, odist
						type, batch_size));

	Timer tr;
	double wtime_t = 0.0;

	// Initalize the memory for the signal
	if (direction == CUFFT_FORWARD)
	{
		for (size_t idx = 0; idx < (invectorLength*batch_size); ++idx)
		{
			h_in_signal[idx].x = 1;
			h_in_signal[idx].y = 0;
		}
	}
	else
	{
		//Inverse FFT
		memset(h_in_signal, 0, inmem_size);

		for (size_t idx = 0; idx < (invectorLength*batch_size); idx+=invectorLength)
		{
			h_in_signal[idx].x = (float) (fftLength);
		}
	}

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, inmem_size,
							   cudaMemcpyHostToDevice));

	for (int i = 0; i < profile_count; i++)
	{
		tr.Start();

		//Execute FFT
		switch (type)
		{
		case CUFFT_C2C:
			checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_in_signal, (cufftComplex *)d_output_signal, direction));
			break;
		case CUFFT_Z2Z:
			checkCudaErrors(cufftExecZ2Z(plan, (cufftDoubleComplex *)d_in_signal, (cufftDoubleComplex *)d_output_signal, direction));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n"); 
			cleanup<T, T>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}

		cudaDeviceSynchronize();
		wtime_t += tr.Sample();

		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = fftLength;
	double opsconst = 5.0 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));

    // Copy device memory to host
    checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, outmem_size,
                               cudaMemcpyDeviceToHost));

    // check result
	bool bTestResult = true;
	// check result only if profile count == 1 or outplace transform
	if (profile_count == 1 || outPlace)
	{
		int idx = 0;
		for (int k = 0; k < batch_size; k++)
		{
			idx = k * outvectorLength;

			for( int i = idx; i < (idx + fftLength*ostride); i += ostride)
			{
				if (direction == CUFFT_FORWARD)
				{
					if (0 == (i % outvectorLength))
					{
						if (h_output_signal[i].x != fftLength)
						{
							printf("fail at %d, value %f\n", i, h_output_signal[i].x);
							bTestResult = false;
							break;
						}
					}
					else
					{
						if (h_output_signal[i].x != 0)
						{
							printf("fail at %d, value %f\n", i, h_output_signal[i].x);
							bTestResult = false;
							break;
						}
					}

					if (h_output_signal[ i ].y != 0)
					{
						printf("fail at %d, value %f\n", i, h_output_signal[i].y);
						bTestResult = false;
						break;
					}
				}
				else
				{
					/*****************************************************************************************
					* cuFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set 
					* followed by an inverse FFT on the resulting set yields data that is equal to the input, 
					* scaled by the number of elements. Scaling either transform by the reciprocal of the size 
					* of the data set is left for the user to perform as seen fit
					******************************************************************************************/
					//Inverse FFT
					if (h_output_signal[i].x != fftLength)
					{
						printf("fail at %d, value %f\n", i, h_output_signal[i].x);
						bTestResult = false;
						break;
					}
					if (h_output_signal[ i ].y != 0)
					{
						printf("fail at %d, value %f\n", i, h_output_signal[i].y);
						bTestResult = false;
						break;
					}
				}
			}
		}
	}
	
    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<T, T>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}

/*
* Run 3D C2R FFT - Basic Data Layout
*/
template<typename In, typename Out>
bool run3DFFTTransformC2R(size_t lengthX, size_t lengthY, size_t lengthZ, cufftType type, int profile_count, int batch_size, bool outPlace)
{
	int rank = 3; //3D Transform

	// Allocate host memory for the signal
	// cuFFT only supports Hermitian Interleaved for input for C2R
	size_t invectorLength = lengthX * lengthY * (lengthZ/2 + 1);
	size_t innerDimLength = outPlace ? lengthZ : lengthZ + 2;
	size_t outvectorLength = lengthX * lengthY * innerDimLength;
	size_t fftLength = lengthX * lengthY * lengthZ;

	int n[3] = {(int)lengthX, (int)lengthY, (int)lengthZ};

	size_t inmem_size = sizeof(In) * (invectorLength) * batch_size;
	size_t outmem_size = sizeof(Out) * (outvectorLength) * batch_size;

    In *h_in_signal = (In *)malloc(inmem_size);
	In *d_in_signal;
	cufftHandle plan;

	Out *h_output_signal; 
	Out *d_output_signal;

	Timer tr;
	double wtime_t =0.0;

	// Allocate device memory for signal		
	checkCudaErrors(cudaMalloc((void **)&d_in_signal, inmem_size));

	if (outPlace)
	{
		h_output_signal = (Out*)malloc(outmem_size);

		// Allocate device memory for output signal		
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, outmem_size));
	}
	else
	{
		h_output_signal = (Out*)h_in_signal;
		d_output_signal = (Out *)d_in_signal;
	}

	// CUFFT plan	
	checkCudaErrors(cufftPlanMany(&plan, rank, n,
						NULL, 1, 0, // *inembed, istride, idist
						NULL, 1, 0,  // *onembed, ostride, odist
						type, batch_size));

	// Initalize the memory for the signal
	memset(h_in_signal, 0, inmem_size);
	for (size_t j = 0; j < (invectorLength * batch_size); j+= (invectorLength))
	{
		h_in_signal[j].x = (float) fftLength;
	}

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, inmem_size,
							   cudaMemcpyHostToDevice));

	// Transform signal and kernel
	for( int i = 0; i < profile_count; ++i )
	{
		tr.Start();

		switch (type)
		{
		case CUFFT_Z2D:
			checkCudaErrors(cufftExecZ2D(plan, (cufftDoubleComplex*)d_in_signal, (cufftDoubleReal *)d_output_signal));
			break;
		case CUFFT_C2R:
			checkCudaErrors(cufftExecC2R(plan, (cufftComplex *)d_in_signal, (cufftReal *)d_output_signal));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n");
			cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}

		cudaDeviceSynchronize();
		wtime_t += tr.Sample();
	
		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = fftLength;
	double opsconst = 2.5 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));

    // Copy device memory to host
    checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, outmem_size,
                               cudaMemcpyDeviceToHost));
	
	bool bTestResult = true;
	
	// check result only if profile count == 1 or outplace transform
	if (profile_count == 1 || outPlace)
	{
		size_t idx = 0;
		size_t innerDim = 0;
		for (int i = 0; i < batch_size; i++)
		{
			for (int k = 0; k < outvectorLength; k++)
			{
				//For in-place transform, skip the 2 elements after inner dimension length
				if (!outPlace && (innerDim == (innerDimLength - 2) || innerDim == (innerDimLength - 1)))
				{
					if (innerDim == (innerDimLength - 1))
						innerDim = 0;
					else
						innerDim++;

					idx++;
					continue;
				}

				/*****************************************************************************************
				* cuFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set 
				* followed by an inverse FFT on the resulting set yields data that is equal to the input, 
				* scaled by the number of elements. Scaling either transform by the reciprocal of the size 
				* of the data set is left for the user to perform as seen fit
				******************************************************************************************/
				if (h_output_signal[idx] != fftLength)
				{
					printf("failed at %d, value=%f\n", idx, h_output_signal[idx]);
					bTestResult = false;
					break;
				}
				idx++;
				innerDim++;
			}
			innerDim = 0;
		}
	}

    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}

/*
* Run 3D R2C FFT - Basic Data Layout
*/
template<typename In, typename Out>
bool run3DFFTTransformR2C(size_t lengthX, size_t lengthY, size_t lengthZ, cufftType type, int profile_count, int batch_size, bool outPlace)
{
	int rank = 3; //3D Transform

	int n[3] = {(int)lengthX, (int)lengthY, (int)lengthZ};

		// Allocate host memory for the signal
	// cuFFT only supports Hermitian Interleaved for output of R2C
	// Hence output size is N/2 + 1 complex. So allocate N + 2 real input
	size_t ivectorLength = lengthX * lengthY * (lengthZ+2);
	size_t ovectorLength = lengthX * lengthY * (lengthZ/2 + 1);
	size_t fftLength = lengthX*lengthY*lengthZ;

	size_t mem_size = sizeof(In) * ivectorLength * batch_size;
    In *h_in_signal = (In *)malloc(mem_size);
	In *d_in_signal;

	Out *h_output_signal;
	Out *d_output_signal;

	// Allocate device memory for signal
    checkCudaErrors(cudaMalloc((void **)&d_in_signal, mem_size));
    
	if (outPlace)
	{
		h_output_signal = (Out *)malloc(mem_size);

		// Allocate device memory for signal
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, mem_size));
	}
	else
	{
		h_output_signal =  (Out*) h_in_signal;
		d_output_signal = (Out*) d_in_signal;
	}

	// CUFFT plan
    cufftHandle plan;
	checkCudaErrors(cufftPlanMany(&plan, rank, n,
						NULL, 1, (int)ivectorLength, // *inembed, istride, idist
						NULL, 1, (int)ovectorLength,  // *onembed, ostride, odist
						type, batch_size));
	
	Timer tr;
	double wtime_t = 0.0;
	
	// Initalize the memory for the signal
	for (unsigned int j = 0; j < (ivectorLength * batch_size); ++j)
	{
		h_in_signal[j] = 1;
	}

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, mem_size,
							   cudaMemcpyHostToDevice));
		
	for (int i = 0; i < profile_count; i++)
	{
		// Transform signal and kernel
		tr.Start();

		switch (type)
		{
		case CUFFT_R2C:
			checkCudaErrors(cufftExecR2C(plan, (cufftReal*)d_in_signal, (cufftComplex*)d_output_signal));
			break;
		case CUFFT_D2Z:
			checkCudaErrors(cufftExecD2Z(plan, (cufftDoubleReal*)d_in_signal, (cufftDoubleComplex*)d_output_signal));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n"); 
			cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}

		cudaDeviceSynchronize();
		wtime_t += tr.Sample();

		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = fftLength;
	double opsconst = 2.5 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));

    // Copy device memory to host
    checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, mem_size,
                               cudaMemcpyDeviceToHost));

	bool bTestResult = true;
	// check result only if profile count == 1 or outplace transform
	if (profile_count == 1 || outPlace)
	{
		for( int i = 0; i < ovectorLength*batch_size; ++i )
		{
			//Check real part of 1st element of every batch is equal to length
			if (i == 0 || (0 == (i % ovectorLength)))
			{
				if (h_output_signal[i].x != fftLength)
				{
					bTestResult = false;
					break;
				}
			}
			else
			{
				if (h_output_signal[i].x != 0)
				{
					bTestResult = false;
					break;
				}
			}

			if (h_output_signal[ i ].y != 0)
			{
				bTestResult = false;
				break;
			}
		}
	}

    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}

/*
* Run 3D C2C FFT - Basic Data Layout
*/
template<typename T>
bool run3DFFTTransformC2C(size_t lengthX, size_t lengthY, size_t lengthZ, int direction, cufftType type, int profile_count, int batch_size, bool outPlace)
{
	int rank = 3; //3D transform

	int n[3] = {(int)lengthX, (int)lengthY, (int)lengthZ};
	size_t vectorLength = lengthX * lengthY * lengthZ;
	// Allocate host memory for the signal
	size_t mem_size = sizeof(T) * vectorLength * batch_size;
    T *h_in_signal = (T *)malloc(mem_size);
	
	T *h_output_signal;
	T *d_in_signal;
	T *d_output_signal;

    // Allocate device memory for signal
    checkCudaErrors(cudaMalloc((void **)&d_in_signal, mem_size));

	if (outPlace)
	{
		h_output_signal = (T *)malloc(mem_size);

		// Allocate device memory for out signal
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, mem_size));
	}
	else
	{
		h_output_signal = h_in_signal;
		d_output_signal = d_in_signal;
	}

    // CUFFT plan
    cufftHandle plan;
	checkCudaErrors(cufftPlanMany(&plan, rank, n,
							NULL, 1, (int)vectorLength, // *inembed, istride, idist
							NULL, 1, (int)vectorLength,  // *onembed, ostride, odist
							type, batch_size));

    // Transform signal and kernel
	
	Timer tr;
	double wtime_t = 0.0;

	// Initalize the memory for the signal
	if (direction == CUFFT_FORWARD)
	{
		for (size_t idx = 0; idx < (vectorLength*batch_size); ++idx)
		{
			h_in_signal[idx].x = 1;
			h_in_signal[idx].y = 0;
		}
	}
	else
	{
		//Inverse FFT
		memset(h_in_signal, 0, mem_size);

		for (size_t idx = 0; idx < (vectorLength*batch_size); idx+=vectorLength)
		{
			h_in_signal[idx].x = (float) (vectorLength);
		}
	}

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, mem_size,
							   cudaMemcpyHostToDevice));

	for (int i = 0; i < profile_count; i++)
	{
		tr.Start();

		//Execute FFT
		switch (type)
		{
		case CUFFT_C2C:
			checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_in_signal, (cufftComplex *)d_output_signal, direction));
			break;
		case CUFFT_Z2Z:
			checkCudaErrors(cufftExecZ2Z(plan, (cufftDoubleComplex *)d_in_signal, (cufftDoubleComplex *)d_output_signal, direction));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n"); 
			cleanup<T, T>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}

		cudaDeviceSynchronize();
		wtime_t += tr.Sample();

		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = vectorLength;
	double opsconst = 5.0 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));

    // Copy device memory to host
    checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, mem_size,
                               cudaMemcpyDeviceToHost));

	bool bTestResult = true;
	
	// check result only if profile count == 1 or outplace transform
	if (profile_count == 1 || outPlace)
	{
		for( int i = 0; i < (vectorLength*batch_size); ++i )
		{
			if (direction == CUFFT_FORWARD)
			{
				if (0 == (i % vectorLength))
				{
					if (h_output_signal[i].x != vectorLength)
					{
						bTestResult = false;
						break;
					}
				}
				else
				{
					if (h_output_signal[i].x != 0)
					{
						bTestResult = false;
						break;
					}
				}

				if (h_output_signal[ i ].y != 0)
				{
					bTestResult = false;
					break;
				}
			}
			else
			{
				/*****************************************************************************************
				* cuFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set 
				* followed by an inverse FFT on the resulting set yields data that is equal to the input, 
				* scaled by the number of elements. Scaling either transform by the reciprocal of the size 
				* of the data set is left for the user to perform as seen fit
				******************************************************************************************/
				//Inverse FFT
				if (h_output_signal[i].x != vectorLength)
				{
					bTestResult = false;
					break;
				}
				if (h_output_signal[ i ].y != 0)
				{
					bTestResult = false;
					break;
				}
			}
		}
	}
	
	//Destroy CUFFT context
	checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<T, T>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}

/*
* Run 2D C2R FFT - Basic Data Layout
*/
template<typename In, typename Out>
bool run2DFFTTransformC2R(size_t lengthX, size_t lengthY, cufftType type, int profile_count, int batch_size, bool outPlace)
{
	int rank = 2; //2D Transform

	// Allocate host memory for the signal
	// cuFFT only supports Hermitian Interleaved for input for C2R
	size_t invectorLength = lengthX * (lengthY/2 + 1);
	size_t innerDimLength = outPlace ? lengthY : (lengthY + 2);
	size_t outvectorLength = lengthX * innerDimLength;

	size_t fftLength = lengthX * lengthY;

	int n[2] = {(int)lengthX, (int)lengthY};

	size_t inmem_size = sizeof(In) * (invectorLength) * batch_size;
	size_t outmem_size = sizeof(Out) * (outvectorLength) * batch_size;

    In *h_in_signal = (In *)malloc(inmem_size);
	In *d_in_signal;
	cufftHandle plan;

	Out *h_output_signal; 
	Out *d_output_signal;

	Timer tr;
	double wtime_t =0.0;

	// Allocate device memory for signal		
	checkCudaErrors(cudaMalloc((void **)&d_in_signal, inmem_size));

	if (outPlace)
	{
		h_output_signal = (Out*)malloc(outmem_size);

		// Allocate device memory for output signal		
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, outmem_size));
	}
	else
	{
		h_output_signal = (Out*)h_in_signal;
		d_output_signal = (Out *)d_in_signal;
	}

	// CUFFT plan	
	checkCudaErrors(cufftPlanMany(&plan, rank, n,
						NULL, 1, 0, // *inembed, istride, idist
						NULL, 1, 0,  // *onembed, ostride, odist
						type, batch_size));

	// Initalize the memory for the signal
	memset(h_in_signal, 0, inmem_size);
	for (size_t j = 0; j < (invectorLength * batch_size); j+= invectorLength)
	{
		h_in_signal[j].x = (float)fftLength;
	}
	
	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, inmem_size,
							   cudaMemcpyHostToDevice));
	
	// Transform signal and kernel
	for( int i = 0; i < profile_count; ++i )
	{
		tr.Start();

		switch (type)
		{
		case CUFFT_Z2D:
			checkCudaErrors(cufftExecZ2D(plan, (cufftDoubleComplex*)d_in_signal, (cufftDoubleReal *)d_output_signal));
			break;
		case CUFFT_C2R:
			checkCudaErrors(cufftExecC2R(plan, (cufftComplex *)d_in_signal, (cufftReal *)d_output_signal));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n"); 
			cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}

		cudaDeviceSynchronize();
		wtime_t += tr.Sample();
	
		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = fftLength;
	double opsconst = 2.5 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));

    // Copy device memory to host
    checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, outmem_size,
                               cudaMemcpyDeviceToHost));
	
	bool bTestResult = true;
	
	// check result only if profile count == 1 or outplace transform
	if (profile_count == 1 || outPlace)
	{
		int idx = 0;
		for (int i = 0; i < batch_size; i++)
		{
			for( int j = 0; j < lengthX; ++j )
			{
				for (int k = 0; k < lengthY; k++)
				{
					idx = i*outvectorLength + j*innerDimLength + k;

					/*****************************************************************************************
					* cuFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set 
					* followed by an inverse FFT on the resulting set yields data that is equal to the input, 
					* scaled by the number of elements. Scaling either transform by the reciprocal of the size 
					* of the data set is left for the user to perform as seen fit
					******************************************************************************************/
					if (h_output_signal[idx] != (float)fftLength)
					{
						printf("failed at %d, value=%f\n", idx, h_output_signal[idx] );
						bTestResult = false;
						break;				
					}
				}
			}
		}
	}

    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}

/*
* Run 2D R2C FFT - Basic Data Layout
*/
template<typename In, typename Out>
bool run2DFFTTransformR2C(size_t lengthX, size_t lengthY, cufftType type, int profile_count, int batch_size, bool outPlace)
{
	int rank = 2; //2D Transform

	int n[2] = {(int)lengthX, (int)lengthY};
	size_t ivectorLength = lengthX * (lengthY+2);
	size_t ovectorLength = lengthX * (lengthY/2 + 1);
	size_t fftLength = lengthX*lengthY;

	// Allocate host memory for the signal
	// cuFFT only supports Hermitian Interleaved for output of R2C
	// Hence output size is N/2 + 1 complex. So allocate N + 2 real input
	size_t mem_size = sizeof(In) * ivectorLength * batch_size;
    In *h_in_signal = (In *)malloc(mem_size);
	In *d_in_signal;

	Out *h_output_signal;
	Out *d_output_signal;

	// Allocate device memory for signal
    checkCudaErrors(cudaMalloc((void **)&d_in_signal, mem_size));
    
	if (outPlace)
	{
		h_output_signal = (Out *)malloc(mem_size);

		// Allocate device memory for signal
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, mem_size));
	}
	else
	{
		h_output_signal =  (Out*) h_in_signal;
		d_output_signal = (Out*) d_in_signal;
	}

	// CUFFT plan
    cufftHandle plan;
	checkCudaErrors(cufftPlanMany(&plan, rank, n,
							NULL, 1, (int)ivectorLength, // *inembed, istride, idist
							NULL, 1, (int)ovectorLength,  // *onembed, ostride, odist
							type, batch_size));

	// Initalize the memory for the signal
	for (unsigned int j = 0; j < (ivectorLength * batch_size); ++j)
	{
		h_in_signal[j] = 1;
	}

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, mem_size,
							   cudaMemcpyHostToDevice));
	
	Timer tr;
	double wtime_t = 0.0;
			
	for (int i = 0; i < profile_count; i++)
	{
		// Transform signal and kernel
		tr.Start();

		switch (type)
		{
		case CUFFT_D2Z:
			checkCudaErrors(cufftExecD2Z(plan, (cufftDoubleReal*)d_in_signal, (cufftDoubleComplex*)d_output_signal));
			break;
		case CUFFT_R2C:
			checkCudaErrors(cufftExecR2C(plan, (cufftReal*)d_in_signal, (cufftComplex*)d_output_signal));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n"); 
			cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}

		cudaDeviceSynchronize();
		wtime_t += tr.Sample();

		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = fftLength;
	double opsconst = 2.5 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));

    // Copy device memory to host
    checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, mem_size,
                               cudaMemcpyDeviceToHost));

    	bool bTestResult = true;
	
	// check result only if profile count == 1 or outplace transform
	if (profile_count == 1 || outPlace)
	{
		for( int i = 0; i < ovectorLength*batch_size; ++i )
		{
			//Check real part of 1st element of every batch is equal to length
			if (i == 0 || (0 == (i % ovectorLength)))
			{
				if (h_output_signal[i].x != fftLength)
				{
					bTestResult = false;
					break;
				}
			}
			else
			{
				if (h_output_signal[i].x != 0)
				{
					bTestResult = false;
					break;
				}
			}

			if (h_output_signal[ i ].y != 0)
			{
				bTestResult = false;
				break;
			}
		}
	}

    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}

/*
* Run 2D C2C FFT - Basic Data layout
*/
template<typename T>
bool run2DFFTTransformC2C(size_t lengthX, size_t lengthY, int direction, cufftType type, int profile_count, int batch_size, bool outPlace)
{
	int rank = 2; //2D transform

	int n[2] = {(int)lengthX, (int)lengthY};
	size_t vectorLength = lengthX * lengthY;
	// Allocate host memory for the signal
	size_t mem_size = sizeof(T) * vectorLength * batch_size;
    T *h_in_signal = (T *)malloc(mem_size);
	
	T *h_output_signal;
	T *d_in_signal;
	T *d_output_signal;

    // Allocate device memory for signal
    checkCudaErrors(cudaMalloc((void **)&d_in_signal, mem_size));

	if (outPlace)
	{
		h_output_signal = (T *)malloc(mem_size);

		// Allocate device memory for out signal
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, mem_size));
	}
	else
	{
		h_output_signal = h_in_signal;
		d_output_signal = d_in_signal;
	}

    // CUFFT plan
    cufftHandle plan;
	checkCudaErrors(cufftPlanMany(&plan, rank, n,
							NULL, 1, (int)vectorLength, // *inembed, istride, idist
							NULL, 1, (int)vectorLength,  // *onembed, ostride, odist
							type, batch_size));

	// Initalize the memory for the signal
	if (direction == CUFFT_FORWARD)
	{
		for (size_t idx = 0; idx < (vectorLength*batch_size); ++idx)
		{
			h_in_signal[idx].x = 1;
			h_in_signal[idx].y = 0;
		}
	}
	else
	{
		//Inverse FFT
		memset(h_in_signal, 0, mem_size);

		for (size_t idx = 0; idx < (vectorLength*batch_size); idx+=vectorLength)
		{
			h_in_signal[idx].x = (float) (vectorLength);
		}
	}

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, mem_size,
							   cudaMemcpyHostToDevice));
	
	Timer tr;
	double wtime_t = 0.0;

	for (int i = 0; i < profile_count; i++)
	{
		tr.Start();

		//Execute FFT
		switch (type)
		{
		case CUFFT_C2C:
			checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_in_signal, (cufftComplex *)d_output_signal, direction));
			break;
		case CUFFT_Z2Z:
			checkCudaErrors(cufftExecZ2Z(plan, (cufftDoubleComplex *)d_in_signal, (cufftDoubleComplex *)d_output_signal, direction));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n"); 
			cleanup<T, T>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}

		cudaDeviceSynchronize();
		wtime_t += tr.Sample();

		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = vectorLength;
	double opsconst = 5.0 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));

    // Copy device memory to host
    checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, mem_size,
                               cudaMemcpyDeviceToHost));

	bool bTestResult = true;

	// check result only if profile count == 1 or outplace transform
	if (profile_count == 1 || outPlace)
	{
		for( int i = 0; i < (vectorLength*batch_size); ++i )
		{
			if (direction == CUFFT_FORWARD)
			{
				if (0 == (i % vectorLength))
				{
					if (h_output_signal[i].x != vectorLength)
					{
						bTestResult = false;
						break;
					}
				}
				else
				{
					if (h_output_signal[i].x != 0)
					{
						bTestResult = false;
						break;
					}
				}

				if (h_output_signal[ i ].y != 0)
				{
					bTestResult = false;
					break;
				}
			}
			else
			{
				/*****************************************************************************************
				* cuFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set 
				* followed by an inverse FFT on the resulting set yields data that is equal to the input, 
				* scaled by the number of elements. Scaling either transform by the reciprocal of the size 
				* of the data set is left for the user to perform as seen fit
				******************************************************************************************/
				//Inverse FFT
				if (h_output_signal[i].x != vectorLength)
				{
					bTestResult = false;
					break;
				}
				if (h_output_signal[ i ].y != 0)
				{
					bTestResult = false;
					break;
				}
			}
		}
	}
	
    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<T, T>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}

/*
* Run 1D C2R FFT - Basic Data Layout
*/
template<typename In, typename Out>
bool run1DFFTTransformC2R(size_t lengthX, cufftType type, int profile_count, int batch_size, bool outPlace)
{
	int rank = 1; //1D Transform

	// Allocate host memory for the signal
	// cuFFT only supports Hermitian Interleaved for input for C2R
	size_t invectorLength = lengthX/2 + 1;
	size_t outvectorLength = outPlace ? lengthX  : lengthX + 2;
	size_t fftLength = lengthX;
	
	size_t inmem_size = sizeof(In) * (invectorLength) * batch_size;
	size_t outmem_size = sizeof(Out) * (outvectorLength) * batch_size;

    In *h_in_signal = (In *)malloc(inmem_size);
	In *d_in_signal;
	cufftHandle plan;

	Out *h_output_signal; 
	Out *d_output_signal;

	Timer tr;
	double wtime_t =0.0;

	// Allocate device memory for signal		
	checkCudaErrors(cudaMalloc((void **)&d_in_signal, inmem_size));

	if (outPlace)
	{
		h_output_signal = (Out*)malloc(outmem_size);

		// Allocate device memory for output signal		
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, outmem_size));
	}
	else
	{
		h_output_signal = (Out*)h_in_signal;
		d_output_signal = (Out *)d_in_signal;
	}

	// CUFFT plan	
	checkCudaErrors(cufftPlanMany(&plan, rank, (int*)&lengthX,
						NULL, 1, 0, // *inembed, istride, idist
						NULL, 1, 0,  // *onembed, ostride, odist
						type, batch_size));

	// Initalize the memory for the signal
	memset(h_in_signal, 0, inmem_size);
	for (size_t j = 0; j < (invectorLength * batch_size); j+= (invectorLength))
	{
		h_in_signal[j].x = (float) fftLength;
	}

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, inmem_size,
							   cudaMemcpyHostToDevice));

	for( int i = 0; i < profile_count; ++i )
	{

		tr.Start();

		switch (type)
		{
		case CUFFT_Z2D:
			checkCudaErrors(cufftExecZ2D(plan, (cufftDoubleComplex*)d_in_signal, (cufftDoubleReal *)d_output_signal));
			break;
		case CUFFT_C2R:
			checkCudaErrors(cufftExecC2R(plan, (cufftComplex *)d_in_signal, (cufftReal *)d_output_signal));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n"); 
			cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}

		cudaDeviceSynchronize();
		wtime_t += tr.Sample();
	
		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = lengthX;
	double opsconst = 2.5 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));

    // Copy device memory to host
    checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, outmem_size,
                               cudaMemcpyDeviceToHost));
	
	bool bTestResult = true;
	
	// check result only if profile count == 1 or outplace transform
	if (profile_count == 1 || outPlace)
	{
		int idx = 0;
		for (int i = 0; i < batch_size; i++)
		{
			for( int j = 0; j < lengthX; ++j )
			{
				idx = i*outvectorLength +  j;
				/*****************************************************************************************
				* cuFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set 
				* followed by an inverse FFT on the resulting set yields data that is equal to the input, 
				* scaled by the number of elements. Scaling either transform by the reciprocal of the size 
				* of the data set is left for the user to perform as seen fit
				******************************************************************************************/
				if (h_output_signal[idx] != fftLength)
				{
					printf("failed at %d, value=%f\n", idx,h_output_signal[idx]);
					bTestResult = false;
					break;
				}
			}
		}
	}

    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}

/*
* Run 1D R2C FFT - Basic Data Layout
*/
template<typename In, typename Out>
bool run1DFFTTransformR2C(size_t lengthX, cufftType type, int profile_count, int batch_size, bool outPlace)
{
	int rank = 1; //1D Transform

	// Allocate host memory for the signal
	// cuFFT only supports Hermitian Interleaved for output of R2C
	// Hence output size is N/2 + 1 complex. So allocate N + 2 real input
	size_t mem_size = sizeof(In) * (lengthX+2) * batch_size;
    In *h_in_signal = (In *)malloc(mem_size);
	In *d_in_signal;

	Out *h_output_signal;
	Out *d_output_signal;

	// Allocate device memory for signal
    checkCudaErrors(cudaMalloc((void **)&d_in_signal, mem_size));
    
	if (outPlace)
	{
		h_output_signal = (Out *)malloc(mem_size);

		// Allocate device memory for signal
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, mem_size));
	}
	else
	{
		h_output_signal =  (Out*) h_in_signal;
		d_output_signal = (Out*) d_in_signal;
	}

	// CUFFT plan
    cufftHandle plan;
	checkCudaErrors(cufftPlanMany(&plan, rank, (int*)&lengthX,
							NULL, 1, (int)(lengthX+2), // *inembed, istride, idist
							NULL, 1, (int)(lengthX/2 + 1),  // *onembed, ostride, odist
							type, batch_size));

	// Initalize the memory for the signal
	for (unsigned int j = 0; j < ((lengthX+2) * batch_size); ++j)
	{
		h_in_signal[j] = 1;
	}

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, mem_size,
							   cudaMemcpyHostToDevice));
	
	Timer tr;
	double wtime_t = 0.0;
			
	for (int i = 0; i < profile_count; i++)
	{
		// Transform signal and kernel
		tr.Start();

		switch (type)
		{
		case CUFFT_D2Z:
			checkCudaErrors(cufftExecD2Z(plan, (cufftDoubleReal*)d_in_signal, (cufftDoubleComplex*)d_output_signal));
			break;
		case CUFFT_R2C:
			checkCudaErrors(cufftExecR2C(plan, (cufftReal*)d_in_signal, (cufftComplex*)d_output_signal));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n"); 
			cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}

		cudaDeviceSynchronize();
		wtime_t += tr.Sample();

		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = lengthX;
	double opsconst = 2.5 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));

    // Copy device memory to host
    checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, mem_size,
                               cudaMemcpyDeviceToHost));

    //// check result
	printf("h_output_signal[0].x=%f, h_output_signal[0].y=%f \n", h_output_signal[0].x, h_output_signal[0].y);
	printf("h_output_signal[1].x=%f, h_output_signal[1].y=%f \n", h_output_signal[1].x, h_output_signal[1].y);
	if (batch_size > 1)
	{
		printf("h_output_signal[lengthX/2 + 1].x=%f, h_output_signal[lengthX/2 + 1].y=%f \n", h_output_signal[lengthX/2 + 1].x, h_output_signal[lengthX/2 + 1].y);
	}

	bool bTestResult = true;
	// check result only if profile count == 1 or outplace transform
	if (profile_count == 1 || outPlace)
	{
		for( int i = 0; i < (lengthX/2 + 1)*batch_size; ++i )
		{
			//Check real part of 1st element of every batch is equal to length
			if (i == 0 || (0 == (i % (lengthX/2 + 1))))
			{
				if (h_output_signal[i].x != lengthX)
				{
					bTestResult = false;
					break;
				}
			}
			else
			{
				if (h_output_signal[i].x != 0)
				{
					bTestResult = false;
					break;
				}
			}

			if (h_output_signal[ i ].y != 0)
			{
				bTestResult = false;
				break;
			}
		}
	}

    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<In, Out>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}

/*
* Run 1D C2C FFT - Basic Data Layout
*/
template<typename T>
bool run1DFFTTransformC2C(size_t lengthX, int direction, cufftType type, int profile_count, int batch_size, bool outPlace)
{
	int rank = 1; //1D transform

	// Allocate host memory for the signal
	size_t mem_size = sizeof(T) * lengthX * batch_size;
    T *h_in_signal = (T *)malloc(mem_size);
	
	T *h_output_signal;
	T *d_in_signal;
	T *d_output_signal;

    // Allocate device memory for signal
    checkCudaErrors(cudaMalloc((void **)&d_in_signal, mem_size));

	if (outPlace)
	{
		h_output_signal = (T *)malloc(mem_size);

		// Allocate device memory for out signal
		checkCudaErrors(cudaMalloc((void **)&d_output_signal, mem_size));
	}
	else
	{
		h_output_signal = h_in_signal;
		d_output_signal = d_in_signal;
	}

    // CUFFT plan
    cufftHandle plan;
	checkCudaErrors(cufftPlanMany(&plan, rank, (int*)&lengthX,
							NULL, 1, (int)lengthX, // *inembed, istride, idist
							NULL, 1, (int)lengthX,  // *onembed, ostride, odist
							type, batch_size));
	

	// Initalize the memory for the signal
	if (direction == CUFFT_FORWARD)
	{
		for (size_t idx = 0; idx < (lengthX*batch_size); ++idx)
		{
			h_in_signal[idx].x = 1;
			h_in_signal[idx].y = 0;
		}
	}
	else
	{
		//Inverse FFT
		memset(h_in_signal, 0, mem_size);

		for (size_t idx = 0; idx < (lengthX*batch_size); idx+=lengthX)
		{
			h_in_signal[idx].x = (float) (lengthX);
		}
	}

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_in_signal, h_in_signal, mem_size,
							   cudaMemcpyHostToDevice));
	
	Timer tr;
	double wtime_t = 0.0;

	for (int i = 0; i < profile_count; i++)
	{
		tr.Start();

		//Execute FFT
		switch (type)
		{
		case CUFFT_C2C:
			checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_in_signal, (cufftComplex *)d_output_signal, direction));
			break;
		case CUFFT_Z2Z:
			checkCudaErrors(cufftExecZ2Z(plan, (cufftDoubleComplex *)d_in_signal, (cufftDoubleComplex *)d_output_signal, direction));
			break;
		default: 
			printf("Input Arguments ERROR!! Invalid FFT type. Exiting..\n"); 
			cleanup<T, T>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);
			exit(EXIT_FAILURE);
		}

		cudaDeviceSynchronize();
		wtime_t += tr.Sample();

		//Ignore the first time sample if profiling for more than one iteration
		if (i == 0 && profile_count > 1) wtime_t = 0.0;
	}

	double iter = (double)( profile_count > 1 ? (profile_count - 1) : profile_count);
	double wtime = wtime_t/iter;
	size_t totalLen = 1;
	totalLen = lengthX;
	double opsconst = 5.0 * (double)totalLen * log((double)totalLen) / log(2.0);

	printf("\nExecution wall time: %lf ms\n", 1000.0*wtime);
	printf("Execution gflops: %lf \n", ((double)batch_size * opsconst)/(1000000000.0*wtime));
	
    // Copy device memory to host
    //cufftComplex *h_output_signal = h_signal;
    checkCudaErrors(cudaMemcpy(h_output_signal, d_output_signal, mem_size,
                               cudaMemcpyDeviceToHost));

	bool bTestResult = true;
	// check result only if profile count == 1 or outplace transform
	if (profile_count == 1 || outPlace)
	{
		for( int i = 0; i < (lengthX*batch_size); ++i )
		{
			if (direction == CUFFT_FORWARD)
			{
				if (0 == (i % lengthX))
				{
					if (h_output_signal[i].x != lengthX)
					{
						bTestResult = false;
						break;
					}
				}
				else
				{
					if (h_output_signal[i].x != 0)
					{
						bTestResult = false;
						break;
					}
				}

				if (h_output_signal[ i ].y != 0)
				{
					bTestResult = false;
					break;
				}
			}
			else
			{
				/*****************************************************************************************
				* cuFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set 
				* followed by an inverse FFT on the resulting set yields data that is equal to the input, 
				* scaled by the number of elements. Scaling either transform by the reciprocal of the size 
				* of the data set is left for the user to perform as seen fit
				******************************************************************************************/
				//Inverse FFT
				if (h_output_signal[i].x != lengthX)
				{
					bTestResult = false;
					break;
				}
				if (h_output_signal[ i ].y != 0)
				{
					bTestResult = false;
					break;
				}
			}
		}
	}

    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
	cleanup<T, T>(h_in_signal, d_in_signal, h_output_signal, d_output_signal, outPlace);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	return bTestResult;
}


/*
* Clean up resources
*/
template<typename In, typename Out>
void cleanup(In *h_in_signal, In *d_in_signal, Out *h_output_signal, Out *d_output_signal, bool outPlace)
{
	    // cleanup memory
	if (h_in_signal)
		free(h_in_signal);
	if (h_output_signal && outPlace)
		free(h_output_signal);
    if (d_in_signal)
		checkCudaErrors(cudaFree(d_in_signal));
	if (d_output_signal && outPlace)
		checkCudaErrors(cudaFree(d_output_signal));
}

