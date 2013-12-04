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


// clfft.transform.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "private.h"
#include "repo.h"
#include "plan.h"

//#define DEBUGGING

using std::vector;

clfftStatus clfftEnqueueTransform(
											clfftPlanHandle plHandle,
											clfftDirection dir,
											cl_uint numQueuesAndEvents,
											cl_command_queue* commQueues,
											cl_uint numWaitEvents,
											const cl_event* waitEvents,
											cl_event* outEvents,
											cl_mem* clInputBuffers,
											cl_mem* clOutputBuffers,
											cl_mem clTmpBuffers
											)
{
	cl_int status = CLFFT_SUCCESS;

	//	We do not currently support multiple command queues, which is necessary to support multi-gpu operations
	if( numQueuesAndEvents > 1 )
	{
		return CLFFT_NOTIMPLEMENTED;
	}

	FFTRepo& fftRepo	= FFTRepo::getInstance( );
	FFTPlan* fftPlan	= NULL;
	lockRAII* planLock	= NULL;

	//	At this point, the user wants to enqueue a plan to execute.  We lock the plan down now, such that
	//	after we finish baking the plan (if the user did not do that explicitely before), the plan cannot
	//	change again through the action of other thread before we enqueue this plan for execution.
	OPENCL_V( fftRepo.getPlan( plHandle, fftPlan, planLock ), _T( "fftRepo.getPlan failed" ) );
	scopedLock sLock( *planLock, _T( "clfftGetPlanBatchSize" ) );

	if( fftPlan->baked == false )
	{
		OPENCL_V( clfftBakePlan( plHandle, numQueuesAndEvents, commQueues, NULL, NULL ), _T( "Failed to bake plan" ) );
	}

	if		(fftPlan->inputLayout == CLFFT_REAL)	dir = CLFFT_FORWARD;
	else if	(fftPlan->outputLayout == CLFFT_REAL)	dir = CLFFT_BACKWARD;


	// we do not check the user provided buffer at this release
	cl_mem localIntBuffer = clTmpBuffers;

	if( clTmpBuffers == NULL && fftPlan->tmpBufSize > 0 && fftPlan->intBuffer == NULL)
	{
		// create the intermediate buffers
		// The intermediate buffer is always interleave and packed
		// For outofplace operation, we have the choice not to create intermediate buffer
		// input ->(col+Transpose) output ->(col) output
		fftPlan->intBuffer = clCreateBuffer( fftPlan->context, CL_MEM_READ_WRITE,
			fftPlan->tmpBufSize, 0, &status);
		OPENCL_V( status, _T("Creating the intermediate buffer for large1D Failed") );

#if defined(DEBUGGING)
		std::cout << "One intermediate buffer is created" << std::endl;
#endif
	}

	if( localIntBuffer == NULL && fftPlan->intBuffer != NULL )
		localIntBuffer = fftPlan->intBuffer;

	if( fftPlan->intBufferRC == NULL && fftPlan->tmpBufSizeRC > 0 )
	{
		fftPlan->intBufferRC = clCreateBuffer( fftPlan->context, CL_MEM_READ_WRITE, fftPlan->tmpBufSizeRC, 0, &status);
		OPENCL_V( status, _T("Creating the intermediate buffer for large1D RC Failed") );
	}

	if( fftPlan->intBufferC2R == NULL && fftPlan->tmpBufSizeC2R > 0 )
	{
		fftPlan->intBufferC2R = clCreateBuffer( fftPlan->context, CL_MEM_READ_WRITE, fftPlan->tmpBufSizeC2R, 0, &status);
		OPENCL_V( status, _T("Creating the intermediate buffer for large1D YZ C2R Failed") );
	}

	//	The largest vector we can transform in a single pass
	//	depends on the GPU caps -- especially the amount of LDS
	//	available
	//
	size_t Large1DThreshold = 0;
	OPENCL_V(fftPlan->GetMax1DLength (&Large1DThreshold), _T("GetMax1DLength failed"));
	BUG_CHECK (Large1DThreshold > 1);

	//Large1DThreshold = 128;

	if(fftPlan->gen != Copy)
	switch( fftPlan->dim )
	{
		case CLFFT_1D:
		{
			if (fftPlan->length[0] <= Large1DThreshold)
				break;

			if( fftPlan->inputLayout == CLFFT_REAL )
			{
				cl_event colOutEvents = NULL;
				cl_event copyInEvents = NULL;

				// First pass
				// column with twiddle first, OUTOFPLACE, + transpose
				OPENCL_V( clfftEnqueueTransform( fftPlan->planX, CLFFT_FORWARD, numQueuesAndEvents, commQueues, numWaitEvents,
					waitEvents, &colOutEvents, clInputBuffers, &(fftPlan->intBufferRC), localIntBuffer),
					_T("clfftEnqueueTransform large1D col pass failed"));


				// another column FFT output, INPLACE
				OPENCL_V( clfftEnqueueTransform( fftPlan->planY, CLFFT_FORWARD, numQueuesAndEvents, commQueues, 1, &colOutEvents,
					&copyInEvents, &(fftPlan->intBufferRC), &(fftPlan->intBufferRC), localIntBuffer ),
					_T("clfftEnqueueTransform large1D second column failed"));
				clReleaseEvent(colOutEvents);

				cl_mem *out_local;
				out_local = (fftPlan->placeness==CLFFT_INPLACE) ? clInputBuffers : clOutputBuffers;

				// copy from full complex to hermitian
				OPENCL_V( clfftEnqueueTransform( fftPlan->planRCcopy, CLFFT_FORWARD, numQueuesAndEvents, commQueues, 1, &copyInEvents,
					outEvents, &(fftPlan->intBufferRC), out_local, localIntBuffer ),
					_T("clfftEnqueueTransform large1D RC copy failed"));
				clReleaseEvent(copyInEvents);

				return	CLFFT_SUCCESS;

			}
			else if( fftPlan->outputLayout == CLFFT_REAL )
			{
				cl_event colOutEvents = NULL;
				cl_event copyOutEvents = NULL;

				// copy from hermitian to full complex
				OPENCL_V( clfftEnqueueTransform( fftPlan->planRCcopy, CLFFT_BACKWARD, numQueuesAndEvents, commQueues, numWaitEvents,
					waitEvents, &copyOutEvents, clInputBuffers, &(fftPlan->intBufferRC), localIntBuffer ),
					_T("clfftEnqueueTransform large1D RC copy failed"));

				// First pass
				// column with twiddle first, INPLACE,
				OPENCL_V( clfftEnqueueTransform( fftPlan->planX, CLFFT_BACKWARD, numQueuesAndEvents, commQueues, 1,
					&copyOutEvents, &colOutEvents, &(fftPlan->intBufferRC), &(fftPlan->intBufferRC), localIntBuffer),
					_T("clfftEnqueueTransform large1D col pass failed"));
				clReleaseEvent(copyOutEvents);

				cl_mem *out_local;
				out_local = (fftPlan->placeness==CLFFT_INPLACE) ? clInputBuffers : clOutputBuffers;

				// another column FFT output, OUTOFPLACE + transpose
				OPENCL_V( clfftEnqueueTransform( fftPlan->planY, CLFFT_BACKWARD, numQueuesAndEvents, commQueues, 1, &colOutEvents,
					outEvents, &(fftPlan->intBufferRC), out_local, localIntBuffer ),
					_T("clfftEnqueueTransform large1D second column failed"));
				clReleaseEvent(colOutEvents);


				return	CLFFT_SUCCESS;
			}
			else
			{
#if defined(DEBUGGING)
				// For debugging interleave data only, initialize the intermediate buffer
				// to a data pattern.  This will show which data in the buffer
				// are being written by the kernel
				//
				size_t buffSizeBytes_complex = fftPlan->tmpBufSize;
				size_t buffersize = buffSizeBytes_complex/sizeof( std::complex< float > );
				std::vector<std::complex< float> > temp(buffersize);

				for (size_t u = 0; u < buffersize; ++u) {
					temp[u] = std::complex<float> (float(u+1), float(buffersize-u));
				}

				if (fftPlan->large1D == 0)
				{
					//First time usage, we can initialize tmp buffer
					OPENCL_V(clEnqueueWriteBuffer( *commQueues,
						localIntBuffer,
						1,		// blocking write
						0,
						buffSizeBytes_complex,
						&temp[0],
						0,
						NULL,
						NULL), _T("clEnqueueWriteBuffer failed") );
				}
#endif

				if (fftPlan->transflag)
				{
					//First transpose
					// Input->tmp
					cl_event transTXOutEvents = NULL;
					OPENCL_V( clfftEnqueueTransform( fftPlan->planTX, dir, numQueuesAndEvents, commQueues, numWaitEvents,
						waitEvents, &transTXOutEvents, clInputBuffers, &localIntBuffer, NULL ),
						_T("clfftEnqueueTransform for large1D transTX failed"));

					cl_mem *mybuffers;
					if (fftPlan->placeness==CLFFT_INPLACE)
						mybuffers = clInputBuffers;
					else
						mybuffers = clOutputBuffers;

					//First Row
					//tmp->output
					cl_event rowXOutEvents = NULL;
					OPENCL_V( clfftEnqueueTransform( fftPlan->planX, dir, numQueuesAndEvents, commQueues, 1,
						&transTXOutEvents, &rowXOutEvents, &localIntBuffer, mybuffers, NULL ),
						_T("clfftEnqueueTransform for large1D rowX failed"));
					clReleaseEvent(transTXOutEvents);

					//Second Transpose
					// output->tmp
					cl_event transTYOutEvents = NULL;
					OPENCL_V( clfftEnqueueTransform( fftPlan->planTY, dir, numQueuesAndEvents, commQueues, 1,
						&rowXOutEvents, &transTYOutEvents, mybuffers, &localIntBuffer, NULL ),
						_T("clfftEnqueueTransform for large1D transTY failed"));
					clReleaseEvent(rowXOutEvents);

					//Second Row
					//tmp->tmp, inplace
					cl_event rowYOutEvents = NULL;
					OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1,
						&transTYOutEvents, &rowYOutEvents, &localIntBuffer, NULL, NULL ),
						_T("clfftEnqueueTransform for large1D rowY failed"));
					clReleaseEvent(transTYOutEvents);

					//Third Transpose
					// tmp->output
					OPENCL_V( clfftEnqueueTransform( fftPlan->planTZ, dir, numQueuesAndEvents, commQueues, 1,
						&rowYOutEvents, outEvents, &localIntBuffer, mybuffers, NULL ),
						_T("clfftEnqueueTransform for large1D transTZ failed"));
					clReleaseEvent(rowYOutEvents);

					if( fftRepo.pStatTimer )
					{
						fftRepo.pStatTimer->AddSample( plHandle, fftPlan, NULL, 0, NULL, std::vector< size_t >( ) );
					}

					return	CLFFT_SUCCESS;
				}

				cl_event colOutEvents = NULL;
				if (fftPlan->large1D == 0)
				{
					// First pass
					// column with twiddle first, OUTOFPLACE, + transpose
					OPENCL_V( clfftEnqueueTransform( fftPlan->planX, dir, numQueuesAndEvents, commQueues, numWaitEvents,
						waitEvents, &colOutEvents, clInputBuffers, &localIntBuffer, localIntBuffer),
						_T("clfftEnqueueTransform large1D col pass failed"));

#if defined(DEBUGGING)
					// debug purpose, interleave input <-> interleave output
					// read the intermediate buffer and print part of it.
					OPENCL_V( clEnqueueReadBuffer( *commQueues, localIntBuffer, CL_TRUE, 0, buffSizeBytes_complex, &temp[ 0 ], 1,
						&colOutEvents, NULL ),
						_T("Reading the result buffer failed") );
					{
						FFTPlan* fftPlanX	= NULL;
						lockRAII* planLockX	= NULL;
						OPENCL_V( fftRepo.getPlan( fftPlan->planX, fftPlanX, planLockX ), _T( "fftRepo.getPlan failed" ) );

						size_t rows = fftPlanX->length[0];
						size_t cols = fftPlanX->batchsize;
						BUG_CHECK (rows * cols <= temp.size())
						size_t print_cols = std::min<size_t> (4, cols);
						size_t print_rows = std::min<size_t> (4, rows);
						//std::cout << std::endl << "Intermediate buffer:" << std::endl;
						//for (size_t jrow = 0; jrow < print_rows; ++jrow) {
						//	for (size_t icol = 0; icol < print_cols; ++icol) {
						//		size_t index = jrow *cols + icol;
						//		std::complex<float> data = temp[index];
						//		std::cout << data;
						//	}
						//	std::cout << std::endl;
						//}
					}
#endif

					//another column FFT output, OUTOFPLACE
					if (fftPlan->placeness == CLFFT_INPLACE)
					{
						OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1, &colOutEvents,
							outEvents, &localIntBuffer, clInputBuffers, localIntBuffer ),
							_T("clfftEnqueueTransform large1D second column failed"));

#if defined(DEBUGGING)
						//  For debugging interleave data only,
						//  read the input buffer back into memory.
						OPENCL_V( clEnqueueReadBuffer( *commQueues, clInputBuffers[0], CL_TRUE, 0, buffSizeBytes_complex, &temp[ 0 ], 1,
							outEvents, NULL ),
							_T("Reading the result buffer failed") );
#endif
					}
					else
					{
#if defined(DEBUGGING)
					// debug purpose, interleave input <-> interleave output
					OPENCL_V( clEnqueueReadBuffer( *commQueues, localIntBuffer, CL_TRUE, 0, buffSizeBytes_complex, &temp[ 0 ], 1,
						&colOutEvents, NULL ),
						_T("Reading the result buffer failed") );
#endif
						OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1, &colOutEvents,
							outEvents, &localIntBuffer, clOutputBuffers, localIntBuffer ),
							_T("clfftEnqueueTransform large1D second column failed"));

#if defined(DEBUGGING)
						//  For debugging interleave data only, read back the output buffer
						//
						OPENCL_V( clEnqueueReadBuffer( *commQueues, clOutputBuffers[0], CL_TRUE, 0, buffSizeBytes_complex, &temp[ 0 ], 1,
							outEvents, NULL ),
							_T("Reading the result buffer failed") );
#endif
					}
				}
				else
				{
					// second pass for huge 1D
					// column with twiddle first, OUTOFPLACE, + transpose
					OPENCL_V( clfftEnqueueTransform( fftPlan->planX, dir, numQueuesAndEvents, commQueues, numWaitEvents,
						waitEvents, &colOutEvents, &localIntBuffer, clOutputBuffers, localIntBuffer),
						_T("clfftEnqueueTransform Huge1D col pass failed"));
#if defined(DEBUGGING)
					// debug purpose, interleave input <-> interleave output
					OPENCL_V( clEnqueueReadBuffer( *commQueues, clOutputBuffers[0], CL_TRUE, 0, buffSizeBytes_complex, &temp[ 0 ], 1,
						&colOutEvents, NULL ),
						_T("Reading the result buffer failed") );
#endif

					OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1, &colOutEvents,
						outEvents, clOutputBuffers, clOutputBuffers, localIntBuffer ),
						_T("clfftEnqueueTransform large1D second column failed"));

				}

				clReleaseEvent(colOutEvents);

				if( fftRepo.pStatTimer )
				{
					fftRepo.pStatTimer->AddSample( plHandle, fftPlan, NULL, 0, NULL, std::vector< size_t >( ) );
				}

				return	CLFFT_SUCCESS;
			}
			break;
		}
		case CLFFT_2D:
		{
			// if transpose kernel, we will fall below
			if (fftPlan->transflag && !(fftPlan->planTX)) break;

			cl_event rowOutEvents = NULL;

#if defined(DEBUGGING)
			size_t buffersize = fftPlan->length[0] * fftPlan->length[1] * fftPlan->batchsize;
			if (fftPlan->length.size() > 2) buffersize *= fftPlan->length[2];
			//size_t buffSizeBytes=sizeof( std::complex< float > )*buffersize;
			//std::vector< std::complex< float > > output2( buffersize );
			size_t buffSizeBytes=sizeof( float) * buffersize;
			std::vector<float> output2(buffersize*2);
#endif
#if defined(DEBUGGING)
			OPENCL_V( clEnqueueReadBuffer( *commQueues, clInputBuffers[0], CL_TRUE, 0, buffSizeBytes, &output2[ 0 ], 0,
				NULL, NULL ),
				_T("Reading the result buffer failed") );

			if (fftPlan->placeness == CLFFT_OUTOFPLACE)
			{
				OPENCL_V( clEnqueueReadBuffer( *commQueues, clOutputBuffers[0], CL_TRUE, 0, buffSizeBytes, &output2[ 0 ], 0,
					NULL, NULL ),
					_T("Reading the result buffer failed") );
			}
#endif
			if (fftPlan->transflag)
			{//first time set up transpose kernel for 2D
				//First row
				OPENCL_V( clfftEnqueueTransform( fftPlan->planX, dir, numQueuesAndEvents, commQueues, numWaitEvents,
					waitEvents, &rowOutEvents, clInputBuffers, clOutputBuffers, NULL ),
					_T("clfftEnqueueTransform for row failed"));

				cl_mem *mybuffers;

				if (fftPlan->placeness==CLFFT_INPLACE)
					mybuffers = clInputBuffers;
				else
					mybuffers = clOutputBuffers;

#if defined(DEBUGGING)
				OPENCL_V( clEnqueueReadBuffer( *commQueues, mybuffers[0], CL_TRUE, 0, buffSizeBytes*2, &output2[ 0 ], 0,
					NULL, NULL ),
					_T("Reading the result buffer failed") );
#endif

				cl_event transXOutEvents = NULL;
				cl_event colOutEvents = NULL;
				bool xyflag = (fftPlan->length[0] == fftPlan->length[1]) ? false : true;

				if (xyflag)
				{
					//First transpose
					OPENCL_V( clfftEnqueueTransform( fftPlan->planTX, dir, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
						&transXOutEvents, mybuffers, &localIntBuffer, NULL ),
						_T("clfftEnqueueTransform for first transpose failed"));
					clReleaseEvent(rowOutEvents);

#if defined(DEBUGGING)
					OPENCL_V( clEnqueueReadBuffer( *commQueues, localIntBuffer, CL_TRUE, 0, buffSizeBytes*2, &output2[0], 0,
						NULL, NULL ),
						_T("Reading the result buffer failed") );
#endif

					if (fftPlan->transposed == CLFFT_NOTRANSPOSE)
					{
						//Second Row transform
						OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1, &transXOutEvents,
							&colOutEvents, &localIntBuffer, NULL, NULL ),
							_T("clfftEnqueueTransform for second row failed"));
						clReleaseEvent(transXOutEvents);

#if defined(DEBUGGING)
						OPENCL_V( clEnqueueReadBuffer( *commQueues, localIntBuffer, CL_TRUE, 0, buffSizeBytes*2, &output2[0], 0,
							NULL, NULL ),
							_T("Reading the result buffer failed") );
#endif

						//Second transpose
						OPENCL_V( clfftEnqueueTransform( fftPlan->planTY, dir, numQueuesAndEvents, commQueues, 1, &colOutEvents,
							outEvents, &localIntBuffer, mybuffers, NULL ),
							_T("clfftEnqueueTransform for second transpose failed"));
						clReleaseEvent(colOutEvents);

#if defined(DEBUGGING)
						OPENCL_V( clEnqueueReadBuffer( *commQueues, mybuffers[0], CL_TRUE, 0, buffSizeBytes*2, &output2[0], 0,
							NULL, NULL ),
							_T("Reading the result buffer failed") );
#endif
					}
					else
					{
						//Second Row transform
						OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1, &transXOutEvents,
							outEvents, &localIntBuffer, mybuffers, NULL ),
							_T("clfftEnqueueTransform for second row failed"));
						clReleaseEvent(transXOutEvents);
					}
				}
				else
				{
					// First Transpose
					OPENCL_V( clfftEnqueueTransform( fftPlan->planTX, dir, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
						&transXOutEvents, mybuffers, NULL, NULL ),
						_T("clfftEnqueueTransform for first transpose failed"));
					clReleaseEvent(rowOutEvents);

					if (fftPlan->transposed == CLFFT_NOTRANSPOSE)
					{
						//Second Row transform
						OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1, &transXOutEvents,
							&colOutEvents, mybuffers, NULL, NULL ),
							_T("clfftEnqueueTransform for Second Row failed"));
						clReleaseEvent(transXOutEvents);

						//Second transpose
						OPENCL_V( clfftEnqueueTransform( fftPlan->planTY, dir, numQueuesAndEvents, commQueues, 1, &colOutEvents,
							outEvents, mybuffers, NULL, NULL ),
							_T("clfftEnqueueTransform for second transpose failed"));
						clReleaseEvent(colOutEvents);
					}
					else
					{
						//Second Row transform
						OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1, &transXOutEvents,
							outEvents, mybuffers, NULL, NULL ),
							_T("clfftEnqueueTransform for second row failed"));
						clReleaseEvent(transXOutEvents);
					}

				}

				if( fftRepo.pStatTimer )
				{
					fftRepo.pStatTimer->AddSample( plHandle, fftPlan, NULL, 0, NULL, std::vector< size_t >( ) );
				}

				return CLFFT_SUCCESS;
			}

			if ( (fftPlan->large2D || fftPlan->length.size()>2) &&
				(fftPlan->inputLayout != CLFFT_REAL) && (fftPlan->outputLayout != CLFFT_REAL))
			{
				if (fftPlan->placeness==CLFFT_INPLACE)
				{
					//deal with row first
					OPENCL_V( clfftEnqueueTransform( fftPlan->planX, dir, numQueuesAndEvents, commQueues, numWaitEvents,
						waitEvents, &rowOutEvents, clInputBuffers, NULL, localIntBuffer ),
						_T("clfftEnqueueTransform for row failed"));

					//deal with column
					OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
						outEvents, clInputBuffers, NULL, localIntBuffer ),
						_T("clfftEnqueueTransform for column failed"));
				}
				else
				{
					//deal with row first
					OPENCL_V( clfftEnqueueTransform( fftPlan->planX, dir, numQueuesAndEvents, commQueues, numWaitEvents,
						waitEvents, &rowOutEvents, clInputBuffers, clOutputBuffers, localIntBuffer ),
						_T("clfftEnqueueTransform for row failed"));

					//deal with column
					OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
						outEvents, clOutputBuffers, NULL, localIntBuffer ),
						_T("clfftEnqueueTransform for column failed"));

				}
			}
			else
			{
				if(fftPlan->inputLayout == CLFFT_REAL)
				{
					if (fftPlan->placeness==CLFFT_INPLACE)
					{
						// deal with row
						OPENCL_V( clfftEnqueueTransform( fftPlan->planX, CLFFT_FORWARD, numQueuesAndEvents, commQueues, numWaitEvents,
							waitEvents, &rowOutEvents, clInputBuffers, NULL, localIntBuffer ),
							_T("clfftEnqueueTransform for row failed"));

						// deal with column
						OPENCL_V( clfftEnqueueTransform( fftPlan->planY, CLFFT_FORWARD, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
							outEvents, clInputBuffers, NULL, localIntBuffer ),
							_T("clfftEnqueueTransform for column failed"));
					}
					else
					{
						// deal with row
						OPENCL_V( clfftEnqueueTransform( fftPlan->planX, CLFFT_FORWARD, numQueuesAndEvents, commQueues, numWaitEvents,
							waitEvents, &rowOutEvents, clInputBuffers, clOutputBuffers, localIntBuffer ),
							_T("clfftEnqueueTransform for row failed"));

						// deal with column
						OPENCL_V( clfftEnqueueTransform( fftPlan->planY, CLFFT_FORWARD, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
							outEvents, clOutputBuffers, NULL, localIntBuffer ),
							_T("clfftEnqueueTransform for column failed"));
					}
				}
				else if(fftPlan->outputLayout == CLFFT_REAL)
				{
					cl_mem *out_local, *int_local, *out_y;

					if(fftPlan->length.size() > 2)
					{
						out_local = clOutputBuffers;
						int_local = NULL;
						out_y = clInputBuffers;
					}
					else
					{
						out_local = (fftPlan->placeness==CLFFT_INPLACE) ? clInputBuffers : clOutputBuffers;
						int_local = fftPlan->tmpBufSizeC2R ? &(fftPlan->intBufferC2R) : &localIntBuffer;
						out_y = int_local;
					}


					// deal with column
					OPENCL_V( clfftEnqueueTransform( fftPlan->planY, CLFFT_BACKWARD, numQueuesAndEvents, commQueues, numWaitEvents,
						waitEvents, &rowOutEvents, clInputBuffers, int_local, localIntBuffer ),
						_T("clfftEnqueueTransform for row failed"));

					// deal with row
					OPENCL_V( clfftEnqueueTransform( fftPlan->planX, CLFFT_BACKWARD, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
						outEvents, out_y, out_local, localIntBuffer ),
						_T("clfftEnqueueTransform for column failed"));

				}
				else
				{
					//deal with row first
					OPENCL_V( clfftEnqueueTransform( fftPlan->planX, dir, numQueuesAndEvents, commQueues, numWaitEvents,
						waitEvents, &rowOutEvents, clInputBuffers, &localIntBuffer, localIntBuffer ),
						_T("clfftEnqueueTransform for row failed"));


					if (fftPlan->placeness==CLFFT_INPLACE)
					{
						//deal with column
						OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
							outEvents, &localIntBuffer, clInputBuffers, localIntBuffer ),
							_T("clfftEnqueueTransform for column failed"));
					}
					else
					{
						//deal with column
						OPENCL_V( clfftEnqueueTransform( fftPlan->planY, dir, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
							outEvents, &localIntBuffer, clOutputBuffers, localIntBuffer ),
							_T("clfftEnqueueTransform for column failed"));

		#if defined(DEBUGGING)
						OPENCL_V( clEnqueueReadBuffer( *commQueues, clOutputBuffers[0], CL_TRUE, 0, buffSizeBytes, &output2[ 0 ], 1,
							outEvents, NULL ),
							_T("Reading the result buffer failed") );
		#endif
					}
				}
			}

			clReleaseEvent(rowOutEvents);

			if( fftRepo.pStatTimer )
			{
				fftRepo.pStatTimer->AddSample( plHandle, fftPlan, NULL, 0, NULL, std::vector< size_t >( ) );
			}

			return	CLFFT_SUCCESS;
		}
		case CLFFT_3D:
		{
			cl_event rowOutEvents = NULL;

#if defined(DEBUGGING)
			size_t buffersize = fftPlan->length[0] * fftPlan->length[1] *fftPlan->length[2] *fftPlan->batchsize;
			size_t buffSizeBytes=sizeof( std::complex< float > )*buffersize;
			std::vector< std::complex< float > > output3( buffersize );
#endif
			if(fftPlan->inputLayout == CLFFT_REAL)
			{
				cl_mem *tmp_local, *out_local;

				tmp_local = (fftPlan->placeness==CLFFT_INPLACE) ? NULL : clOutputBuffers;
				out_local = (fftPlan->placeness==CLFFT_INPLACE) ? clInputBuffers : clOutputBuffers;

				//deal with 2D row first
				OPENCL_V( clfftEnqueueTransform( fftPlan->planX, CLFFT_FORWARD, numQueuesAndEvents, commQueues, numWaitEvents,
					waitEvents, &rowOutEvents, clInputBuffers, tmp_local, localIntBuffer ),
					_T("clfftEnqueueTransform for 3D-XY row failed"));

				//deal with 1D Z column
				OPENCL_V( clfftEnqueueTransform( fftPlan->planZ, CLFFT_FORWARD, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
					outEvents, out_local, NULL, localIntBuffer ),
					_T("clfftEnqueueTransform for 3D-Z column failed"));

			}
			else if(fftPlan->outputLayout == CLFFT_REAL)
			{
				cl_mem *out_local;
				out_local = (fftPlan->placeness==CLFFT_INPLACE) ? clInputBuffers : clOutputBuffers;

				cl_mem *int_local;
				int_local = fftPlan->tmpBufSizeC2R ? &(fftPlan->intBufferC2R) : &localIntBuffer;

				//deal with 1D Z column first
				OPENCL_V( clfftEnqueueTransform( fftPlan->planZ, CLFFT_BACKWARD, numQueuesAndEvents, commQueues, numWaitEvents,
					waitEvents, &rowOutEvents, clInputBuffers, int_local, localIntBuffer ),
					_T("clfftEnqueueTransform for 3D-Z column failed"));

				//deal with 2D row
				OPENCL_V( clfftEnqueueTransform( fftPlan->planX, CLFFT_BACKWARD, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
					outEvents, int_local, out_local, localIntBuffer ),
					_T("clfftEnqueueTransform for 3D-XY row failed"));
			}
			else
			{
				if (fftPlan->placeness==CLFFT_INPLACE)
				{
					//deal with 2D row first
					OPENCL_V( clfftEnqueueTransform( fftPlan->planX, dir, numQueuesAndEvents, commQueues, numWaitEvents,
						waitEvents, &rowOutEvents, clInputBuffers, NULL, localIntBuffer ),
						_T("clfftEnqueueTransform for 3D-XY row failed"));

					//deal with 1D Z column
					OPENCL_V( clfftEnqueueTransform( fftPlan->planZ, dir, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
						outEvents, clInputBuffers, NULL, localIntBuffer ),
						_T("clfftEnqueueTransform for 3D-Z column failed"));
				}
				else
				{
	#if defined(DEBUGGING)
					OPENCL_V( clEnqueueReadBuffer( *commQueues, clOutputBuffers[0], CL_TRUE, 0, buffSizeBytes, &output3[ 0 ], 0,
						NULL, NULL ),
						_T("Reading the result buffer failed") );
	#endif

					//deal with 2D row first
					OPENCL_V( clfftEnqueueTransform( fftPlan->planX, dir, numQueuesAndEvents, commQueues, numWaitEvents,
						waitEvents, &rowOutEvents, clInputBuffers, clOutputBuffers, localIntBuffer ),
						_T("clfftEnqueueTransform for 3D-XY row failed"));

	#if defined(DEBUGGING)
					OPENCL_V( clEnqueueReadBuffer( *commQueues, clOutputBuffers[0], CL_TRUE, 0, buffSizeBytes, &output3[ 0 ], 0,
						NULL, NULL ),
						_T("Reading the result buffer failed") );
	#endif

					//deal with 1D Z column
					OPENCL_V( clfftEnqueueTransform( fftPlan->planZ, dir, numQueuesAndEvents, commQueues, 1, &rowOutEvents,
						outEvents, clOutputBuffers, NULL, localIntBuffer ),
						_T("clfftEnqueueTransform for 3D-Z column failed"));
	#if defined(DEBUGGING)
					OPENCL_V( clEnqueueReadBuffer( *commQueues, clOutputBuffers[0], CL_TRUE, 0, buffSizeBytes, &output3[ 0 ], 1,
						outEvents, NULL ),
						_T("Reading the result buffer failed") );
	#endif
				}
			}

			clReleaseEvent(rowOutEvents);

			if( fftRepo.pStatTimer )
			{
				fftRepo.pStatTimer->AddSample( plHandle, fftPlan, NULL, 0, NULL, std::vector< size_t >( ) );
			}

			return	CLFFT_SUCCESS;
		}
	}

	// 1d with normal length will fall into the below category
	// add: 2d transpose kernel will fall into here too.
	vector< cl_mem >	inputBuff;
	vector< cl_mem >	outputBuff;
	inputBuff.reserve( 2 );
	outputBuff.reserve( 2 );

	//	Decode the relevant properties from the plan paramter to figure out how many input/output buffers we have
	switch( fftPlan->inputLayout )
	{
		case CLFFT_COMPLEX_INTERLEAVED:
		{
			switch( fftPlan->outputLayout )
			{
				case CLFFT_COMPLEX_INTERLEAVED:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				case CLFFT_COMPLEX_PLANAR:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						//	Invalid to be an inplace transform, and go from 1 to 2 buffers
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 1 ] );
					}

					break;
				}
				case CLFFT_HERMITIAN_INTERLEAVED:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				case CLFFT_HERMITIAN_PLANAR:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 1 ] );
					}

					break;
				}
				case CLFFT_REAL:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				default:
				{
					//	Don't recognize output layout
					return CLFFT_INVALID_ARG_VALUE;
				}
			}

			break;
		}
		case CLFFT_COMPLEX_PLANAR:
		{
			switch( fftPlan->outputLayout )
			{
				case CLFFT_COMPLEX_INTERLEAVED:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						inputBuff.push_back( clInputBuffers[ 1 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				case CLFFT_COMPLEX_PLANAR:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						inputBuff.push_back( clInputBuffers[ 1 ] );
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						inputBuff.push_back( clInputBuffers[ 1 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 1 ] );
					}

					break;
				}
				case CLFFT_HERMITIAN_INTERLEAVED:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						inputBuff.push_back( clInputBuffers[ 1 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				case CLFFT_HERMITIAN_PLANAR:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						inputBuff.push_back( clInputBuffers[ 1 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 1 ] );
					}

					break;
				}
				case CLFFT_REAL:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						inputBuff.push_back( clInputBuffers[ 1 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				default:
				{
					//	Don't recognize output layout
					return CLFFT_INVALID_ARG_VALUE;
				}
			}

			break;
		}
		case CLFFT_HERMITIAN_INTERLEAVED:
		{
			switch( fftPlan->outputLayout )
			{
				case CLFFT_COMPLEX_INTERLEAVED:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				case CLFFT_COMPLEX_PLANAR:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 1 ] );
					}

					break;
				}
				case CLFFT_HERMITIAN_INTERLEAVED:
				{
					return CLFFT_INVALID_ARG_VALUE;
				}
				case CLFFT_HERMITIAN_PLANAR:
				{
					return CLFFT_INVALID_ARG_VALUE;
				}
				case CLFFT_REAL:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				default:
				{
					//	Don't recognize output layout
					return CLFFT_INVALID_ARG_VALUE;
				}
			}

			break;
		}
		case CLFFT_HERMITIAN_PLANAR:
		{
			switch( fftPlan->outputLayout )
			{
				case CLFFT_COMPLEX_INTERLEAVED:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						inputBuff.push_back( clInputBuffers[ 1 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				case CLFFT_COMPLEX_PLANAR:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						inputBuff.push_back( clInputBuffers[ 1 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 1 ] );
					}

					break;
				}
				case CLFFT_HERMITIAN_INTERLEAVED:
				{
					return CLFFT_INVALID_ARG_VALUE;
				}
				case CLFFT_HERMITIAN_PLANAR:
				{
					return CLFFT_INVALID_ARG_VALUE;
				}
				case CLFFT_REAL:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						inputBuff.push_back( clInputBuffers[ 1 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				default:
				{
					//	Don't recognize output layout
					return CLFFT_INVALID_ARG_VALUE;
				}
			}

			break;
		}
		case CLFFT_REAL:
		{
			switch( fftPlan->outputLayout )
			{
				case CLFFT_COMPLEX_INTERLEAVED:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				case CLFFT_COMPLEX_PLANAR:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 1 ] );
					}

					break;
				}
				case CLFFT_HERMITIAN_INTERLEAVED:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 0 ] );
					}

					break;
				}
				case CLFFT_HERMITIAN_PLANAR:
				{
					if( fftPlan->placeness == CLFFT_INPLACE )
					{
						return CLFFT_INVALID_ARG_VALUE;
					}
					else
					{
						inputBuff.push_back( clInputBuffers[ 0 ] );

						outputBuff.push_back( clOutputBuffers[ 0 ] );
						outputBuff.push_back( clOutputBuffers[ 1 ] );
					}

					break;
				}
				default:
				{
					//	Don't recognize output layout
					return CLFFT_INVALID_ARG_VALUE;
				}
			}

			break;
		}
		default:
		{
			//	Don't recognize output layout
			return CLFFT_INVALID_ARG_VALUE;
		}
	}

	//	TODO:  In the case of length == 1, FFT is a trivial NOP, but we still need to apply the forward and backwards tranforms
	//	TODO:  Are map lookups expensive to call here?  We can cache a pointer to the cl_program/cl_kernel in the plan

	FFTKernelGenKeyParams fftParams;
	//	Translate the user plan into the structure that we use to map plans to clPrograms
	OPENCL_V( fftPlan->GetKernelGenKey( fftParams ), _T("GetKernelGenKey() failed!") );

	cl_program	prog;
	cl_kernel	kern;
	OPENCL_V( fftRepo.getclProgram( fftPlan->gen, fftParams, prog, fftPlan->context ), _T( "fftRepo.getclProgram failed" ) );
	OPENCL_V( fftRepo.getclKernel( prog, dir, kern ), _T( "fftRepo.getclKernels failed" ) );



	cl_uint uarg = 0;
	if (!fftPlan->transflag && !(fftPlan->gen == Copy))
	{
		//	::clSetKernelArg() is not thread safe, according to the openCL spec for the same cl_kernel object
		//	TODO:  Need to verify that two different plans (which would get through our lock above) with exactly the same
		//	parameters would NOT share the same cl_kernel objects

		/* constant buffer */
		OPENCL_V( clSetKernelArg( kern, uarg++, sizeof( cl_mem ), (void*)&fftPlan->const_buffer ), _T( "clSetKernelArg failed" ) );
	}

	//	Input buffer(s)
	//	Input may be 1 buffer  (CLFFT_COMPLEX_INTERLEAVED)
	//	          or 2 buffers (CLFFT_COMPLEX_PLANAR)

	for (size_t i = 0; i < inputBuff.size(); ++i)
	{
		OPENCL_V( clSetKernelArg( kern, uarg++, sizeof( cl_mem ), (void*)&inputBuff[i] ), _T( "clSetKernelArg failed" ) );
	}
	//	Output buffer(s)
	//	Output may be 0 buffers (CLFFT_INPLACE)
	//	           or 1 buffer  (CLFFT_COMPLEX_INTERLEAVED)
	//	           or 2 buffers (CLFFT_COMPLEX_PLANAR)
	for (size_t o = 0; o < outputBuff.size(); ++o)
	{
		OPENCL_V( clSetKernelArg( kern, uarg++, sizeof( cl_mem ), (void*)&outputBuff[o] ), _T( "clSetKernelArg failed" ) );
	}

	vector< size_t > gWorkSize;
	vector< size_t > lWorkSize;
	clfftStatus result = fftPlan->GetWorkSizes (gWorkSize, lWorkSize);

	// TODO:  if GetWorkSizes returns CLFFT_INVALID_GLOBAL_WORK_SIZE, that means
	// that this multidimensional input data array is too large to be transformed
	// with a single call to clEnqueueNDRangeKernel.  For now, we will just return
	// the error code back up the call stack.
	// The *correct* course of action would be to split the work into mutliple
	// calls to clEnqueueNDRangeKernel.
	if (CLFFT_INVALID_GLOBAL_WORK_SIZE == result)
	{
		OPENCL_V( result, _T("Work size too large for clEnqueNDRangeKernel()"));
	}
	else
	{
		OPENCL_V( result, _T("FFTPlan::GetWorkSizes failed"));
	}
	BUG_CHECK (gWorkSize.size() == lWorkSize.size());

	size_t *lwSize = NULL;
	if(fftPlan->gen != Copy) lwSize = &lWorkSize[ 0 ];

	status = clEnqueueNDRangeKernel( *commQueues, kern, static_cast< cl_uint >( gWorkSize.size( ) ),
		NULL, &gWorkSize[ 0 ], lwSize, numWaitEvents, waitEvents, outEvents );
	OPENCL_V( status, _T( "clEnqueueNDRangeKernel failed" ) );

	if( fftRepo.pStatTimer )
	{
		fftRepo.pStatTimer->AddSample( plHandle, fftPlan, kern, numQueuesAndEvents, outEvents, gWorkSize );
	}

	return	CLFFT_SUCCESS;
}
