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

#include <stdio.h>
#include <stdlib.h>

/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>

int main( void )
{
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufX;
    float *X;
    cl_event event = NULL;
    int ret = 0;
    size_t N = 16;
    char platform_name[128];
    char device_name[128];

    /* FFT library realted declarations */
    clfftPlanHandle planHandle;
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {N};

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );

    size_t ret_param_size = 0;
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME,
            sizeof(platform_name), platform_name,
            &ret_param_size);
    printf("Platform found: %s\n", platform_name);

    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL );

    err = clGetDeviceInfo(device, CL_DEVICE_NAME,
            sizeof(device_name), device_name,
            &ret_param_size);
    printf("Device found on the above platform: %s\n", device_name);

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    queue = clCreateCommandQueue( ctx, device, 0, &err );

    /* Setup clFFT. */
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);

    /* Allocate host & initialize data. */
    /* Only allocation shown for simplicity. */
    X = (float *)malloc(N * 2 * sizeof(*X));

    /* print input array */
    printf("\nPerforming fft on an one dimensional array of size N = %lu\n", (unsigned long)N);
    int print_iter = 0;
    while(print_iter<N) {
        float x = (float)print_iter;
        float y = (float)print_iter*3;
        X[2*print_iter  ] = x;
        X[2*print_iter+1] = y;
        printf("(%f, %f) ", x, y);
        print_iter++;
    }
    printf("\n\nfft result: \n");

    /* Prepare OpenCL memory objects and place data inside them. */
    bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, N * 2 * sizeof(*X), NULL, &err );

    err = clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,
            N * 2 * sizeof( *X ), X, 0, NULL, NULL );

    /* Create a default plan for a complex FFT. */
    err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);

    /* Set plan parameters. */
    err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
    err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

    /* Bake the plan. */
    err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);

    /* Execute the plan. */
    err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, NULL, NULL);

    /* Wait for calculations to be finished. */
    err = clFinish(queue);

    /* Fetch results of calculations. */
    err = clEnqueueReadBuffer( queue, bufX, CL_TRUE, 0, N * 2 * sizeof( *X ), X, 0, NULL, NULL );

    /* print output array */
    print_iter = 0;
    while(print_iter<N) {
        printf("(%f, %f) ", X[2*print_iter], X[2*print_iter+1]);
        print_iter++;
    }
    printf("\n");

    /* Release OpenCL memory objects. */
    clReleaseMemObject( bufX );

    free(X);

    /* Release the plan. */
    err = clfftDestroyPlan( &planHandle );

    /* Release clFFT library. */
    clfftTeardown( );

    /* Release OpenCL working objects. */
    clReleaseCommandQueue( queue );
    clReleaseContext( ctx );

    return ret;
}
