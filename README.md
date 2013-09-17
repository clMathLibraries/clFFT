clFFT
=====

clMath is a software library containing FFT and BLAS functions written in OpenCL. In addition to GPU devices, the libraries also support running on CPU devices to facilitate debugging and multicore programming.

<a href="http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-math-libraries/">APPML 1.10</a> is the most current generally available version of the library, and pre-built binaries are available for download on both Linux and Windows platforms.

[![Build Status](https://travis-ci.org/kknox/clFFT.png)](https://travis-ci.org/kknox/clFFT)
## Introduction to clFFT

The FFT is an implementation of the Discrete Fourier Transform (DFT) that makes use of symmetries in the FFT definition to reduce the mathematical intensity required from O(N<sup>2</sup>) to O(N log<sub>2</sub>( N )) when the sequence length N is the product of small prime factors. Currently, there is no standard API for FFT routines. Hardware vendors usually provide a set of high-performance FFTs optimized for their systems: no two vendors employ the same interfaces for their FFT routines. clFFT provides a set of FFT routines that are optimized for AMD graphics processors, but also are functional across CPU and other compute devices.

The clFFT library is an open source OpenCL library implementation of discrete Fast Fourier Transforms. It:

* Provides a fast and accurate platform for calculating discrete FFTs. 
* Works on CPU or GPU backends. 
* Supports in-place or out-of-place transforms. 
* Supports 1D, 2D, and 3D transforms with a batch size that can be greater than 1. 
* Supports planar (real and complex components in separate arrays) and interleaved (real and complex components as a pair contiguous in memory) formats. 
* Supports dimension lengths that can be any mix of powers of 2, 3, and 5. 
* Supports single and double precision floating point formats.

## clFFT library user documentation
[Library and API documentation]( http://clmathlibraries.github.io/clFFT/ ) for developers is available online as a GitHub Pages website

## clFFT Wiki
The [project wiki](https://github.com/clMathLibraries/clFFT/wiki) contains helpful documentation, including a [build primer](https://github.com/clMathLibraries/clFFT/wiki/Build)

## Contributing code
Please refer to and read the [Contributing](CONTRIBUTING.md) document for guidelines on how to contribute code to this open source project

## License
The source for clFFT is licensed under the [Apache License, Version 2.0]( http://www.apache.org/licenses/LICENSE-2.0 )

## Example
The simple example below shows how to use clFFT to compute an simple 1D forward transform

```c
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
	
	/* FFT library realted declarations */
	clfftPlanHandle planHandle;
	clfftDim dim = CLFFT_1D;
	size_t clLengths[1] = {N};
                
    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

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
```

## Build dependencies
### Library for Windows
*  Windows® 7/8
*  Visual Studio 2010 SP1, 2012
*  Latest CMake
*  An OpenCL SDK, such as APP SDK 2.8

### Library for Linux
*  GCC 4.6 and onwards
*  Latest CMake
*  An OpenCL SDK, such as APP SDK 2.8

### Test infrastructure
* Latest Googletest
* Latest FFTW 
* Latest Boost

### Performance infrastructure
* Python