clFFT
=====
[![Build Status](https://travis-ci.org/clMathLibraries/clFFT.png)](https://travis-ci.org/clMathLibraries/clFFT)

clFFT is a software library containing FFT functions written
in OpenCL. In addition to GPU devices, the libraries also support
running on CPU devices to facilitate debugging and heterogeneous
programming.

Pre-built binaries are available [here][binary_release].

## What's New

The [commit](https://github.com/clMathLibraries/clFFT/commit/2a533358c6a559d1a996e1c3077dd05bc8845191) in the
develop branch has added important updates to the library. They 
include:
-   Significant uplift of 1D complex transform performance
-   1D large size limit relaxation for complex transforms
-   2D/3D size limit relaxation on real and complex transforms
-   Binary caching feature
-   Several minor fixes and improvements


## Introduction to clFFT

The FFT is an implementation of the Discrete Fourier Transform (DFT)
that makes use of symmetries in the FFT definition to reduce the
mathematical intensity required from O(N2) to O(N log2( N )) when the
sequence length N is the product of small prime factors. Currently,
there is no standard API for FFT routines. Hardware vendors usually
provide a set of high-performance FFTs optimized for their systems: no
two vendors employ the same interfaces for their FFT routines. clFFT
provides a set of FFT routines that are optimized for AMD graphics
processors, but also are functional across CPU and other compute
devices.

The clFFT library is an open source OpenCL library implementation of
discrete Fast Fourier Transforms. It:

-   Provides a fast and accurate platform for calculating discrete FFTs.

-   Works on CPU or GPU backends.

-   Supports in-place or out-of-place transforms.

-   Supports 1D, 2D, and 3D transforms with a batch size that can be
    greater than 1.

-   Supports planar (real and complex components in separate arrays) and
    interleaved (real and complex components as a pair contiguous in
    memory) formats.

-   Supports dimension lengths that can be any mix of powers of 2, 3,
    and 5.

-   Supports single and double precision floating point formats.

## clFFT library user documentation

[Library and API documentation][] for developers is available online as
a GitHub Pages website

### Google Groups

Two mailing lists have been created for the clMath projects:

-   [clmath@googlegroups.com][] - group whose focus is to answer
    questions on using the library or reporting issues

-   [clmath-developers@googlegroups.com][] - group whose focus is for
    developers interested in contributing to the library code itself

## clFFT Wiki

The [project wiki][clmath@googlegroups.com] contains helpful
documentation, including a [build
primer][clmath-developers@googlegroups.com]

## Contributing code

Please refer to and read the [Contributing][] document for guidelines on
how to contribute code to this open source project. The code in the
/master branch is considered to be stable, and all pull-requests should
be made against the /develop branch.

## License

The source for clFFT is licensed under the [Apache License, Version
2.0][]

## Example

The simple example below shows how to use clFFT to compute an simple 1D
forward transform
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

-   Windows® 7/8

-   Visual Studio 2010 SP1, 2012

-   Latest CMake

-   An OpenCL SDK, such as APP SDK 2.9

### Library for Linux

-   GCC 4.6 and onwards

-   Latest CMake

-   An OpenCL SDK, such as APP SDK 2.9

### Library for Mac OSX

-   Recommended to generate Unix makefiles with cmake

### Test infrastructure

-   Googletest v1.6

-   Latest FFTW

-   Latest Boost

### Performance infrastructure

-   Python

  [Library and API documentation]: http://clmathlibraries.github.io/clFFT/
  [clmath@googlegroups.com]: https://github.com/clMathLibraries/clFFT/wiki
  [clmath-developers@googlegroups.com]: https://github.com/clMathLibraries/clFFT/wiki/Build
  [Contributing]: CONTRIBUTING.md
  [Apache License, Version 2.0]: http://www.apache.org/licenses/LICENSE-2.0
  [binary_release]: https://github.com/clMathLibraries/clFFT/releases
