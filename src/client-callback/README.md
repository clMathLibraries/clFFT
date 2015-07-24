clFFT - Callback Client
=======================


clFFT Callback client is a sample application demonstrating the use of 
callback feature of clFFT. 

Callback feature provides ability to do custom processing when reading 
input data or when writing output data. There are 2 types of callback,
Pre-callback and Post-callback. Pre-callback invokes user callback 
function to do custom preprocessing of input data before FFT is executed.
Post-callback invokes user callback function to do custom post-processing 
of output data after FFT is executed. The intent is to avoid additional 
kernels and kernel launches to carry out the pre/post processing. Instead 
the pre/post processing logic can be included in an inline opencl function 
(one each for pre and post) and passed as a string to library which would 
then be incorporated into the generated FFT kernel.

The block below shows the help message given by the callback client listing 
all the command line options.

```c
C:\clFFT\src\build\staging\Debug>clFFT-callback.exe -h
clFFT client command line options:
  -h [ --help ]               produces this help message
  -g [ --gpu ]                Force selection of OpenCL GPU devices only
  -c [ --cpu ]                Force selection of OpenCL CPU devices only
  -a [ --all ]                Force selection of all OpenCL devices (default)
  -o [ --outPlace ]           Out of place FFT transform (default: in place)
  --double                    Double precision transform (default: single)
  --inv                       Backward transform (default: forward)
  -d [ --dumpKernels ]        FFT engine will dump generated OpenCL FFT kernels
                              to disk (default: dump off)
  --noprecall                 Disable Precallback (default: precallback on)
  -x [ --lenX ] arg (=1024)   Specify the length of the 1st dimension of a test
                              array
  -y [ --lenY ] arg (=1)      Specify the length of the 2nd dimension of a test
                              array
  -z [ --lenZ ] arg (=1)      Specify the length of the 3rd dimension of a test
                              array
  --isX arg (=1)              Specify the input stride of the 1st dimension of
                              a test array
  --isY arg (=0)              Specify the input stride of the 2nd dimension of
                              a test array
  --isZ arg (=0)              Specify the input stride of the 3rd dimension of
                              a test array
  --iD arg (=0)               input distance between subsequent sets of data
                              when batch size > 1
  --osX arg (=1)              Specify the output stride of the 1st dimension of
                              a test array
  --osY arg (=0)              Specify the output stride of the 2nd dimension of
                              a test array
  --osZ arg (=0)              Specify the output stride of the 3rd dimension of
                              a test array
  --oD arg (=0)               output distance between subsequent sets of data
                              when batch size > 1
  -b [ --batchSize ] arg (=1) If this value is greater than one, arrays will be
                              used
  -p [ --profile ] arg (=1)   Time and report the kernel speed of the FFT
                              (default: profiling off)
  --inLayout arg (=1)         Layout of input data:
                              1) interleaved
                              2) planar
                              3) hermitian interleaved
                              4) hermitian planar
                              5) real
  --outLayout arg (=1)        Layout of input data:
                              1) interleaved
                              2) planar
                              3) hermitian interleaved
                              4) hermitian planar
                              5) real

```
"--noprecall" option can be used to disable Pre-callback (default: precallback on)

## What's New

Callback client in the develop branch demonstrates use of pre-callback 
for Single Precision Complex-Complex 1D transforms for lengths upto 4096. Output data
is verified against fftw library.

## Example

Some examples are shown below.

1D Complex-Complex Interleaved transform with pre-callback for length 1024
```c
C:\clFFT\src\build\staging\Debug>clFFT-callback.exe -x 1024 --inLayout 1 --outLayout 1


                Internal Client Test *****PASS*****
```				

1D Complex-Complex Planar transform with pre-callback for length 1024
```c
C:\clFFT\src\build\staging\Debug>clFFT-callback.exe -x 1024 --inLayout 2 --outLayout 2


                Internal Client Test *****PASS*****
```

1D Complex-Complex Interleaved transform with pre-callback for length 1024 and batch size of 2
```c
C:\clFFT\src\build\staging\Debug>clFFT-callback.exe -x 1024 --inLayout 1 --outLayout 1 -b 2


                Internal Client Test *****PASS*****
```

1D Complex-Complex Interleaved transform without pre-callback for length 1024 
```c
C:\clFFT\src\build\staging\Debug>clFFT-callback.exe -x 1024 --inLayout 1 --outLayout 1 --noprecall


                Internal Client Test *****PASS*****
```