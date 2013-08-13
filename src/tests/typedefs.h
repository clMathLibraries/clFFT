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


#pragma once
#if !defined( CLFFT_TYPEDEFS_H )
#define CLFFT_TYPEDEFS_H

#include "test_constants.h"
#include "fftw_transform.h"
#include "cl_transform.h"

typedef clfft<float, cl_float> clfft_single;
typedef clfft<double, cl_double> clfft_double;
typedef buffer<float> buffer_single;
typedef buffer<double> buffer_double;
typedef fftw<float, fftwf_complex> fftw_single;
typedef fftw<double, fftw_complex> fftw_double;

#endif