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


#include <gtest/gtest.h>
#include<math.h>

#include "test_constants.h"
#include "fftw_transform.h"
#include "cl_transform.h"
#include "typedefs.h"
#include "accuracy_test_common.h"
#include <stdexcept>
#include <vector>

/*****************************************************/
clfftResultLocation cl_placeness( placeness::placeness_t placeness )
{
	if( placeness == placeness::in_place )
		return CLFFT_INPLACE;
	else if( placeness == placeness::out_of_place )
		return CLFFT_OUTOFPLACE;
	else
		throw std::runtime_error( "invalid placeness" );
}

/*****************************************************/
clfftLayout cl_layout( layout::buffer_layout_t layout_in )
{
	if( layout_in == layout::real )
		return CLFFT_REAL;
	else if( layout_in == layout::hermitian_planar )
		return CLFFT_HERMITIAN_PLANAR;
	else if( layout_in == layout::complex_planar )
		return CLFFT_COMPLEX_PLANAR;
	else if( layout_in == layout::hermitian_interleaved )
		return CLFFT_HERMITIAN_INTERLEAVED;
	else if( layout_in == layout::complex_interleaved )
		return CLFFT_COMPLEX_INTERLEAVED;
	else
		throw std::runtime_error( "invalid layout_in" );
}
