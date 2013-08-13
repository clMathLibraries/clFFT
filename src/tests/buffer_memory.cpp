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


#include <stdint.h>

/*****************************************************/
/*****************************************************/
uint32_t float_as_hex( float a ) {
	return *(uint32_t*)&a;
}

/*****************************************************/
/*****************************************************/
uint64_t float_as_hex( double a ) {
	return *(uint64_t*)&a;
}

/*****************************************************/
/*****************************************************/
uint32_t nan_as_hex( float a ) {
	a;
	return ~0x0;
}

/*****************************************************/
/*****************************************************/
uint64_t nan_as_hex( double a ) {
	a;
	return ~0x0ull;
}