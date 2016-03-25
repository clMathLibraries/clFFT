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

#include "test_constants.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include "../client/openCL.misc.h"

#if defined( _WIN32 ) && defined( _DEBUG )
#include <Windows.h>
#endif

void handle_exception( const std::exception& except )
{
	std::string error_message(except.what());

	std::cout << "--- Exception caught ---" << std::endl;

	if( error_message.find("problem too large for device") != std::string::npos ||
		error_message.find("CLFFT_INVALID_BUFFER_SIZE" ) != std::string::npos ||
		error_message.find("CLFFT_MEM_OBJECT_ALLOCATION_FAILURE" ) != std::string::npos ||
		error_message.find("CLFFT_OUT_OF_HOST_MEMORY" ) != std::string::npos ||
		error_message.find("CLFFT_OUT_OF_RESOURCES" ) != std::string::npos )
	{
		std::cout << "Data set is too large for this device -- skipping test" << std::endl;
		//TODO put in (this problem size[data + stride]/max problem size/gpu or cpu) specifics
	}
	else if( error_message.find("system memory allocation failure") != std::string::npos )
	{
		std::cout << "Framework was denied enough system memory to support the data set"
			<< " -- skipping test" << std::endl;
	}
	else if( error_message.find("CLFFT_DEVICE_NO_DOUBLE") != std::string::npos )
	{
		std::cout << "Device in context does not support double precision"
			<< " -- skipping test" << std::endl;
	}
	else if( error_message.find("dereference null pointer") != std::string::npos )
	{
		std::cout << error_message << std::endl;
		FAIL();
	}
	else if( error_message.find("in-place transform, unmatched in/out layouts")
					!= std::string::npos )
	{
		std::cout << "Invalid arguments: for an in-place transform, "
			<< "in/output layouts must be the same" << std::endl;
		FAIL();
	}
	else if( error_message.find("device list is empty at transform") != std::string::npos )
	{
		std::cout << "A clfft transform is requested, but the device list is empty" << std::endl;
		FAIL();
	}
	else
	{
		std::cout << "Unrecognized exception: " << std::endl;
		std::cout << error_message << std::endl;
		/*
#if defined( _WIN32 ) && defined( _DEBUG )
		::DebugBreak( );
#endif
		*/
		FAIL();
	}
}

/*****************************************************/
size_t max_mem_available_on_cl_device(size_t device_index) {

	static size_t g_device_max_mem_size  = 0;

	// this is not thread-safe using globals, it is just quick fix for now, todo proper fix
	if (g_device_max_mem_size == 0)
	{
		std::vector< cl_device_id >	device_id;
		cl_context tempContext = NULL;
		device_id = initializeCL(
			g_device_type,
			(cl_int)device_index,
			g_platform_id,
			tempContext,
			false
			);

		cl_ulong device_max_to_allocate = 0;
		if (device_id.size() == 0 || device_index > device_id.size())
		{
		}
		else
		{
			OPENCL_V_THROW(::clGetDeviceInfo(device_id[device_index], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &device_max_to_allocate, NULL),
				"Getting CL_DEVICE_MAX_MEM_ALLOC_SIZE device info ( ::clGetDeviceInfo() )");
		}

		cl_command_queue tempQueue = NULL;
		cl_event tempEvent = NULL;
		::cleanupCL(&tempContext, &tempQueue, 0, NULL, 0, NULL, &tempEvent);

		g_device_max_mem_size = static_cast<size_t>(device_max_to_allocate);
	}

	return g_device_max_mem_size;
}
