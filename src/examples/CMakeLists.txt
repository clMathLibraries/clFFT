# ########################################################################
# Copyright 2013 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################

INCLUDE_DIRECTORIES(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${OpenCL_INCLUDE_DIRS}"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/include"
    )

LINK_DIRECTORIES("${PROJECT_BINARY_DIR}/package/lib${SUFFIX_LIB}")

FILE(GLOB FILES "*.c")

FOREACH(FILE ${FILES})

    if( MSVC )
    	if( MSVC_VERSION LESS 1800 )
    		# Use C++ with Microsoft compiler
    		SET_SOURCE_FILES_PROPERTIES( ${FILE} PROPERTIES LANGUAGE CXX)
    	endif ()
    endif( )

    GET_FILENAME_COMPONENT(EXAMPLE ${FILE} NAME_WE)
    GET_FILENAME_COMPONENT(FULL_DIR_NAME ${FILE} PATH)
    GET_FILENAME_COMPONENT(DIR_NAME ${FULL_DIR_NAME} NAME)
    SET(EXAMPLE_NAME example_${DIR_NAME}_${EXAMPLE})
    ADD_EXECUTABLE(${EXAMPLE_NAME} ${FILE})

    TARGET_LINK_LIBRARIES(${EXAMPLE_NAME} clFFT ${OpenCL_LIBRARIES} ${CMAKE_DL_LIBS})

    SET_TARGET_PROPERTIES(${EXAMPLE_NAME}
        PROPERTIES
        OUTPUT_NAME ${EXAMPLE}
        RUNTIME_OUTPUT_DIRECTORY ${DIR_NAME})

    INSTALL(FILES ${FILE} DESTINATION "share/clFFT/examples")
    INSTALL(TARGETS ${EXAMPLE_NAME}
        RUNTIME DESTINATION "share/clFFT/examples")
ENDFOREACH()
