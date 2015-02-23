
option(USE_SYSTEM_GTEST "Use system installed gtest when set to ON, or build gtest locally when set to OFF" OFF)

if(USE_SYSTEM_GTEST)
  if( (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_LESS 2.8) )
    message( STATUS "Cmake version 2.8 or greater needed to use GTest" )
  else()
    # This will define GTEST_FOUND
    find_package( GTest )
  endif()
else()
#  find_package(Threads REQUIRED)
  include(ExternalProject)

  ExternalProject_Add(
    gtest-external
    URL http://googletest.googlecode.com/files/gtest-1.7.0.zip
    URL_MD5 2d6ec8ccdf5c46b05ba54a9fd1d130d7
    INSTALL_COMMAND "")
  
  ExternalProject_Get_Property(gtest-external binary_dir)
  add_library(gtest IMPORTED STATIC GLOBAL)
  set_target_properties(gtest PROPERTIES
      IMPORTED_LOCATION ${binary_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
      # IMPORTED_LINK_INTERFACE_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
      )
  add_dependencies(gtest gtest-external)

  add_library(gtest_main IMPORTED STATIC GLOBAL)
  set_target_properties(gtest_main PROPERTIES
      IMPORTED_LOCATION ${binary_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}
      # IMPORTED_LINK_INTERFACE_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
      )
  add_dependencies(gtest_main gtest-external)

  ExternalProject_Get_Property(gtest-external source_dir)
  set(GTEST_INCLUDE_DIRS ${source_dir}/include)
  set(GTEST_LIBRARIES gtest gtest_main)
  set(GTEST_FOUND ON)
endif()

# Hack to get googletest v1.6 to work with vs2012
if( MSVC11 )
  add_definitions( "/D_VARIADIC_MAX=10" )
endif( )
