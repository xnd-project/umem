cmake_minimum_required (VERSION 3.9)
project (UMEM)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake_modules/")

#
# UMEM project version.
#
set(UMEM_VERSION_MAJOR 0)
set(UMEM_VERSION_MINOR 1)
set(UMEM_VERSION_MICRO 0)

#
# Set default build type.
#
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release"
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

#
# Set location where to checkout thirdparty sources
#
set(UMEM_THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)

#
# CMake configuration options
#
option(ENABLE_FILE "Enable File memory context" ON)
option(ENABLE_MMAP "Enable MMap memory context [NOT IMPLEMENTED]" OFF)
option(ENABLE_CUDA "Enable Cuda memory context (CUDA RT)" ON)
option(ENABLE_CUDA_HOST "Enable Cuda Host memory context" OFF)
option(ENABLE_CUDA_MANAGED "Enable Cuda Managed memory context" ON)
option(ENABLE_RMM "Enable Rapidsai cudf RMM memory context" ON)
option(BUILD_SHARED "Build shared umem library" ON)
option(BUILD_STATIC "Build static umem library [NOT IMPLEMENTED]" OFF)
option(BUILD_TESTS "Build unittests" ON)
if (CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
  option(BUILD_DOXYGEN_DOC "Build Doxygen Documentation" ON)
else()
  option(BUILD_DOXYGEN_DOC "Build Doxygen Documentation" OFF)
endif()

#
# There are two options to enable CUDA dependencies in cmake:
# enable_language(CUDA) or find_package(CUDA) (deprecated)
# -DWITH_CUDA_LANGUAGE=ON means the former
#
option(WITH_CUDA_LANGUAGE "Enable CUDA language" ON)

if (NOT TRUE) # DEBUG make
  set(CMAKE_RULE_MESSAGES OFF)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()


#
# Compiler options, be pedantic.
#
if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-long-long -pedantic")
endif()

set(UMEM_INCLUDE_DIRS)
set(UMEM_LIBRARY_DIRS)
set(UMEM_LIBRARIES)
set(UMEM_ENABLED_CONTEXTS)

#
# Check CUDA availability.
#
set(NEED_CUDA FALSE)
if (ENABLE_CUDA OR ENABLE_RMM OR ENABLE_CUDA_HOST OR ENABLE_CUDA_MANAGED)
  set(NEED_CUDA TRUE)
endif()

if (NEED_CUDA)
  if (WITH_CUDA_LANGUAGE)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
      message(STATUS "  CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
      message(STATUS "  CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES=${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
      set(UMEM_INCLUDE_DIRS ${UMEM_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
      set(UMEM_LIBRARY_DIRS ${UMEM_LIBRARY_DIRS} ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
      find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
      message(STATUS "  CUDART_LIBRARY=${CUDART_LIBRARY}")
      if (ENABLE_CUDA OR ENABLE_CUDA_HOST OR ENABLE_CUDA_MANAGED)
        set(UMEM_LIBRARIES ${UMEM_LIBRARIES} cudart)
      endif()
      if (ENABLE_RMM)
        set(UMEM_LIBRARIES ${UMEM_LIBRARIES} cuda)
      endif()
      set(UMEM_LIBRARIES ${UMEM_LIBRARIES} pthread rt)
    else()
      message(STATUS "No CUDA support, disabling memory context types: Cuda, CudaHost, CudaManaged, RMM.")
      set(ENABLE_CUDA OFF)
      set(ENABLE_RMM OFF)
      set(ENABLE_CUDA_HOST OFF)
      set(ENABLE_CUDA_MANAGED OFF)
    endif()
  else()
    find_package(CUDA 9.0)    
  endif()
endif()

#
# UMEM Host context type support, always enabled.
#
set(UMEM_INCLUDE_DIRS ${UMEM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
set(UMEM_SOURCES umem.c umem_host.c umem_utils.c umem_connect.c)
set(UMEM_LIBRARIES umem ${UMEM_LIBRARIES})
set(UMEM_ENABLED_CONTEXTS ${UMEM_ENABLED_CONTEXTS} Host)

#
# UMEM File context type support
#
if (ENABLE_FILE)
  set(UMEM_ENABLED_CONTEXTS ${UMEM_ENABLED_CONTEXTS} File)
  add_definitions("-DHAVE_FILE_CONTEXT")
  set(UMEM_SOURCES ${UMEM_SOURCES} umem_file.c)
endif(ENABLE_FILE)

#
# UMEM MMap context type support [NOT IMPLEMENTED]
#
if (ENABLE_MMAP)
  set(UMEM_ENABLED_CONTEXTS ${UMEM_ENABLED_CONTEXTS} MMap)
  add_definitions("-DHAVE_MMAP_CONTEXT")
  set(UMEM_SOURCES ${UMEM_SOURCES} umem_mmap.c)
endif(ENABLE_MMAP)

#
# UMEM Cuda context type support
#
if (ENABLE_CUDA OR ENABLE_RMM OR ENABLE_CUDA_HOST OR ENABLE_CUDA_MANAGED)
  set(UMEM_SOURCES ${UMEM_SOURCES} umem_cuda_utils.c)
endif()

if (ENABLE_CUDA)
  set(UMEM_ENABLED_CONTEXTS ${UMEM_ENABLED_CONTEXTS} Cuda)
  add_definitions("-DHAVE_CUDA_CONTEXT")
  set(UMEM_SOURCES ${UMEM_SOURCES} umem_cuda.c)
endif(ENABLE_CUDA)

#
# UMEM CudaHost context type support
#
if (ENABLE_CUDA_HOST)
  set(UMEM_ENABLED_CONTEXTS ${UMEM_ENABLED_CONTEXTS} CudaHost)
  add_definitions("-DHAVE_CUDA_HOST_CONTEXT")
  set(UMEM_SOURCES ${UMEM_SOURCES} umem_cuda_host.c)
endif(ENABLE_CUDA_HOST)

#
# UMEM CudaHost context type support
#
if (ENABLE_CUDA_MANAGED)
  set(UMEM_ENABLED_CONTEXTS ${UMEM_ENABLED_CONTEXTS} CudaManaged)
  add_definitions("-DHAVE_CUDA_MANAGED_CONTEXT")
  set(UMEM_SOURCES ${UMEM_SOURCES} umem_cuda_managed.c)
endif(ENABLE_CUDA_MANAGED)

#
# UMEM RMM context type support
#
if (ENABLE_RMM)
  set(UMEM_ENABLED_CONTEXTS ${UMEM_ENABLED_CONTEXTS} RMM)
  if (NOT EXISTS "${UMEM_THIRDPARTY_DIR}/cudf")
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} clone --recurse-submodules https://github.com/rapidsai/cudf.git
      WORKING_DIRECTORY ${UMEM_THIRDPARTY_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git clone command failed with ${GIT_SUBMOD_RESULT}.")
    endif()
  endif()
  include_directories(
    ${UMEM_THIRDPARTY_DIR}/cudf/cpp/src/rmm/
    ${UMEM_THIRDPARTY_DIR}/cudf/cpp/thirdparty/cnmem/include/)
  
  set(UMEM_SOURCES ${UMEM_SOURCES} umem_rmm.cc
    ${UMEM_THIRDPARTY_DIR}/cudf/cpp/src/rmm/memory.cpp
    ${UMEM_THIRDPARTY_DIR}/cudf/cpp/src/rmm/memory_manager.cpp
    ${UMEM_THIRDPARTY_DIR}/cudf/cpp/thirdparty/cnmem/src/cnmem.cpp
    )
  add_definitions("-DHAVE_RMM_CONTEXT")
endif(ENABLE_RMM)

#
# Generate umem_config.h file
#
include(CheckIncludeFiles)
check_include_files(malloc.h HAVE_MALLOC_H)  # just for testing, not used
configure_file (
  ${PROJECT_SOURCE_DIR}/umem_config.h.in
  ${PROJECT_BINARY_DIR}/umem_config.h
  )
set(UMEM_INCLUDE_DIRS ${UMEM_INCLUDE_DIRS} ${PROJECT_BINARY_DIR})

#
# Set build environment
#
include_directories(${UMEM_INCLUDE_DIRS})
link_directories(${UMEM_LIBRARY_DIRS})

#
# Define libraries to be built.
#
if (BUILD_SHARED)
  add_library(umem SHARED ${UMEM_SOURCES})
endif()

if (BUILD_STATIC)
  add_library(umem_static STATIC ${UMEM_SOURCES})
endif()

#
# Build Doxygen documentation
#
if (BUILD_DOXYGEN_DOC)
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
    #set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/../doc/Doxyfile.in)
    #set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    #configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")
    add_custom_target( doc_doxygen ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../doc/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM )
  else()
    message("Doxygen need to be installed to generate the doxygen documentation")
  endif()
endif()


#
# Enable unit tests.
#
if (BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

message(STATUS "\n\n  Enabled context types: ${UMEM_ENABLED_CONTEXTS}\n")

#
# Install umem
#
install(FILES umem.h umem.hpp umem_utils.h "${PROJECT_BINARY_DIR}/umem_config.h" DESTINATION include)
if (BUILD_SHARED)
  install(TARGETS umem DESTINATION lib)
endif()
if (BUILD_STATIC)
  install(TARGETS umem_static DESTINATION lib)
endif()
