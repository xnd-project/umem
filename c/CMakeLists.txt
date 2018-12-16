cmake_minimum_required (VERSION 3.9)
project (UMEM)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake_modules/")

option(ENABLE_FILE "Enable File memory context" ON)
option(ENABLE_MMAP "Enable MMap memory context [NOT IMPLEMENTED]" OFF)
option(ENABLE_CUDA "Enable Cuda memory context using CUDA RT" ON)

option(BUILD_DOXYGEN_DOC "Build Doxygen Documentation" ON)

#
# Maximize compiler warnings
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

#enable_testing()
include(CTest)

# The version number.
set (UMEM_VERSION_MAJOR 0)
set (UMEM_VERSION_MINOR 1)
set (UMEM_VERSION_MICRO 0)

set(UMEM_SOURCES umem.c umem_host.c umem_utils.c umem_connect.c)
set(UMEM_LIBRARIES umem)
set(UMEM_INCLUDE_DIRS ${CMAKE_SOURCE_DIR})

if (ENABLE_FILE)
  add_definitions("-DHAVE_FILE_CONTEXT")
  set(UMEM_SOURCES ${UMEM_SOURCES} umem_file.c)
endif(ENABLE_FILE)

if (ENABLE_MMAP)
  add_definitions("-DHAVE_MMAP_CONTEXT")
  set(UMEM_SOURCES ${UMEM_SOURCES} umem_mmap.c)
endif(ENABLE_MMAP)


if (ENABLE_CUDA)
  find_package(CUDA 9.0)
  if(NOT CUDA_FOUND)
  endif()
endif(ENABLE_CUDA)

#enable_language(CUDA)
#include(CheckLanguage)
#check_language(CUDA)
# While FindCUDA is deprecated, we need it for linking with non-.cu files

if(CUDA_FOUND)
  message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "  CUDA_INCLUDE_DIRS=${CUDA_INCLUDE_DIRS}")
  message(STATUS "  CUDA_LIBRARIES=${CUDA_LIBRARIES} CUDA_CUDA_LIBRARY=${CUDA_CUDA_LIBRARY}")
  set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
  cuda_select_nvcc_arch_flags(ARCH_FLAGS)

  add_definitions("-DHAVE_CUDA_CONTEXT")
  add_definitions("-DHAVE_CUDA")
  
  set(UMEM_INCLUDE_DIRS ${UMEM_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
  set(UMEM_SOURCES ${UMEM_SOURCES} umem_cuda.c)
  set(UMEM_LIBRARIES ${UMEM_LIBRARIES} ${CUDA_CUDA_LIBRARY})
  
  include_directories(${UMEM_INCLUDE_DIRS})
  cuda_add_library(umem SHARED ${UMEM_SOURCES})
  cuda_add_library(umem_static STATIC ${UMEM_SOURCES})
else()
  add_library(umem SHARED ${UMEM_SOURCES})
  add_library(umem_static STATIC ${UMEM_SOURCES})
endif()

#
# Generate umem_config.h file
#
include(CheckIncludeFiles)
check_include_files(malloc.h HAVE_MALLOC_H)

configure_file (
  "${PROJECT_SOURCE_DIR}/umem_config.h.in"
  "${PROJECT_BINARY_DIR}/umem_config.h"
  )


include_directories("${PROJECT_BINARY_DIR}")

add_subdirectory(tests)

#add_executable(Test test.c)
#target_link_libraries(Test PUBLIC ${UMEM_LIBRARIES})
#add_dependencies(Test umem)

install(TARGETS umem umem_static DESTINATION lib)
install(FILES umem.h umem_utils.h "${PROJECT_BINARY_DIR}/umem_config.h" DESTINATION include)

if (CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
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
  endif(BUILD_DOXYGEN_DOC)

endif()
