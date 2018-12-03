find_program( MEMORYCHECK_COMMAND valgrind )
set( MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full" )
# Usage: ctest -D ExperimentalMemCheck -E test_cuda

if (MSVC)
  SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
endif()

include_directories(${UMEM_INCLUDE_DIRS})
set(LIBS ${UMEM_LIBRARIES})
add_definitions("-DTMPDIR=\"/tmp\"") # TODO: make it portable and uid independent

message(STATUS "UMEM_LIBRARIES=${UMEM_LIBRARIES}")

file(GLOB test_files "test_*_*.c")
foreach(filename ${test_files})
  get_filename_component(basename ${filename} NAME_WE)
  if ("${basename}" MATCHES ".*cuda" AND NOT CUDA_FOUND)
    message(STATUS "Skipping ${basename}")
    continue()
  endif()
  message(STATUS "Adding ${basename} to tests")
  add_executable(${basename} ${filename})
  target_link_libraries(${basename} ${LIBS})
  add_test(${basename} ${CMAKE_CURRENT_BINARY_DIR}/${basename})
endforeach()
