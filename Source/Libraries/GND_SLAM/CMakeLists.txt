if(NOT DEFINED BUILD_SHARED_LIBRARIES)
  set(BUILD_SHARED_LIBRARIES NO)
endif()

if(BUILD_SHARED_LIBRARIES)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
endif()

cmake_minimum_required(VERSION 3.10)
project(GND_SLAM)

set(CMAKE_CXX_STANDARD 17)

# Add subdirectories
add_subdirectory(Oneshot_Simulator)
add_subdirectory(G2O_Graph)

