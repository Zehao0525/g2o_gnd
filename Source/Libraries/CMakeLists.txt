if(NOT DEFINED BUILD_SHARED_LIBRARIES)
  set(BUILD_SHARED_LIBRARIES NO)
endif()

if(BUILD_SHARED_LIBRARIES)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
endif()

# types first: tutorial SE2 graph types (CMake target: G2O_Graph)
add_subdirectory(types)

# core: GND kernels / edges (CMake target: GND_Core)
add_subdirectory(core)

# runtime: simulators, slam bases, domain frontends
add_subdirectory(runtime)

# viz: Pangolin views (CMake target: Incremental_Visualizer)
# After Multidrone + UTISA: Incremental_Visualizer links both
add_subdirectory(viz)
