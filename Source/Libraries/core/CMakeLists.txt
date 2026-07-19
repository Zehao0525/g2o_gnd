# GND / non-Gaussian kernels and edges (depends on tutorial_slam2d types)
add_library(GND_Core ${LIB_TYPE}
  GNDEdges/edge_none_gaussian_unary.h
  GNDEdges/edge_none_gaussian_binary.h
  GNDEdges/edge_platform_loc_prior_gnd.h
  GNDEdges/edge_platform_loc_prior_gnd.cpp
  GNDEdges/edge_platform_pose_prior_gnd.h
  GNDEdges/edge_platform_pose_prior_gnd.cpp
  GNDEdges/cauchy_edge_se2_pointxy.h
  GNDEdges/cauchy_edge_se2_pointxy.cpp
  GNDEdges/gnd_kernel.h
  GNDEdges/gnd_kernel.cpp
  GNDEdges/gnd_cauchy_kernel.h
  GNDEdges/gnd_cauchy_kernel.cpp
)

target_include_directories(GND_Core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/GNDEdges
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(GND_Core PUBLIC
  G2O_Graph
  Eigen3::Eigen
  g2o::core
  g2o::stuff
  g2o::solver_eigen
)
