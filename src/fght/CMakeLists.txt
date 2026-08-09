# GGD / non-Gaussian kernels and edges (depends on tutorial_slam2d types)
add_library(GGD_Core ${LIB_TYPE}
  GGDEdges/edge_none_gaussian_unary.h
  GGDEdges/edge_none_gaussian_binary.h
  GGDEdges/edge_platform_loc_prior_ggd.h
  GGDEdges/edge_platform_loc_prior_ggd.cpp
  GGDEdges/cauchy_edge_se2_pointxy.h
  GGDEdges/cauchy_edge_se2_pointxy.cpp
  GGDEdges/ggd_kernel.h
  GGDEdges/ggd_kernel.cpp
  GGDEdges/ggd_cauchy_kernel.h
  GGDEdges/ggd_cauchy_kernel.cpp
)

target_include_directories(GGD_Core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/GGDEdges
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(GGD_Core PUBLIC
  G2O_Graph
  Eigen3::Eigen
  g2o::core
  g2o::stuff
  g2o::solver_eigen
)
