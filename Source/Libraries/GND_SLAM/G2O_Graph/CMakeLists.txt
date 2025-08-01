
add_library(G2O_Graph ${LIB_TYPE}
  edge_se2_pointxy.cpp  se2.h                vertex_point_xy.h
  edge_se2.cpp    edge_se2_pointxy.h    vertex_se2.cpp
  edge_se2.h      vertex_point_xy.cpp  vertex_se2.h
  parameter_se2_offset.h parameter_se2_offset.cpp
  types_tutorial_slam2d.h types_tutorial_slam2d.cpp
  g2o_tutorial_slam2d_api.h
  edge_range_bearing.h
  edge_range_bearing.cpp
  edge_platform_loc_prior.h
  edge_platform_loc_prior.cpp
  edge_platform_pose_prior.h
  edge_platform_pose_prior.cpp
  edge_se2_wt.cpp
  edge_se2_wt.h

  GNDEdges/edge_none_gaussian_unary.h
  GNDEdges/edge_platform_loc_prior_gnd.h
  GNDEdges/edge_platform_loc_prior_gnd.cpp
  GNDEdges/edge_none_gaussian_binary.h
  GNDEdges/cauchy_edge_se2_pointxy.h
  GNDEdges/cauchy_edge_se2_pointxy.cpp
  GNDEdges/gnd_kernel.h
  GNDEdges/gnd_kernel.cpp
)

#target_include_directories(G2O_Graph PUBLIC ${CMAKE_CURRENT_DIRECTORY})
target_include_directories(G2O_Graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

get_target_property(g2o_core_include_dirs g2o::core INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "🔍 g2o::core include dirs: ${g2o_core_include_dirs}")



target_link_libraries(G2O_Graph PUBLIC
  Eigen3::Eigen
  g2o::core
  g2o::stuff
  g2o::solver_eigen

)

target_include_directories(G2O_Graph PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/GNDEdges
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
