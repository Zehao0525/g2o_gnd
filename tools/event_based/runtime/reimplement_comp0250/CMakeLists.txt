add_library(ReimplementComp0250 ${LIB_TYPE}
  events_comp.h
  sensor_data.h
  incremental_simulator.h
  incremental_simulator.cpp
  system_model.h
  system_model.cpp
  platform_controller.h
  platform_controller.cpp
  slam_system.h
  slam_system.cpp
)

find_package(nlohmann_json REQUIRED)

target_include_directories(ReimplementComp0250 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(ReimplementComp0250 PUBLIC
  SlamSystemBase
  Eigen3::Eigen
  g2o::core
  g2o::stuff
  g2o::solver_eigen
  -Wl,--whole-archive
  g2o::types_slam3d
  -Wl,--no-whole-archive
  nlohmann_json::nlohmann_json
)

target_link_libraries(ReimplementComp0250 PRIVATE G2O_Graph GGD_Core)

target_include_directories(ReimplementComp0250 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Backward-compatible alias for existing Example / viz link lines.
add_library(Incremental_Simulator ALIAS ReimplementComp0250)
