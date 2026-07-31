add_library(GlennMultirobot ${LIB_TYPE}
  events_glenn.h
  file_simulator.h
  file_simulator.cpp
  file_slam_system.h
  file_slam_system.cpp
)

find_package(nlohmann_json REQUIRED)

target_include_directories(GlennMultirobot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(GlennMultirobot PUBLIC
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

target_link_libraries(GlennMultirobot PRIVATE G2O_Graph GND_Core)

target_include_directories(GlennMultirobot PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
