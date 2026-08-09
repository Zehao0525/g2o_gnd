# Header-only runtime base (Event, SlamSystemBase, OrderedEventQueue).
add_library(SlamSystemBase INTERFACE)

target_include_directories(SlamSystemBase INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

find_package(nlohmann_json REQUIRED)

target_link_libraries(SlamSystemBase INTERFACE
  Eigen3::Eigen
  g2o::core
  g2o::stuff
  g2o::solver_eigen
  nlohmann_json::nlohmann_json
  G2O_Graph
  GGD_Core
)
