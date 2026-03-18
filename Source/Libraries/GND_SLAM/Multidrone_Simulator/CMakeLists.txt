## Follow pattern used by Incremental_Simulator
add_library(Multidrone_Simulator ${LIB_TYPE}
  agent_manager.h
  agent_manager.cpp
  data_based_simulation.h
  data_based_simulation.cpp
  drone_slam_system.h
  drone_slam_system.cpp
  messages.hpp
  stamp_map.hpp
)

find_package(nlohmann_json REQUIRED)

target_link_libraries(Multidrone_Simulator PRIVATE
  G2O_Graph
  Incremental_Simulator
)

target_include_directories(Multidrone_Simulator PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(Multidrone_Simulator PUBLIC
  Eigen3::Eigen
  g2o::core
  g2o::stuff
  g2o::solver_eigen
  -Wl,--whole-archive
  g2o::types_slam3d
  -Wl,--no-whole-archive
  nlohmann_json::nlohmann_json
)

target_include_directories(Multidrone_Simulator PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

