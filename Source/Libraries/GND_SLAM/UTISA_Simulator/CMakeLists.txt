add_library(UTISA_Simulator ${LIB_TYPE}
  utisa_agent_manager.h
  utisa_agent_manager.cpp
  utisa_simulation.h
  utisa_simulation.cpp
  utisa_slam_system.h
  utisa_slam_system.cpp
  utisa_messages.hpp
  utisa_stamp_map.hpp
  utisa_events.h
  utisa_simulator_api.h
)

find_package(nlohmann_json REQUIRED)

target_link_libraries(UTISA_Simulator PRIVATE
  G2O_Graph
  Incremental_Simulator
)

target_include_directories(UTISA_Simulator PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(UTISA_Simulator PUBLIC
  Eigen3::Eigen
  g2o::core
  g2o::stuff
  g2o::solver_eigen
  nlohmann_json::nlohmann_json
)

target_include_directories(UTISA_Simulator PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
