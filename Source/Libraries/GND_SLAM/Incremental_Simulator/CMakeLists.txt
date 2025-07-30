#add_subdirectory(visualizer)

add_library(Incremental_Simulator ${LIB_TYPE}
  events.h 
  incremental_simulator.h
  incremental_simulator.cpp 
  system_model.h
  system_model.cpp 
  ordered_event_queue.hpp 
  platform_controller.h
  platform_controller.cpp 
  slam_system.h
  slam_system.cpp 
  file_simulator.h
  file_simulator.cpp
)

find_package(nlohmann_json REQUIRED)

target_link_libraries(Incremental_Simulator PRIVATE G2O_Graph)
target_include_directories(Incremental_Simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


target_link_libraries(Incremental_Simulator PUBLIC
  Eigen3::Eigen
  g2o::core
  g2o::stuff
  g2o::solver_eigen
  nlohmann_json::nlohmann_json
)

target_include_directories(Incremental_Simulator PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

