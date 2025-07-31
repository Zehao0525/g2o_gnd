add_library(Incremental_Visualizer ${LIB_TYPE}
  #visualizer.h 
  #visualizer.cpp 
  view.h
  view.cpp
  view_manager.h
  view_manager.cpp
  simulator_view.h
  simulator_view.cpp
  slam_system_view.h
  slam_system_view.cpp
  file_simulator_view.h
  file_simulator_view.cpp
  file_slam_system_view.h
  file_slam_system_view.cpp

  #viz_shape.h
)

find_package(OpenGL REQUIRED)
find_package(Pangolin REQUIRED)
find_package(nlohmann_json REQUIRED)

target_link_libraries(Incremental_Visualizer PRIVATE 
  G2O_Graph
  Incremental_Simulator
)
target_include_directories(Incremental_Visualizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


target_link_libraries(Incremental_Visualizer PUBLIC
  Eigen3::Eigen
  g2o::core
  g2o::stuff
  g2o::solver_eigen
  ${OPENGL_LIBRARIES}
  pango_display
  pango_opengl
  nlohmann_json::nlohmann_json
)

target_include_directories(Incremental_Visualizer PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

