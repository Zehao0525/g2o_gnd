add_executable(utisa_slam_experiment_new
  experiment.cpp
)

target_include_directories(utisa_slam_experiment_new PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(utisa_slam_experiment_new PRIVATE
  UTISA_Simulator
  G2O_Graph
  SlamSystemBase
  Incremental_Visualizer
)

add_executable(utisa_slam_unit_experiment_new
  unit_experiment.cpp
)

target_include_directories(utisa_slam_unit_experiment_new PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(utisa_slam_unit_experiment_new PRIVATE
  UTISA_Simulator
  G2O_Graph
  SlamSystemBase
  Incremental_Visualizer
)
