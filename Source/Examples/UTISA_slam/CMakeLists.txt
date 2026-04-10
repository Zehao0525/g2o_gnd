add_executable(utisa_slam_experiment
  experiment.cpp
)

target_include_directories(utisa_slam_experiment PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(utisa_slam_experiment PRIVATE
  UTISA_Simulator
  G2O_Graph
  Incremental_Simulator
  Incremental_Visualizer
)

add_executable(batch_utisa_slam
  batch_experiment.cpp
)

target_include_directories(batch_utisa_slam PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(batch_utisa_slam PRIVATE
  UTISA_Simulator
  G2O_Graph
  Incremental_Simulator
  Incremental_Visualizer
)
