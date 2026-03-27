add_executable(multi_drone_experiment
  experiment.cpp
)

add_executable(batch_multi_drone_experiment
  batch_experiment.cpp
)

target_link_libraries(multi_drone_experiment PRIVATE
  G2O_Graph
  Incremental_Simulator
  Incremental_Visualizer
  Multidrone_Simulator
)

target_link_libraries(batch_multi_drone_experiment PRIVATE
  G2O_Graph
  Incremental_Simulator
  Incremental_Visualizer
  Multidrone_Simulator
)
