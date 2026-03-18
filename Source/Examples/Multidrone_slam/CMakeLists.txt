add_executable(multi_drone_experiment
  experiment.cpp
)


target_link_libraries(multi_drone_experiment PRIVATE
  G2O_Graph
  Incremental_Simulator
  Incremental_Visualizer
  Multidrone_Simulator
)
