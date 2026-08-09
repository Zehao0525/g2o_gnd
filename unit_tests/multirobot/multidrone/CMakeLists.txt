add_executable(multi_drone_experiment_new
  experiment.cpp
)

target_link_libraries(multi_drone_experiment_new PRIVATE
  G2O_Graph
  SlamSystemBase
  Incremental_Visualizer
  Multidrone_Simulator
)
