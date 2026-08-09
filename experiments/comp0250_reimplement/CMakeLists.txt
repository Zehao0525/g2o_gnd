add_executable(incsim_test
  incsim_test.cpp
)

target_link_libraries(incsim_test PRIVATE
  G2O_Graph
  ReimplementComp0250
  Incremental_Visualizer
)
