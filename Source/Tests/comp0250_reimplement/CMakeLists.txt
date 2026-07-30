add_executable(incsim_test_new
  incsim_test.cpp
)

target_link_libraries(incsim_test_new PRIVATE
  G2O_Graph
  ReimplementComp0250
  Incremental_Visualizer
)
