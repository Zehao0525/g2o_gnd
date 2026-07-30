add_executable(multirobot_incsim_test_new
  multirobot_incsim_test.cpp
)

target_link_libraries(multirobot_incsim_test_new PRIVATE
  G2O_Graph
  GlennMultirobot
  Incremental_Visualizer
)
