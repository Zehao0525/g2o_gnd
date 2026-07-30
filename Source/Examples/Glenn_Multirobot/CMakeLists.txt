add_executable(multirobot_incsim_test
  multirobot_incsim_test.cpp
)

target_link_libraries(multirobot_incsim_test PRIVATE
  G2O_Graph
  GlennMultirobot
  Incremental_Visualizer
)

add_executable(multirobot_concept_test
  multirobot_concept_test.cpp
)

target_link_libraries(multirobot_concept_test PRIVATE
  G2O_Graph
  GlennMultirobot
  Incremental_Visualizer
)

add_executable(multibot_full_graph
  multibot_full_graph.cpp
)

target_link_libraries(multibot_full_graph PRIVATE
  G2O_Graph
  GlennMultirobot
)
