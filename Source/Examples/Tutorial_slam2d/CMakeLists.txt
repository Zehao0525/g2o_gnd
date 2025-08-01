add_executable(tutorial_slam2d
  tutorial_slam2d.cpp
)

target_link_libraries(tutorial_slam2d PRIVATE
  G2O_Graph
  Oneshot_Simulator
)


add_executable(incsim_test
  incsim_test.cpp
)


target_link_libraries(incsim_test PRIVATE
  G2O_Graph
  Incremental_Simulator
  Incremental_Visualizer
)


add_executable(multirobot_incsim_test
  multirobot_incsim_test.cpp
)


target_link_libraries(multirobot_incsim_test PRIVATE
  G2O_Graph
  Incremental_Simulator
  Incremental_Visualizer
)



add_executable(cauchy_gauss_test
  cauchy_edge_validity_test.cpp
)

add_executable(cauchy_cauchy_test
  cauchy_edge_validity_test2.cpp
)


target_link_libraries(cauchy_gauss_test PRIVATE
  G2O_Graph
  Oneshot_Simulator
)

target_link_libraries(cauchy_cauchy_test PRIVATE
  G2O_Graph
  Oneshot_Simulator
)


add_executable(gauss_clone_test
  cauchy_edge_validity_test3.cpp
)


target_link_libraries(gauss_clone_test PRIVATE
  G2O_Graph
  Oneshot_Simulator
)