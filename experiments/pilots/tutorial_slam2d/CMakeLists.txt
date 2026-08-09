add_executable(tutorial_slam2d
  tutorial_slam2d.cpp
)

target_link_libraries(tutorial_slam2d PRIVATE
  G2O_Graph
  Oneshot_Simulator
)

add_executable(tutorial_bearing_test
  tutorial_w_bearing.cpp
)

target_link_libraries(tutorial_bearing_test PRIVATE
  G2O_Graph
  GGD_Core
  Oneshot_Simulator
)
