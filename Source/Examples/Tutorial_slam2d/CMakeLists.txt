add_executable(tutorial_slam2d
  tutorial_slam2d.cpp
)

target_link_libraries(tutorial_slam2d PRIVATE
  G2O_Graph
  Oneshot_Simulator
)
