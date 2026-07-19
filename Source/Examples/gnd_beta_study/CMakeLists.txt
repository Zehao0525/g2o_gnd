find_package(nlohmann_json REQUIRED)

add_executable(tutorial_references_test
  tutorial_w_references.cpp
)

target_link_libraries(tutorial_references_test PRIVATE
  G2O_Graph
  Oneshot_Simulator
  nlohmann_json::nlohmann_json
)

add_executable(tutorial_gps_period_sweep
  tutorial_gps_period_sweep.cpp
)

target_link_libraries(tutorial_gps_period_sweep PRIVATE
  G2O_Graph
  Oneshot_Simulator
  nlohmann_json::nlohmann_json
)

add_executable(tutorial_beta_sweep
  tutorial_beta_sweep.cpp
)

target_link_libraries(tutorial_beta_sweep PRIVATE
  G2O_Graph
  Oneshot_Simulator
  nlohmann_json::nlohmann_json
)
