add_library(Oneshot_Simulator ${LIB_TYPE}
  events.h incremental_simulator.cpp system_model.cpp ordered_event_queue.hpp
)

target_link_libraries(Oneshot_Simulator PRIVATE G2O_Graph)
#target_include_directories(Oneshot_Simulator PUBLIC ${CMAKE_CURRENT_DIRECTORY})
target_include_directories(Oneshot_Simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


target_link_libraries(Oneshot_Simulator PUBLIC
  Eigen3::Eigen
  g2o::core
  g2o::stuff
  g2o::solver_eigen
)

target_include_directories(Oneshot_Simulator PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

