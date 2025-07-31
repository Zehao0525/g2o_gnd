// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <iostream>
#include <unistd.h>

#include "edge_se2.h"
#include "edge_se2_pointxy.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"


#include "incremental_simulator.h"
#include "slam_system.h"
#include "types_tutorial_slam2d.h"
#include "vertex_point_xy.h"
#include "vertex_se2.h"
#include "view_manager.h"
#include "simulator_view.h"
#include "slam_system_view.h"

using namespace std;
using namespace g2o;
using namespace g2o::tutorial;
using namespace Eigen;

namespace g2o::tutorial {
  void forceLinkTypesTutorialSlam2d();  // Forward declaration

  void checkTypeRegistration() {
      auto* factory = g2o::Factory::instance();

      std::vector<std::string> tags = {
          "TUTORIAL_VERTEX_SE2",
          "TUTORIAL_VERTEX_POINT_XY",
          "TUTORIAL_PARAMS_SE2_OFFSET",
          "TUTORIAL_CACHE_SE2_OFFSET",
          "TUTORIAL_EDGE_SE2",
          "TUTORIAL_EDGE_SE2_POINT_XY"
      };

      for (const auto& tag : tags) {
          if (factory->knowsTag(tag)) {
              std::cout << "✅ Factory knows type: " << tag << std::endl;
          } else {
              std::cout << "❌ Factory DOES NOT know type: " << tag << std::endl;
          }
      }
  }
}


int main() {
  forceLinkTypesTutorialSlam2d();
  checkTypeRegistration();

  std::string viewFilenme = "Source/Examples/Tutorial_slam2d/view_config.json";

  IncrementalSimulator incsim = IncrementalSimulator("Source/Examples/Tutorial_slam2d/simulator_config.json");
  SlamSystem slamSystem = SlamSystem("Source/Examples/Tutorial_slam2d/slam_system_config.json");
  viz::ViewManager vizer = viz::ViewManager("Source/Examples/Tutorial_slam2d/view_config.json");
  std::shared_ptr<viz::SimulatorView> simVizer = std::make_shared<viz::SimulatorView>(&incsim, "Source/Examples/Tutorial_slam2d/view_config.json");
  //std::shared_ptr<viz::SLAMSystemView> slamVizer = std::make_shared<viz::SLAMSystemView>(&slamSystem, Vector3f(0.0f, 0.0f, 1.0f));
  std::shared_ptr<viz::SLAMSystemView> slamVizer = std::make_shared<viz::SLAMSystemView>(&slamSystem,viewFilenme);
  vizer.addView(simVizer);
  vizer.addView(slamVizer);



  std::ifstream f(viewFilenme);
    if (!f) {
        throw std::runtime_error("Cannot open Simulator config file: " + viewFilenme);
    }
  nlohmann::json viewJson;
  f >> viewJson;
  int frame_pause = viewJson.value("frame_pause", 50000);



  cerr << "Simulator starting ... "<<endl;
  incsim.start();
  cerr << "SLAM System Starting ..."<<endl;
  slamSystem.start();
  cerr << "Visualizer Starting ..."<<endl;
  vizer.start();
  cerr << "Aquiring events from simulator ..."<<endl;
  std::vector<EventPtr> events = incsim.aquireEvents();
  cerr << "Slam system processing events ..."<<endl;
  slamSystem.processEvents(events);
  simVizer->processEvents(events);
  
  for(int i=0;i<4000;i++){
    cerr <<endl;
    cerr << "(loop) iteration: " << i <<endl;
    incsim.step();
    cerr << "(loop) aquiring events from simulator ... ..."<<endl;
    std::vector<EventPtr> events = incsim.aquireEvents();
    cerr << "(loop) slam system processing events ... ..."<<endl;
    slamSystem.processEvents(events);
    simVizer->processEvents(events);
    cerr << "(loop) determining loop break ... ..."<<endl << endl;

    Vector3d simx = incsim.xTrue().toVector();
    simVizer->updateRobotPose(simx);

    slamVizer->update();


    // use slamSystem.platformEstimate(SE2& x, Matrix2d& P) to access the estimated position.
    // use incsim.xTrue() to access the ground truth. 
    if(!incsim.keepRunning()){
      cerr << " loop break"<<endl;
      break;
    }
    usleep(frame_pause);
  }
  cerr << endl;
  // incsim.stop();
  cerr << "SLAM System Stopping ..."<<endl;
  slamSystem.saveOptimizerResults("trajectory_before.g2o");
  slamSystem.stop();

  Vector3d simx = incsim.xTrue().toVector();
  simVizer->updateRobotPose(simx);

  slamVizer->update();
  vizer.pause();
  sleep(5);
  vizer.stop();

  slamSystem.saveOptimizerResults("trajectory_est.g2o");
  incsim.saveGroundTruth("trajectory_gt.g2o");
}
