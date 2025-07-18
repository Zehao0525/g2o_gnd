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

using namespace std;
using namespace g2o;
using namespace g2o::tutorial;

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
  IncrementalSimulator incsim = IncrementalSimulator();
  SlamSystem slamSystem = SlamSystem();

  cerr << "Simulator starting ... "<<endl;
  incsim.start();
  cerr << "SLAM System Starting ..."<<endl;
  slamSystem.start();
  cerr << "Aquiring events from simulator ..."<<endl;
  std::vector<EventPtr> events = incsim.aquireEvents();
  cerr << "Slam system processing events ..."<<endl;
  slamSystem.processEvents(events);
  for(int i=0;i<10;i++){
    cerr <<endl;
    cerr << "(loop) iteration: " << i <<endl;
    incsim.step();
    cerr << "(loop) aquiring events from simulator ... ..."<<endl;
    std::vector<EventPtr> events = incsim.aquireEvents();
    cerr << "(loop) slam system processing events ... ..."<<endl;
    slamSystem.processEvents(events);
    cerr << "(loop) determining loop break ... ..."<<endl;
    if(!incsim.keepRunning()){
      cerr << " loop break"<<endl;
      break;
    }
  }
  cerr << endl;
  // incsim.stop();
  cerr << "SLAM System Stopping ..."<<endl;
  slamSystem.stop();
}
