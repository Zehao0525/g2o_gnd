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


#include "file_simulator.h"
#include "file_slam_system.h"
#include "types_tutorial_slam2d.h"
#include "vertex_point_xy.h"
#include "vertex_se2.h"
#include "view_manager.h"
#include "file_simulator_view.h"
#include "file_slam_system_view.h"

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

  std::string viewFilenme = "Source/Examples/Tutorial_slam2d/multirobot_configs/overall_view.json";
  std::ifstream f(viewFilenme);
  if (!f) {
      throw std::runtime_error("Cannot open Simulator config file: " + viewFilenme);
  }
  nlohmann::json viewJson;
  f >> viewJson;
  int frame_pause = viewJson.value("frame_pause", 50000);
  int num_bots = viewJson.value("num_robots", 5);
  int loop_count = viewJson.value("loop_count", 1000000);
  int comms_interval = viewJson.value("comms_interval", 1000);

  bool running = true;
  vector<std::shared_ptr<FileSimulator>> filesims;
  vector<std::shared_ptr<FileSlamSystem>> fileslamsystems;
  vector<std::shared_ptr<viz::FileSimulatorView>> simVizers;
  vector<std::shared_ptr<viz::FileSlamSystemView>> slamVizers;
  viz::ViewManager vizer = viz::ViewManager(viewFilenme);
  for(int i=0;i<num_bots;i++){
    cerr << "Adding Simulator and visualizer for bot " << i << endl;
    std::string datafilename = viewJson.value("file_source", "test1_2_data") + "/bot" + std::to_string(i);
    std::shared_ptr<FileSimulator> filesimPtr = std::make_shared<FileSimulator>(i, datafilename);
    std::string slamConfig = "Source/Examples/Tutorial_slam2d/multirobot_configs/slam_system_config.json";
    std::shared_ptr<FileSlamSystem> fileslamsystemPtr = std::make_shared<FileSlamSystem>(i, slamConfig);

    filesims.emplace_back(filesimPtr);
    fileslamsystems.emplace_back(fileslamsystemPtr);

    std::string configname = "Source/Examples/Tutorial_slam2d/multirobot_configs/bot" + std::to_string(i) + "_view.json";
    std::shared_ptr<viz::FileSimulatorView> simVizerPtr = std::make_shared<viz::FileSimulatorView>(filesimPtr.get(), configname);
    std::shared_ptr<viz::FileSlamSystemView> slamVizerPtr = std::make_shared<viz::FileSlamSystemView>(fileslamsystemPtr.get(), configname);
    vizer.addView(simVizerPtr);
    vizer.addView(slamVizerPtr);
    simVizers.emplace_back(simVizerPtr);
    slamVizers.emplace_back(slamVizerPtr);
  }

  //vizer.addView(slamVizer);



  cerr << "Simulator starting ... "<<endl;
  for(int i=0;i<num_bots;i++){
    filesims[i]->start();
    fileslamsystems[i]->start();
  }
  //cerr << "SLAM System Starting ..."<<endl;
  //slamSystem.start();
  cerr << "Visualizer Starting ..."<<endl;
  vizer.start();
  cerr << "Aquiring events from simulator ..."<<endl;
  for(int i=0;i<num_bots;i++){
    std::vector<EventPtr> events = filesims[i]->aquireEvents();
    simVizers[i]->processEvents(events);
    fileslamsystems[i]->processEvents(events);
    slamVizers[i]->update();

    std::cout << "sys" << i << " optimizer@ " << fileslamsystems[i]->optimizer() << "\n";
  }
  
  for(int i=0;i<6000;i++){
    cerr <<endl;
    cerr << "(loop) iteration: " << i <<endl;
    //cerr << "(loop) slam system processing events ... ..."<<endl;
    //slamSystem.processEvents(events);
    for(int j=0;j<num_bots;j++){
      cerr << "(loop) aquiring events from simulator "<<j<<" ... ..."<<endl;
      std::vector<EventPtr> events = filesims[j]->aquireEvents();
      cerr << "(loop) stepping simulator "<<j<<" ... ..."<<endl;
      filesims[j]->step();
      cerr << "(loop) vizer processing events ... ..."<<endl;
      simVizers[j]->processEvents(events);

      cerr << "(loop) slam system processing event ... ..."<<endl;
      fileslamsystems[j]->processEvents(events);
      cerr << "(loop) slam vizer updating ... ..."<<endl;
      slamVizers[j]->update();
      Vector3d simx = filesims[j]->xTrue2d().toVector();
      simVizers[j]->updateRobotPose(simx);
    }

    if(i%comms_interval == 0){
      for(int j=0;j<num_bots;j++){
        FileSlamSystem::ObsSyncMessage sync_msg = fileslamsystems[j]->broadcastObsSyncMessage();
        for(int k=0;k<num_bots;k++){
          FileSlamSystem::ObsSyncMessage osmsg = fileslamsystems[k]->handleObservationSyncRequest(sync_msg);
          fileslamsystems[j]->handleObservationSyncResponse(osmsg);
        }
      }
    }



    //slamVizer->update();


    // use slamSystem.platformEstimate(SE2& x, Matrix2d& P) to access the estimated position.
    // use incsim.xTrue() to access the ground truth. 
    for(int j=0;j<num_bots;j++){
      if(filesims[j]->keepRunning()){
        break;
      }
      if(j >= num_bots-1){
        running = false;
      }
    }
    if(!running || i >= loop_count){
      cerr << " loop break"<<endl; 
      break;
    }
    usleep(frame_pause);
  }
  cerr << endl;
  // incsim.stop();
  //cerr << "SLAM System Stopping ..."<<endl;
  //slamSystem.saveOptimizerResults("trajectory_before.g2o");
  //slamSystem.stop();

  for(int j=0;j<num_bots;j++){
    Vector3d simx = filesims[j]->xTrue2d().toVector();
    simVizers[j]->updateRobotPose(simx);
    slamVizers[j]->update();
  }

  for(int j=0;j<num_bots;j++){
    fileslamsystems[j]->optimize(20);
  }

  for(int i=0;i<num_bots;i++){
    std::string filename = "file_trajectory_pre_comm_bot" + std::to_string(i) + ".g2o";
    fileslamsystems[i]->saveOptimizerResults(filename);
  }
  
  for(int a=0;a<1;a++){
    for(int j=0;j<num_bots;j++){
      cerr << " j:"<<j<<endl; 
      FileSlamSystem::ObsSyncMessage sync_msg = fileslamsystems[j]->broadcastObsSyncMessage();
      for(int k=0;k<num_bots;k++){
        cerr << " k:"<<k<<endl; 
        if(j==k){continue;}
        FileSlamSystem::ObsSyncMessage osmsg = fileslamsystems[k]->handleObservationSyncRequest(sync_msg);
        fileslamsystems[j]->handleObservationSyncResponse(osmsg);
      }
    }
    for(int j=0;j<num_bots;j++){
      fileslamsystems[j]->optimize(20);
    }
  }

  for(int i=0;i<num_bots;i++){
    fileslamsystems[i]->stop();
    std::string filename = "file_trajectory_gt_bot" + std::to_string(i) + ".g2o";
    std::string slamfilename = "file_trajectory_opt_bot" + std::to_string(i) + ".g2o";
    filesims[i]->saveGroundTruth(filename);
    fileslamsystems[i]->saveOptimizerResults(slamfilename);
  }
  //slamVizer->update();
  vizer.pause();
  sleep(5);
  vizer.stop();

  //slamSystem.saveOptimizerResults("trajectory_est.g2o");
}
