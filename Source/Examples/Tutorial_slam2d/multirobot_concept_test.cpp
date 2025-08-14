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


void writeObsSyncRequestsToFile(const FileSlamSystem::ObsSyncMessage& osmsg,
                                 const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "❌ Failed to open " << filename << " for writing.\n";
        return;
    }

    for (const auto& req : osmsg.syncRequests) {
        const Eigen::Isometry3d& pose = req.observedVtxLocation;

        // Extract translation
        Eigen::Vector3d t = pose.translation();

        // Extract rotation as quaternion (Eigen::Quaternion is w, x, y, z order)
        Eigen::Quaterniond q(pose.rotation());

        // Write in g2o VERTEX_SE3:QUAT format: qx qy qz qw
        out << "VERTEX_SE3:QUAT " << req.observedVertexId << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
            << "\n";
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
  std::shared_ptr<FileSimulator> filesim;
  std::shared_ptr<FileSlamSystem> fileslamsystem;
  std::shared_ptr<viz::FileSimulatorView> simVizer;
  std::shared_ptr<viz::FileSlamSystemView> slamVizer;
  viz::ViewManager vizer = viz::ViewManager(viewFilenme);


  cerr << "Adding Simulator and visualizer for bot " << 0 << endl;
  std::string datafilename = viewJson.value("file_source", "test1_new_data") + "/bot0";
  filesim = std::make_shared<FileSimulator>(0, datafilename);
  std::string slamConfig = "Source/Examples/Tutorial_slam2d/multirobot_configs/slam_system_config.json";
  fileslamsystem = std::make_shared<FileSlamSystem>(0, slamConfig);


  std::string configname = "Source/Examples/Tutorial_slam2d/multirobot_configs/bot0_view.json";
  simVizer = std::make_shared<viz::FileSimulatorView>(filesim.get(), configname);
  slamVizer = std::make_shared<viz::FileSlamSystemView>(fileslamsystem.get(), configname);
  vizer.addView(simVizer);
  vizer.addView(slamVizer);

  //vizer.addView(slamVizer);





  std::filesystem::path folder("test1_new_data/bot1");

  // 2) Build full paths for the two .g2o files
  auto vertPath = folder / "vertices.g2o";

  std::ifstream vf(vertPath);
  if (!vf) { std::cerr << "Cannot open vertices.g2o\n";}

  std::vector<FileSimulator::Vertex> bot1_vertices;
  bot1_vertices.reserve(10000);
  
  std::map<int,int> bot1_idToIndex;

  std::string line;
  while (std::getline(vf, line)) {
    if (line.rfind("VERTEX_SE3:QUAT",0) != 0) continue;
    std::istringstream ss(line);
    std::string tag;
    int id;
    double x,y,z, qx,qy,qz,qw;
    ss >> tag >> id
      >> x >> y >> z
      >> qx >> qy >> qz >> qw;

    Isometry3 pose = Isometry3::Identity();  // Start with identity
    pose.linear() = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();    // Set rotation
    pose.translation() = Eigen::Vector3d(x, y, z);    
    size_t idx = bot1_vertices.size();
    bot1_vertices.emplace_back(id, pose);
    bot1_idToIndex[id] = idx;
  }
  vf.close();




  cerr << "Simulator starting ... "<<endl;

  filesim->start();
  fileslamsystem->start();
  //cerr << "SLAM System Starting ..."<<endl;
  //slamSystem.start();
  cerr << "Visualizer Starting ..."<<endl;
  vizer.start();
  cerr << "Aquiring events from simulator ..."<<endl;
  std::vector<EventPtr> events = filesim->aquireEvents();
  simVizer->processEvents(events);
  fileslamsystem->processEvents(events);
  slamVizer->update();
  
  //cerr << "(loop) slam system processing events ... ..."<<endl;
  //slamSystem.processEvents(events);
  for(int i=0;i<6000;i++){
    cerr <<endl;
    cerr << "(loop) iteration: " << i <<endl;
    std::vector<EventPtr> events = filesim->aquireEvents();
    filesim->step();
    cerr << "(loop) vizer processing events ... ..."<<endl;
    simVizer->processEvents(events);

    cerr << "(loop) slam system processing event ... ..."<<endl;
    fileslamsystem->processEvents(events);
    cerr << "(loop) slam vizer updating ... ..."<<endl;
    slamVizer->update();
    Vector3d simx = filesim->xTrue2d().toVector();
    simVizer->updateRobotPose(simx);

    // TODO CHANGE
    // if(i%comms_interval == 0){
    //   for(int j=0;j<num_bots;j++){
    //     FileSlamSystem::ObsSyncMessage sync_msg = fileslamsystem->broadcastObsSyncMessage();
    //     for(int k=0;k<num_bots;k++){
    //       FileSlamSystem::ObsSyncMessage osmsg = fileslamsystems[k]->handleObservationSyncRequest(sync_msg);
    //       fileslamsystems[j]->handleObservationSyncResponse(osmsg);
    //     }
    //   }
    // }



    //slamVizer->update();


    // use slamSystem.platformEstimate(SE2& x, Matrix2d& P) to access the estimated position.
    // use incsim.xTrue() to access the ground truth. 
    if(!filesim->keepRunning()){
      break;
    }
    usleep(frame_pause);
  }
  cerr << endl;
  // incsim.stop();
  // cerr << "SLAM System Stopping ..."<<endl;
  // slamSystem.saveOptimizerResults("trajectory_before.g2o");
  // slamSystem.stop();

  std::string pofilename = "test_results/file_trajectory_pre_opt_bot0.g2o";
  fileslamsystem->saveOptimizerResults(pofilename);

  Vector3d simx = filesim->xTrue2d().toVector();
  simVizer->updateRobotPose(simx);
  slamVizer->update();

  fileslamsystem->optimize(20);

  std::string filename = "test_results/file_trajectory_pre_comm_bot0.g2o";
  fileslamsystem->saveOptimizerResults(filename);
  
  FileSlamSystem::ObsSyncMessage osmsg;
  for(int a=0;a<1;a++){
    FileSlamSystem::ObsSyncMessage sync_msg = fileslamsystem->broadcastObsSyncMessage();

    std::vector<FileSlamSystem::ObsSyncRequest> localRequests = sync_msg.syncRequests;
    std::vector<FileSlamSystem::ObsSyncRequest> validResponses;
    validResponses.reserve(localRequests.size());
    for (size_t i = 0; i < localRequests.size(); ++i) {
      //if(verbose_){std::cout << "entering for loop \n";}
      auto& req = localRequests[i];
      //if(! (req.observerVertexId == 2606 || req.observerVertexId == 898 || req.observerVertexId == 2605 )){continue;}
      req.observedVtxLocation = bot1_vertices[bot1_idToIndex[req.observedVertexId]].pose;
      std::cout << "Translation (x y z):\n" << req.observedVtxLocation.translation().transpose() << "\n\n";
      std::cout << "Rotation matrix (3x3):\n" << req.observedVtxLocation.rotation() << "\n\n";

      req.observedVtxInformation = Eigen::Matrix<double,6,6>::Identity();
      validResponses.push_back(std::move(req));
      
    }

    // Step 5: return only the valid responses
    osmsg =  FileSlamSystem::ObsSyncMessage(
      /* sourceId  */ sync_msg.sourceId,
      /* outGoing  */ false,
      /* syncReqs  */ std::move(validResponses)
    );

    writeObsSyncRequestsToFile(osmsg, "test_results/bot0_observation_vtxs_refs.g2o");

    
    
    // TODO Communication
    // FileSlamSystem::ObsSyncMessage osmsg = fileslamsystem->handleObservationSyncRequest(sync_msg);
    fileslamsystem->handleObservationSyncResponse(osmsg);
    fileslamsystem->optimize(20);
    std::string slamfilename1 = "test_results/file_trajectory_opt_wognd_bot0.g2o";
    fileslamsystem->saveOptimizerResults(slamfilename1);
    (fileslamsystem->gndActive_) = true;

  }

  fileslamsystem->stop();
  std::string gtfilename = "test_results/file_trajectory_gt_bot0.g2o";
  std::string slamfilename = "test_results/file_trajectory_opt_bot0.g2o";
  filesim->saveGroundTruth(gtfilename);
  fileslamsystem->saveOptimizerResults(slamfilename);
  //slamVizer->update();
  vizer.pause();
  sleep(5);
  vizer.stop();

  //slamSystem.saveOptimizerResults("trajectory_est.g2o");
}
