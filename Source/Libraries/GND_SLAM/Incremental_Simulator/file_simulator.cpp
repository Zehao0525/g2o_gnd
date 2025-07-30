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

#include "file_simulator.h"

#include <cmath>
#include <iostream>
#include <map>
#include <fstream>
#include <iomanip>

#include "g2o/stuff/sampler.h"
using namespace std;

namespace g2o {
namespace tutorial {

void forceLinkTypesTutorialSlam2d();

using namespace Eigen;


FileSimulator::FileSimulator(const std::string& filename):
      currentVtxNumber_(0), carryOnRunning_(true), initialized_(false){
       // 1) Read the folder name from the command line
    std::filesystem::path folder(filename);

    // 2) Build full paths for the two .g2o files
    auto vertPath = folder / "vertices.g2o";
    auto edgePath = folder / "edges.g2o";
    auto obsEdgePath = folder / "observation_edges.g2o";

    std::ifstream vf(vertPath);
    if (!vf) { std::cerr << "Cannot open vertices.g2o\n";}

    vertices_.reserve(10000);
    
    idToIndex_.reserve(10000);

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

      SE3Quat pose(
        Eigen::Quaterniond(qw, qx, qy, qz), 
        Eigen::Vector3d(x, y, z)
      );
      size_t idx = vertices_.size();
      vertices_.emplace_back(id, pose);
      idToIndex_[id] = idx;
    }
    vf.close();

    // --- load edges.g2o ---
    std::ifstream ef(edgePath);
    if (!ef) { std::cerr << "Cannot open edges.g2o\n";}

    
    odomEdgesFrom_.reserve(10000);

    while (std::getline(ef, line)) {
      if (line.rfind("EDGE_SE3:QUAT",0) != 0) continue;
      std::istringstream ss(line);
      std::string tag;
      int v0, v1;
      double dx,dy,dz, dqx,dqy,dqz,dqw;
      ss >> tag >> v0 >> v1
        >> dx >> dy >> dz
        >> dqx >> dqy >> dqz >> dqw;

      // build SE3Quat delta
      SE3Quat delta(
        Eigen::Quaterniond(dqw, dqx, dqy, dqz),
        Eigen::Vector3d(dx, dy, dz)
      );

      // read 21 upper-triangular entries into a 6×6 matrix
      Eigen::Matrix<double, 6,6> I = Eigen::Matrix<double, 6,6>::Zero();
      // mapping of linear index i=0..20 to (row,col)
      int idx = 0;
      for (int row = 0; row < 6; ++row) {
        for (int col = row; col < 6; ++col) {
          double v; ss >> v;
          I(row,col) = v;
          I(col,row) = v;
          ++idx;
        }
      }

      odomEdgesFrom_[v0].emplace_back(v0, v1, delta, I);
    }
    ef.close();



    // --- load observation_edges.g2o ---
    std::ifstream oef(obsEdgePath);
    if (!oef) { std::cerr << "Cannot open observation_edges.g2o\n";}
    obsEdgesFrom_.reserve(10000);
    while (std::getline(oef, line)) {
      if (line.rfind("EDGE_SE3:QUAT",0) != 0) continue;
      std::istringstream ss(line);
      std::string tag;
      int v0, v1;
      double dx,dy,dz, dqx,dqy,dqz,dqw;
      ss >> tag >> v0 >> v1
        >> dx >> dy >> dz
        >> dqx >> dqy >> dqz >> dqw;

      // build SE3Quat delta
      SE3Quat delta(
        Eigen::Quaterniond(dqw, dqx, dqy, dqz),
        Eigen::Vector3d(dx, dy, dz)
      );

      // read 21 upper-triangular entries into a 6×6 matrix
      Eigen::Matrix<double, 6,6> I = Eigen::Matrix<double, 6,6>::Zero();
      // mapping of linear index i=0..20 to (row,col)
      int idx = 0;
      for (int row = 0; row < 6; ++row) {
        for (int col = row; col < 6; ++col) {
          double v; ss >> v;
          I(row,col) = v;
          I(col,row) = v;
          ++idx;
        }
      }

      int known_id = -1;
      bool keepOrder = true;
      if (idToIndex_.count(v0)) {
        known_id = v0;
      } else if (idToIndex_.count(v1)) {
        known_id = v1;
        keepOrder = false;
      }

      if (known_id != -1) {
        if(keepOrder){obsEdgesFrom_[known_id].emplace_back(v0, v1, delta, I);}
        else{obsEdgesFrom_[known_id].emplace_back(v1, v0, delta, I);}
      } else {
        if (!verbose_) { std::cerr << "edge does not match existing verticies\n"; }
      }
    }
    oef.close();
}

FileSimulator::~FileSimulator() = default;


SE3Quat FileSimulator::xTrue() const{
  return x_;
}

SE2 FileSimulator::xTrue2d() const{
  // 1) translation
  const Eigen::Vector3d& t3 = x_.translation();
  double x = t3.x();
  double y = t3.y();

  // 2) extract yaw from rotation matrix R
  Eigen::Matrix3d R = x_.rotation().toRotationMatrix();
  // yaw = atan2(sinθ, cosθ) = atan2(R(1,0), R(0,0))
  double yaw = std::atan2(R(1,0), R(0,0));

  return SE2(x, y, yaw);
}


void FileSimulator::history(std::vector<double> &timeHistory, std::vector<g2o::SE3Quat> & xTrueHistory) const{
  timeHistory = timeStore_;
  xTrueHistory = xTrueStore_;
}

void FileSimulator::start(){
  //start@ebe.core.EventBasedSimulator(obj);
  forceLinkTypesTutorialSlam2d();

  if(verbose_){std::cout << " - Trigger initialization event ... " << std::endl;}
  initialize();


  if(verbose_){std::cout << " - FileSimulator start() complete ... " << std::endl;}
}


std::vector<EventPtr> FileSimulator::aquireEvents(){
    std::vector<EventPtr> events = eventQueue_.orderedEvents();
    eventQueue_.clear();
    return events;
  }



//Scenario FileSimulator::getScenario() const;

bool FileSimulator::keepRunning() const{
  return ((carryOnRunning_) && (currentVtxNumber_ < vertices_.size()));
}

void FileSimulator::step(){
  currentVtxNumber_ ++;
  FileSimulator::Vertex& currentVtx = vertices_[currentVtxNumber_];
  for(auto obsEdge : obsEdgesFrom_[currentVtx.id]){
    updateObservation(obsEdge);
  }
  updateOdometry(odomEdgesFrom_[currentVtx.id][0]);
  x_ = currentVtx.pose;
  storeStepResults();
}


void FileSimulator::initialize(){
  currentVtxNumber_ = 0;
  FileSimulator::Vertex& currentVtx = vertices_[currentVtxNumber_];
  std::shared_ptr<FileInitEvent> initEventPtr = std::make_shared<FileInitEvent>(currentVtxNumber_, currentVtx.id, currentVtx.pose, Eigen::Matrix<double,6,6>::Identity());
  eventQueue_.push(initEventPtr);
}
// We consult the system model and we give an event.




void FileSimulator::updateOdometry(FileSimulator::Edge& odomEdge){
  // Psudo time: We make sure that the Odometry has time greater than observation, this way the SLAM system will observe first then tally odometry
  std::shared_ptr<FileOdomEvent> odomEventPtr= std::make_shared<FileOdomEvent>(currentVtxNumber_ + 0.7, odomEdge.v1, odomEdge.delta, odomEdge.info);
  eventQueue_.push(odomEventPtr);
  
}




//void predictGPSObservation();
//void predictCompassObservation();
//void predictBearingObservations();
void FileSimulator::updateObservation(FileSimulator::Edge& obsEdge) {
  std::shared_ptr<FileObsEvent> obsEventPtr= std::make_shared<FileObsEvent>( currentVtxNumber_ + 0.7, obsEdge.v1, obsEdge.delta, obsEdge.info);
  eventQueue_.push(obsEventPtr);
}


//void generateHeartbeat();
void FileSimulator::storeStepResults(){
  timeStore_.emplace_back(currentVtxNumber_);
  xTrueStore_.emplace_back(xTrue());
}

void FileSimulator::saveGroundTruth(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < xTrueStore_.size(); ++i) {
        const auto& pose = xTrueStore_[i].toVector();
        out << std::fixed << std::setprecision(6);
        out << "VERTEX_SE3:QUAT" << i << " "
            << pose[0] << " "
            << pose[1] << " "
            << pose[2] << " "
            << pose[3] << " "
            << pose[4] << " "
            << pose[5] << " "
            << pose[6] << "\n";
    }

    // Optional: fix the first pose if needed
    if (!xTrueStore_.empty()) {
        out << "FIX 0\n";
    }

    out.close();
    std::cout << "Ground truth written to " << filename << std::endl;
}


}  // namespace tutorial
}  // namespace g2o
