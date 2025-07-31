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

#include "file_slam_system.h"



namespace g2o {
namespace tutorial {

using namespace Eigen;

using VertexContainer = g2o::OptimizableGraph::VertexContainer;



  FileSlamSystem::FileSlamSystem(const std::string& filename)
    :SlamSystemBase<VertexSE3, EdgeSE3>(filename){

  }

  FileSlamSystem::~FileSlamSystem()=default;


  void FileSlamSystem::start(){

  }


  void FileSlamSystem::stop(){
    optimize(optCountStop_);

    //if (fixOlderPlatformVertices_ == true){
    // TODO We are doing id = 0 for now.
    for (const auto& vertex : optimizer_->vertices()) {
      g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(vertex.second);
      if(v->id() > 0){
        v->setFixed(false);
      }
    }
    optimize(optCountStopFix_);
  }

  void FileSlamSystem::processEvent(Event& event){
    switch(event.type()){
      case Event::EventType::FileObservation:
        handleObservationEvent(static_cast<FileObsEvent&>(event));
        break;
      case Event::EventType::FileOdometry:
        stepNumber_ +=1;
        handleOdometryEvent(static_cast<FileOdomEvent&>(event));
        break;
      case Event::EventType::FileInitialization:
        handleInitializationEvent(static_cast<FileInitEvent&>(event));
        break;
      default:
        if(verbose_){std::cout << " - Unknown Event ..." << std::endl;}
        ignoreUnknownEventType();
        break;
    }
  }



  void FileSlamSystem::ignoreUnknownEventType(){}


  void FileSlamSystem::handleInitializationEvent(FileInitEvent event){
    if(verbose_){std::cout << " - SlamSystem handleInitializationEvent start ..." << std::endl;}
    if(verbose_){std::cout << " - Creaing vertex ..." << std::endl;}
    currentPlatformVertex_ = new VertexSE3();


    currentPlatformVertex_->setId(event.vtxId);
    currentPlatformVertex_->setEstimate(event.value);
    if(verbose_){std::cout << " - Adding vertex to optimizer ..." << std::endl;}
    optimizer_->addVertex(currentPlatformVertex_);

    platformVertices_.emplace_back(currentPlatformVertex_);
    // place the id into the vertex id map
    if (VertexIdMap_.find(event.vtxId) == VertexIdMap_.end()) {
      VertexIdMap_[event.vtxId] = static_cast<int>(VertexIdMap_.size());
    }

    // TODO replace with initialization prior
    if(verbose_){std::cout << " - Fixing initial vertex ..." << std::endl;}
    currentPlatformVertex_->setFixed(true);
    initialized_ = true;
    if(verbose_){std::cout << " - SlamSystem handleInitializationEvent end ..." << std::endl;}

  }


  void FileSlamSystem::handleOdometryEvent(FileOdomEvent event){
    if(verbose_){std::cout << " - SlamSystem handleOdometryEvent start ..." << std::endl;}
    if(verbose_){std::cout << " - Creaing vertex ..." << std::endl;}
    currentPlatformVertex_ = new VertexSE3();


    currentPlatformVertex_->setId(event.vtxId);
    if(verbose_){std::cout << " - Adding vertex to optimizer ..." << std::endl;}
    optimizer_->addVertex(currentPlatformVertex_);

    platformVertices_.emplace_back(currentPlatformVertex_);
    // place the id into the vertex id map
    if (VertexIdMap_.find(event.vtxId) == VertexIdMap_.end()) {
      VertexIdMap_[event.vtxId] = static_cast<int>(VertexIdMap_.size());
    }

    // TODO replace with initialization prior
    if(verbose_){std::cout << " - Adding Edge ..." << std::endl;}
    EdgeSE3* odometry = new EdgeSE3();
    VertexSE3* v0 = platformVertices_[platformVertices_.size() - 2];
    odometry->setVertex(0, v0);
    odometry->setVertex(1, currentPlatformVertex_);

    OptimizableGraph::VertexSet fromSet;
    fromSet.insert(v0);
    //odometry->initialEstimate(fromSet, v0);
    currentPlatformVertex_->setEstimate(v0->estimate() * event.value);

    if(verbose_){std::cout << " - Vertex set, setting measurments ..." << std::endl;}
    odometry->setMeasurement(event.value);
    //assert(odometry->information().rows() == 3);
    if(verbose_){std::cout << " - measurements set, setting information ..." << std::endl;}
    odometry->setInformation((event.information));
    if(verbose_){std::cout << " - Adding edge to optimizer ..." << std::endl;}
    optimizer_->addEdge(odometry);

    processModelEdges_.emplace_back(odometry);
    numProcessModelEdges_ += 1;

    if(verbose_ && false){
      // Optional: Set formatting for better readability
      std::cout << std::fixed << std::setprecision(6);
      std::cout << "\n\n\n\n Current Vertex\n";
      Isometry3 v0Iso  = v0->estimate();
      std::cout << "Translation (x y z):\n" << v0Iso.translation().transpose() << "\n\n";
      std::cout << "Rotation matrix (3x3):\n" << v0Iso.rotation() << "\n\n";
      // Print translation
      std::cout << "Odom Edge Vertex\n";
      std::cout << "Translation (x y z):\n" << event.value.translation().transpose() << "\n\n";
      // Print rotation matrix
      std::cout << "Rotation matrix (3x3):\n" << event.value.rotation() << "\n\n";
      // Print quaternion form
      Quaterniond q(event.value.rotation());
      std::cout << "Quaternion (w x y z):\n" 
          << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "\n\n";
      // Print full isometry matrix
      std::cout << "Isometry3 (4x4 homogeneous transformation):\n" << event.value.matrix() << "\n\n";
      // Print 6x6 information matrix
      std::cout << "Information matrix (6x6):\n" << event.information << "\n";
    }


    if(verbose_){std::cout << " - SlamSystem handleOdometryEvent end ..." << std::endl;}
  }


  void FileSlamSystem::handleObservationEvent(FileObsEvent event){
    //TODO implement
     if(verbose_){std::cout << " - SlamSystem handleObservationEvent start ..." << std::endl;}

  }

  void FileSlamSystem::platformEstimate2d(Eigen::Vector3d& x, Eigen::Matrix2d& P){

  }

  void FileSlamSystem::platformEstimate2d(Eigen::Vector3d& pose) const {
    // 1) translation
    Isometry3 pos = currentPlatformVertex_->estimate();
    const Eigen::Vector3d& t3 = pos.translation();
    double x = t3.x();
    double y = t3.y();

    // 2) extract yaw from rotation matrix R
    Eigen::Matrix3d R = pos.rotation();
    // yaw = atan2(sinθ, cosθ) = atan2(R(1,0), R(0,0))
    double yaw = std::atan2(R(1,0), R(0,0));

    pose = Eigen::Vector3d(x, y, yaw);
  }


}  // namespace tutorial
}  // namespace g2o
