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

#include "slam_system.h"

namespace g2o {
namespace tutorial {

using namespace Eigen;

//TODO opt period update
  SlamSystem::SlamSystem(const std::string& filename) : SlamSystemBase<VertexSE2, EdgeVelocitySE2>(filename){
    std::ifstream f(filename);

      // Reading in the files
      if (!f) {
          throw std::runtime_error("Cannot open SLAM config file: " + filename);
      }
      nlohmann::json j;
      f >> j;

      auto offset = j.value("sensor_offset", std::vector<double>{0.0, 0.0, 0.0});
      if (offset.size() != 3) {
          throw std::runtime_error("sensor_offset must be size 3");
      }
      SE2 sensorOffsetTransf(offset[0], offset[1], offset[2]);
      sensorOffset_ = new ParameterSE2Offset();
      sensorOffset_->setOffset(sensorOffsetTransf);
      sensorOffset_->setId(0);

      optimizationAlg_ = j.value("optimization_algorithm", "GaussNewton");
      

      //optimizer_ = std::make_unique<SparseOptimizer>();

      optimizer_->addParameter(sensorOffset_);
      landmarkIdMap_.clear();
  }
  SlamSystem::~SlamSystem(){}

  void SlamSystem::start(){

    // % Set up the event handlers

    // The SLAM system has been started before, a lot of the initialisation work as been done
    if(!componentsReady_){
      componentsReady_ = true;
    }
    // add Initial edges
  }

  void SlamSystem::stop(){
    // % Run the optimizer

    // % If we are fixing past vehicle states (Q3) then handle
    // % unfixing for the final optimization pass
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
    //}
  }

  
  void SlamSystem::setMaxObservationsPerLandmark(int maxObservationsPerLandmark){
    maxObservationsPerLandmark_ = maxObservationsPerLandmark;
  }

  
  void SlamSystem::landmarkEstimates(std::vector<Vector2d>& m, std::vector<Matrix2d>& Pmm, std::vector<int>& landmarkIds){
    // Clear output vectors to avoid accidental accumulation
    m.clear();
    Pmm.clear();
    landmarkIds.clear();

    SparseBlockMatrix<MatrixX> spinv;
    optimizer_->computeMarginals(spinv, landmarkVertices_);

    int numBlocks = spinv.rowBlockIndices().size();

    for (int i = 0; i < numBlocks; ++i) {
        const Eigen::MatrixXd* block = spinv.block(i, i);

        Vector2d est;
        landmarkVertices_[i]->getEstimateData(est);
        m.emplace_back(est);
        landmarkIds.emplace_back(landmarkVertices_[i]->id());
        if (block) {
            // Check size of block
            if (block->rows() < 2 || block->cols() < 2) {
                throw std::runtime_error("Block at (" + std::to_string(i) + "," + std::to_string(i) + ") is too small.");
            }
            Pmm.emplace_back(block->topLeftCorner<2,2>());
        } else {
            // If no block exists, append zero
            Pmm.emplace_back(Eigen::Matrix2d::Zero());
        }
    }

  }

  void SlamSystem::getSceneEstimates(Eigen::Vector3d& x, std::vector<Eigen::Vector2d>& m, std::vector<int>& landmarkIds) const{
    m.clear();
    landmarkIds.clear();

    x = (currentPlatformVertex_->estimate()).toVector();

    for (int i = 0; i < landmarkVertices_.size(); ++i) {
        Vector2d est;
        landmarkVertices_[i]->getEstimateData(est);
        m.emplace_back(est);
        landmarkIds.emplace_back(landmarkVertices_[i]->id());
    }
  }

  void SlamSystem::getSceneEstimatesWithP(Eigen::Vector3d& x, Eigen::Matrix2d& P, std::vector<Eigen::Vector2d>& m, std::vector<Eigen::Matrix2d>& Pmm, std::vector<int>& landmarkIds){

    optimize(20);

    platformEstimate(x,P);
    //landmarkEstimates(m,Pmm,landmarkIds);
  }

  /**
   * @brief process a event
   * @param events event to process
   */
  void SlamSystem::processEvent(Event& event){
    double dT = event.time - currentTime_;
    
    // TODO Change this to match minDT
    if(dT < 1e-3){
      handleNoPrediction();
    }
    else{
      handlePredictForwards(dT);
      currentTime_ = event.time;
      stepNumber_ +=1;
    }

    auto* compEvent = dynamic_cast<CompEventBase*>(&event);
    if (!compEvent) {
      ignoreUnknownEventType();
      return;
    }
    switch (compEvent->compEventType()) {
      case CompEventType::HeartBeat:
        break;
      case CompEventType::LandmarkObservations:
        handleSLAMObservationEvent(static_cast<LandmarkObservationsEvent&>(event));
        break;
      case CompEventType::LMRangeBearingObservations:
        handleRangeBearingObservationEvent(
            static_cast<LMRangeBearingObservationsEvent&>(event));
        break;
      case CompEventType::GPSObservation:
        handleGPSObservationEvent(static_cast<GPSObservationEvent&>(event));
        break;
      case CompEventType::Odometry:
        handleUpdateOdometryEvent(static_cast<OdometryEvent&>(event));
        break;
      case CompEventType::Initialization:
        handleInitializationEvent(static_cast<InitializationEvent&>(event));
        break;
      default:
        ignoreUnknownEventType();
        break;
    }
  }

  /**
   * @brief process a event
   * At this moment we won't have this step. mybe in the future.
   * @param eventType type of event
   * @param eventHandler handler that handles that type of event
   */
  //void registerEventHandler(EventType eventType, EventHandler eventHandler);

  void SlamSystem::ignoreUnknownEventType(){}

  void SlamSystem::handlePredictForwards(double dT){
    SE2 lastpredX = currentPlatformVertex_->estimate();
    SE2 newX = lastpredX * (u_*dT);
    
    currentPlatformVertex_ = new VertexSE2;
    currentPlatformVertex_->setEstimate(newX);
    currentPlatformVertex_->setId(++vertexId_);
    optimizer_->addVertex(currentPlatformVertex_);
    platformVertices_.emplace_back(currentPlatformVertex_);

    EdgeVelocitySE2* odometry = new EdgeVelocitySE2(dT);
    odometry->setVertex(0, platformVertices_[platformVertices_.size() - 2]);
    odometry->setVertex(1, currentPlatformVertex_);
    odometry->setMeasurement(u_);
    assert(odometry->information().rows() == 3);
    odometry->setInformation((sigmaU_).inverse());
    optimizer_->addEdge(odometry);

    processModelEdges_.emplace_back(odometry);
    numProcessModelEdges_ += 1;

    // I'll skip fix older platform vertex for now.
  }

  void SlamSystem::handleNoPrediction(){}

  void SlamSystem::handleInitializationEvent(InitializationEvent event){
    currentPlatformVertex_ = new VertexSE2;
    currentPlatformVertex_->setId(++vertexId_);
    currentPlatformVertex_->setEstimate(event.pose);
    optimizer_->addVertex(currentPlatformVertex_);
    platformVertices_.emplace_back(currentPlatformVertex_);

    // TODO replace with initialization prior
    currentPlatformVertex_->setFixed(true);

    u_ = event.velocity;
    sigmaU_ = event.sigmaU;
    initialized_ = true;
  }

  void SlamSystem::handleUpdateOdometryEvent(OdometryEvent event){
    u_ = event.value;
    sigmaU_ = event.covariance;
  }

  /**
   * @brief event handler for landmark observation events
   * @param event
   */
  void SlamSystem::handleSLAMObservationEvent(LandmarkObservationsEvent event){
    
    //Matrix2d P;
    SE2 curvtxEst = currentPlatformVertex_->estimate();

    for(const auto& lmObs : event.landmarkObservations){
      assert(lmObs.value.size() == 2);
      assert(lmObs.covariance.rows() == 2 && lmObs.covariance.cols() == 2);
      VertexPointXY* lmVertex;
      bool vtxCreated = createOrGetLandmark(lmObs.landmark_id, lmVertex);
      if(vtxCreated){
        lmVertex->setEstimate(curvtxEst * lmObs.value );  // Initial guess
      }

      EdgeSE2PointXY* landmarkObservation = new EdgeSE2PointXY;
      //landmarkObservation->resize(2);
      landmarkObservation->setVertex(0,currentPlatformVertex_);
      landmarkObservation->setVertex(1, lmVertex);

      landmarkObservation->setMeasurement(lmObs.value);

      landmarkObservation->setInformation(lmObs.covariance.inverse());

      landmarkObservation->setParameterId(0, sensorOffset_->id());

      //checkTypeRegistration();
      //landmarkObservation->linearizeOplus();

      optimizer_->addEdge(landmarkObservation);
    }

  }

  void SlamSystem::handleRangeBearingObservationEvent(LMRangeBearingObservationsEvent event){
    
    //Matrix2d P;
    Vector3d curvtxEst = (currentPlatformVertex_->estimate()).toVector();

    for(const auto& lmObs : event.landmarkObservations){
      assert(lmObs.value.size() == 2);
      assert(lmObs.covariance.rows() == 2 && lmObs.covariance.cols() == 2);
      VertexPointXY* lmVertex;
      bool vtxCreated = createOrGetLandmark(lmObs.landmark_id, lmVertex);
      if(vtxCreated){
        double trueBearing = lmObs.value[1] + curvtxEst[2];
        Vector2d disp = Vector2d(lmObs.value[0] * cos(trueBearing) + curvtxEst[0], lmObs.value[0] * sin(trueBearing) + curvtxEst[1]);

        lmVertex->setEstimate(disp);  // Initial guess
      }

      
      EdgeRangeBearing* landmarkObservation = new EdgeRangeBearing;
      //landmarkObservation->resize(2);
      landmarkObservation->setVertex(0,currentPlatformVertex_);
      landmarkObservation->setVertex(1, lmVertex);

      landmarkObservation->setMeasurement(lmObs.value);

      landmarkObservation->setInformation(lmObs.covariance.inverse());

      landmarkObservation->setParameterId(0, sensorOffset_->id());

      //checkTypeRegistration();
      //landmarkObservation->linearizeOplus();

      optimizer_->addEdge(landmarkObservation);
    }

  }

  /**
   * @brief given landmark id, retrieve landmark. Create landmark if landmark not already there
   * @param id landmark id
   * @param lmVertex the returned landmark vertex
   * @return true if landmark created, false otherwise
   */
  bool SlamSystem::createOrGetLandmark(int id, VertexPointXY*& lmVertex){
    auto it = landmarkIdMap_.find(id);

    if (it != landmarkIdMap_.end()) {
        // Landmark already exists
        int idx = it->second;
        lmVertex = static_cast<VertexPointXY*>(landmarkVertices_[idx]);
        return false;
    } else {
        // Create new landmark
        lmVertex = new VertexPointXY();
        lmVertex->setId(++vertexId_);
        // lmVertex->getId();
        optimizer_->addVertex(lmVertex);
        landmarkVertices_.push_back(lmVertex);

        landmarkIdMap_[id] = landmarkVertices_.size() - 1;  // Map landmark id to index
        return true;
    }
  }

  void SlamSystem::handleGPSObservationEvent(GPSObservationEvent event){
    EdgePlatformLocPriorGND* gpsObservation = new EdgePlatformLocPriorGND;
    gpsObservation->setVertex(0,currentPlatformVertex_);
    gpsObservation->setMeasurement(event.value);
    gpsObservation->gndSetInformation(event.covariance.inverse(), 8);
    gpsObservation->setParameterId(0, sensorOffset_->id());
    optimizer_->addEdge(gpsObservation);
  }
  // handlenoUpdate()
  // handleInitializationEvent(event)
  
  // ... all other observations ,,,
  

}  // namespace tutorial
}  // namespace g2o
