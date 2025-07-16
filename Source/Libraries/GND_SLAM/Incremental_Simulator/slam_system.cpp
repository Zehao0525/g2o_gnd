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

thread_local std::unique_ptr<SparseOptimizer> SlamSystem::optimizer_;
//
  SlamSystem::SlamSystem():verbose_(true){}
  SlamSystem::~SlamSystem(){}


  void SlamSystem::start(){
    currentTime_ = 0;
    stepNumber_ = 0;
    initialized_ = false;

    // % Set up the event handlers

    // The SLAM system has been started before, a lot of the initialisation work as been done
    if(!componentsReady_){
      platformVertexId_ = -1;
      numProcessModelEdges_ = 0;
      currentPlatformVertex_ = nullptr;

      // allocating the optimizer
      auto linearSolver = std::make_unique<SlamLinearSolver>();
      linearSolver->setBlockOrdering(false);

      // Change to Levenburg Marquit later
      OptimizationAlgorithmGaussNewton* solver =
          new OptimizationAlgorithmGaussNewton(
              std::make_unique<SlamBlockSolver>(std::move(linearSolver)));
      
      if(!optimizer_){
        optimizer_ = std::make_unique<SparseOptimizer>();
      }
      optimizer_->setAlgorithm(solver);

      // All vectors and maps start empty
      // (optional, clear() is safe to call)
      platformVertices_.clear();
      processModelEdges_.clear();
      landmarkIdMap_.clear();
      componentsReady_ = true;
    }
    // add Initial edges
  }



  void SlamSystem::stop(){
    // % Run the optimizer


    // % If we are fixing past vehicle states (Q3) then handle
    // % unfixing for the final optimization pass
    optimize(200);

    //if (fixOlderPlatformVertices_ == true){
      for (const auto& vertex : optimizer_->vertices()) {
        g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(vertex.second);
        v->setFixed(false);
      }
      optimize(50);
    //}

  }



  int SlamSystem::optimize(int maximumNumberOfOptimizationSteps){
    optimizer_->initializeOptimization();
    int numIterations = optimizer_->optimize(maximumNumberOfOptimizationSteps);
    if(verbose_){
      std::cerr << "Final chi2: " << optimizer_->activeChi2() << std::endl;
      std::cerr << "Num Iterations: " << numIterations << std::endl;
    }
    return numIterations;
    // Add performance data?
  }



  
  SparseOptimizer& SlamSystem::optimizer(){
    return *optimizer_;
  }



  
  void SlamSystem::setMaxObservationsPerLandmark(int maxObservationsPerLandmark){
    maxObservationsPerLandmark_ = maxObservationsPerLandmark;
  }


  
  void SlamSystem::setFixOlderPlatformVertices(double unfixedTimeWindow){
    unfixedTimeWindow_ = unfixedTimeWindow;
  }


  
  void SlamSystem::platformEstimate(SE2& x, Matrix2d& P){
    optimize(20);
    SparseBlockMatrix<MatrixX> spinv;
    optimizer_->computeMarginals(spinv, currentPlatformVertex_);
    P = spinv.block(0, 0)->topLeftCorner<2,2>();
    x = currentPlatformVertex_->estimate();
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


  
  void SlamSystem::processEvents(EventPtrVector& events){
    for (const auto& event : events) {
      processEvent(*event);
    }
  }



  void SlamSystem::setVerbose(bool verbose){
    verbose_ = verbose;
  }


  /**
   * @brief process a event
   * @param events event to process
   */
  void SlamSystem::processEvent(Event& event){
    double dT = event.time - currentTime_;
    if(initialized_){
      // TODO Change this to match minDT
      if(dT < 1e-4){
        handleNoPrediction();
      }
      else{
        handlePredictForwards(dT);
        currentTime_ = event.time;
        stepNumber_ +=1;
      }
      switch(event.type()){
        case Event::EventType::HeartBeat:
          break;
        case Event::EventType::LandmarkObservations:
          handleSLAMObservationEvent(static_cast<LandmarkObservationsEvent&>(event));
          break;
        case Event::EventType::Odometry:
          handleUpdateOdometryEvent(static_cast<OdometryEvent&>(event));
          break;
        case Event::EventType::Initialization:
          handleInitializationEvent(static_cast<InitializationEvent&>(event));
          break;
        default:
          ignoreUnknownEventType();
          break;
      }
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
    currentPlatformVertex_->setId(platformVertexId_+1);
    currentPlatformVertex_->setEstimate(newX);
    optimizer_->addVertex(currentPlatformVertex_);
    platformVertices_.emplace_back(currentPlatformVertex_);

    EdgeSE2* odometry = new EdgeSE2;
    odometry->vertices()[0] = optimizer_->vertex(platformVertexId_);
    odometry->vertices()[1] = currentPlatformVertex_;
    odometry->setMeasurement(lastpredX.inverse() * newX);
    odometry->setInformation(sigmaU_*dT);
    optimizer_->addEdge(odometry);

    processModelEdges_.emplace_back(odometry);
    numProcessModelEdges_ += 1;
    platformVertexId_ += 1;

    // I'll skip fix older platform vertex for now.
  }

  void SlamSystem::handleNoPrediction(){}

  void SlamSystem::handleInitializationEvent(InitializationEvent event){
    currentPlatformVertex_ = new VertexSE2;
    currentPlatformVertex_->setId(platformVertexId_+1);
    currentPlatformVertex_->setEstimate(event.pose);
    optimizer_->addVertex(currentPlatformVertex_);
    platformVertices_.emplace_back(currentPlatformVertex_);

    // TODO replace with initialization prior
    currentPlatformVertex_->setFixed(true);

    u_ = event.velocity;
    sigmaU_ = event.sigmaU;
    platformVertexId_ += 1;
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
    SE2 curvtxEst = currentPlatformVertex_->estimate();
    for(const auto& lmObs : event.landmarkObservations){
      VertexPointXY* lmVertex;
      bool vtxCreated = createOrGetLandmark(lmObs.landmark_id, lmVertex);
      if(vtxCreated){
        lmVertex->setEstimate(lmObs.value + curvtxEst.translation());  // Initial guess
      }

      EdgeSE2PointXY* landmarkObservation = new EdgeSE2PointXY;
      landmarkObservation->vertices()[0] = currentPlatformVertex_;
      landmarkObservation->vertices()[1] = lmVertex;
      landmarkObservation->setMeasurement(lmObs.value);
      landmarkObservation->setInformation(lmObs.covariance.inverse());
      //landmarkObservation->setParameterId(0, sensorOffset->id());
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
        lmVertex->setId(id);
        optimizer_->addVertex(lmVertex);
        landmarkVertices_.push_back(lmVertex);

        landmarkIdMap_[id] = landmarkVertices_.size() - 1;  // Map landmark id to index
        return true;
    }
  }
  // handlenoUpdate()
  // handleInitializationEvent(event)
  
  // ... all other observations ,,,
  

}  // namespace tutorial
}  // namespace g2o
