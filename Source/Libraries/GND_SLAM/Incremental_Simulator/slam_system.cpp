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


  void SlamSystem::checkTypeRegistration() {
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

using namespace Eigen;

//thread_local std::unique_ptr<SparseOptimizer> SlamSystem::optimizer_;
std::unique_ptr<SparseOptimizer> SlamSystem::optimizer_;
//
  SlamSystem::SlamSystem():verbose_(true){}
  SlamSystem::~SlamSystem(){}


  void SlamSystem::start(){
    if(verbose_){std::cout << " - SlamSystem start() ... " << std::endl;}
    currentTime_ = 0;
    stepNumber_ = 0;
    initialized_ = false;

    // % Set up the event handlers

    // The SLAM system has been started before, a lot of the initialisation work as been done
    if(!componentsReady_){
      vertexId_ = -1;
      numProcessModelEdges_ = 0;
      currentPlatformVertex_ = nullptr;

      // allocating the optimizer
      auto linearSolver = std::make_unique<LinearSolverEigen<BlockSolver<BlockSolverTraits<-1, -1> >::PoseMatrixType>>();
      linearSolver->setBlockOrdering(false);

      // Change to Levenburg Marquit later
      OptimizationAlgorithmLevenberg* solver =
          new OptimizationAlgorithmLevenberg(
              std::make_unique<BlockSolver<BlockSolverTraits<-1, -1> >>(std::move(linearSolver)));
      
      if(!optimizer_){
        if(verbose_){std::cout << " - No optimizers, Creating optmizer" << std::endl;}
        optimizer_ = std::make_unique<SparseOptimizer>();
      }
      else{
        if(verbose_){std::cout << " - Existing optimizer, clearing optimizer" << std::endl;}
        optimizer_->clear();
      }
      optimizer_->setAlgorithm(solver);


      // All vectors and maps start empty
      // (optional, clear() is safe to call)
      platformVertices_.clear();
      processModelEdges_.clear();
      landmarkIdMap_.clear();
      componentsReady_ = true;


      // Setup sensor offset
      SE2 sensorOffsetTransf(0.0, 0.0, 0.0);
      sensorOffset_ = new ParameterSE2Offset();
      sensorOffset_->setOffset(sensorOffsetTransf);
      sensorOffset_->setId(0);
      optimizer_->addParameter(sensorOffset_);
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
    if(verbose_){optimizer_->setVerbose(true);}
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
    if(verbose_){std::cout << " - SlamSystem platformEstimate start ..." << std::endl;}
    if(verbose_){std::cerr << "Graph verification success - " << optimizer_->verifyInformationMatrices(true) << ", checking for null pointers in edges" << std::endl;}
    if(verbose_){
      if (!optimizer_->parameter(0)) {
        std::cerr << "Sensor offset parameter not registered in optimizer!" << std::endl;
        throw std::runtime_error("Missing parameter!");
      }
      else{
        std::cerr << optimizer_->parameter(0) << std::endl;
      }
    }
    for (const auto& e : optimizer_->edges()) {
      for (size_t i = 0; i < e->vertices().size(); ++i) {
          if (!e->vertex(i)) {
              std::cerr << "Edge has null vertex at position " << i << std::endl;
              throw std::runtime_error("Null vertex in edge");
          }
      }
    }
    std::cout << "Number of vertices: " << optimizer_->vertices().size() << std::endl;
    std::cout << "Number of edges: " << optimizer_->edges().size() << std::endl;


    optimize(20);
    SparseBlockMatrix<MatrixX> spinv;
    if(verbose_){std::cout << " - Optimizer Computing Marginals ..." << std::endl;}


    std::cout << " - currentPlatformVertex_->fixed() " << currentPlatformVertex_->fixed() << std::endl;
    if (currentPlatformVertex_->fixed()) {
      if (verbose_) {
        std::cout << " - Current vertex is fixed. Skipping marginal computation." << std::endl;
      }
      x = currentPlatformVertex_->estimate();
      P.setZero();
      return;
    }

    int idx = currentPlatformVertex_->hessianIndex();

    if(verbose_){std::cout << " - Compute Marginals..." << std::endl;}
    bool success = optimizer_->computeMarginals(spinv, currentPlatformVertex_);



    if (!success) {
      if (verbose_) {
        std::cout << " - Optimization failed" << std::endl;
      }
      x = currentPlatformVertex_->estimate();
      P.setZero();
      return;
    }

    if(verbose_){std::cout << " - currentPlatformVertex_->hessianIndex():" << idx << ", Compute margianl success - " << success << std::endl;}
    if(verbose_){std::cout << " - Assigning to x and P ..." << std::endl;}
    const auto block = spinv.block(idx, idx);
    if(block){
        P = block->topLeftCorner<2,2>();
    } else {
        if(verbose_){std::cout << "   - WARNING: Marginal block is null (probably vertex is fixed), setting P to zero." << std::endl;}
        P.setZero();
    }
    x = currentPlatformVertex_->estimate();
    if(verbose_){std::cout << " - SlamSystem platformEstimate end ..." << std::endl;}
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
    if(verbose_){std::cout << " - SlamSystem processEvents start ..." << std::endl;}
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
    
    // TODO Change this to match minDT
    if(dT < 1e-4){
      handleNoPrediction();
    }
    else{
      handlePredictForwards(dT);
      currentTime_ = event.time;
      stepNumber_ +=1;
    }
    if(verbose_){std::cout << " - processing event with type:" << static_cast<int>(event.type()) << std::endl;}
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

  /**
   * @brief process a event
   * At this moment we won't have this step. mybe in the future.
   * @param eventType type of event
   * @param eventHandler handler that handles that type of event
   */
  //void registerEventHandler(EventType eventType, EventHandler eventHandler);


  void SlamSystem::ignoreUnknownEventType(){}


  void SlamSystem::handlePredictForwards(double dT){
    if(verbose_){std::cout << " - SlamSystem handlePredictForwards start ..." << std::endl;}
    SE2 lastpredX = currentPlatformVertex_->estimate();
    SE2 newX = lastpredX * (u_*dT);
    
    if(verbose_){std::cout << " - Creating new vertex ..." << std::endl;}
    currentPlatformVertex_ = new VertexSE2;
    currentPlatformVertex_->setEstimate(newX);
    currentPlatformVertex_->setId(++vertexId_);
    if(verbose_){std::cout << " - Adding vertex to optimizer ..." << std::endl;}
    optimizer_->addVertex(currentPlatformVertex_);
    platformVertices_.emplace_back(currentPlatformVertex_);

    if(verbose_){std::cout << " - Creating new odometry edge ..." << std::endl;}
    EdgeSE2* odometry = new EdgeSE2;
    odometry->setVertex(0, platformVertices_[platformVertices_.size() - 2]);
    odometry->setVertex(1, currentPlatformVertex_);
    if(verbose_){std::cout << " - Vertex set, setting measurments ..." << std::endl;}
    if(verbose_){std::cout << (lastpredX.inverse() * newX).toVector() << std::endl;}
    odometry->setMeasurement(lastpredX.inverse() * newX);
    assert(odometry->information().rows() == 3);
    if(verbose_){std::cout << " - measurements set, setting information ..." << std::endl;}
    if(verbose_){std::cout << (sigmaU_*dT) << std::endl;}
    odometry->setInformation((sigmaU_*dT).inverse());
    if(verbose_){std::cout << " - Adding edge to optimizer ..." << std::endl;}
    optimizer_->addEdge(odometry);

    processModelEdges_.emplace_back(odometry);
    numProcessModelEdges_ += 1;

    // I'll skip fix older platform vertex for now.
  }

  void SlamSystem::handleNoPrediction(){}

  void SlamSystem::handleInitializationEvent(InitializationEvent event){
    if(verbose_){std::cout << " - SlamSystem handleInitializationEvent start ..." << std::endl;}
    if(verbose_){std::cout << " - Creaing vertex ..." << std::endl;}
    currentPlatformVertex_ = new VertexSE2;
    currentPlatformVertex_->setId(++vertexId_);
    currentPlatformVertex_->setEstimate(event.pose);
    if(verbose_){std::cout << " - Adding vertex to optimizer ..." << std::endl;}
    optimizer_->addVertex(currentPlatformVertex_);
    platformVertices_.emplace_back(currentPlatformVertex_);

    // TODO replace with initialization prior
    if(verbose_){std::cout << " - Fixing initial vertex ..." << std::endl;}
    currentPlatformVertex_->setFixed(true);

    if(verbose_){std::cout << " - Setting controls parameters ..." << std::endl;}
    u_ = event.velocity;
    sigmaU_ = event.sigmaU;
    initialized_ = true;
    if(verbose_){std::cout << " - SlamSystem handleInitializationEvent end ..." << std::endl;}
  }

  void SlamSystem::handleUpdateOdometryEvent(OdometryEvent event){
    if(verbose_){std::cout << " - SlamSystem handleUpdateOdometryEvent start ..." << std::endl;}
    u_ = event.value;
    sigmaU_ = event.covariance;
  }

  /**
   * @brief event handler for landmark observation events
   * @param event
   */
  void SlamSystem::handleSLAMObservationEvent(LandmarkObservationsEvent event){
    if(verbose_){std::cout << " - SlamSystem handleSLAMObservationEvent start ..." << std::endl;}
    SE2 curvtxEst;
    Matrix2d P;
    if(verbose_){std::cout << " - Estimating platform position ..." << std::endl;}
    SlamSystem::platformEstimate(curvtxEst, P);
    for(const auto& lmObs : event.landmarkObservations){
      assert(lmObs.value.size() == 2);
      assert(lmObs.covariance.rows() == 2 && lmObs.covariance.cols() == 2);
      if(verbose_){std::cout << " - Processing LM observation" << std::endl;}
      VertexPointXY* lmVertex;
      if(verbose_){std::cout << " - Creating/Getting Landmark Vertex..." << std::endl;}
      bool vtxCreated = createOrGetLandmark(lmObs.landmark_id, lmVertex);
      if(verbose_){std::cout << lmVertex << ",  id = " << lmVertex->id()  << std::endl;}
      if(vtxCreated){
        if(verbose_){std::cout<<" = New vertex created, setting estimates"<<std::endl;}
        lmVertex->setEstimate(lmObs.value + curvtxEst.translation());  // Initial guess
        if(verbose_){std::cout<< lmVertex->estimate() <<std::endl;}
      }

      if(verbose_){std::cout << " - Creating Observation Edge ..." << std::endl;}
      EdgeSE2PointXY* landmarkObservation = new EdgeSE2PointXY;
      //landmarkObservation->resize(2);
      landmarkObservation->setVertex(0,currentPlatformVertex_);
      landmarkObservation->setVertex(1, lmVertex);

      if(verbose_){std::cout << " = Setting measurments" << std::endl;}
      if(verbose_){std::cout << lmObs.value << std::endl;}
      landmarkObservation->setMeasurement(lmObs.value);
      if(verbose_){std::cout<< landmarkObservation->estimate() <<std::endl;}

      if(verbose_){std::cout << " = Setting information" << std::endl;}

      landmarkObservation->setInformation(lmObs.covariance.inverse());
      if(verbose_){std::cout<< landmarkObservation->information() <<std::endl;}

      if(verbose_){std::cout << " - Setting parameter id" << std::endl;}
      if(verbose_){std::cout << sensorOffset_->id() << std::endl;}
      landmarkObservation->setParameterId(0, sensorOffset_->id());
      if(verbose_){std::cout << " - Adding edge to factor graph..." << std::endl;}

      checkTypeRegistration();
      landmarkObservation->linearizeOplus();

      optimizer_->addEdge(landmarkObservation);
    }
    if(verbose_){std::cout << " - SlamSystem handleSLAMObservationEvent end ..." << std::endl;}

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
  // handlenoUpdate()
  // handleInitializationEvent(event)
  
  // ... all other observations ,,,
  

}  // namespace tutorial
}  // namespace g2o
