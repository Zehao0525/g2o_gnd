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

      verbose_ = j.value("verbose", false);
      if(verbose_){std::cout<<"- SlamSystem Created, verbose_ = true."<<std::endl;}
      if(verbose_){std::cout<<"- Reading all other parameters."<<std::endl;}
      optPeriod_ = j.value("optimization_period", 100);
      if(verbose_){std::cout<<"- optPeriod_ = " << optPeriod_ <<std::endl;}
      auto offset = j.value("sensor_offset", std::vector<double>{0.0, 0.0, 0.0});
      if (offset.size() != 3) {
          throw std::runtime_error("sensor_offset must be size 3");
      }
      if(verbose_){std::cout<<"- sensorOffset_ set" <<std::endl;}
      SE2 sensorOffsetTransf(offset[0], offset[1], offset[2]);
      sensorOffset_ = new ParameterSE2Offset();
      sensorOffset_->setOffset(sensorOffsetTransf);
      sensorOffset_->setId(0);

      optimizationAlg_ = j.value("optimization_algorithm", "GaussNewton");
      if(verbose_){std::cout<<"- optimizationAlg_ = " << optimizationAlg_ <<std::endl;}


      if(verbose_){std::cout<<"- creating optimizer ..." <<std::endl;}
      //optimizer_ = std::make_unique<SparseOptimizer>();

      optimizer_->addParameter(sensorOffset_);
      setupOptimizer();
      landmarkIdMap_.clear();

      optCountProcess_ = j["optimize_count"].value("process", 10);
      optCountStop_ = j["optimize_count"].value("stop", 10);
      optCountStopFix_ = j["optimize_count"].value("stop_fixed", 10);
  }
  SlamSystem::~SlamSystem(){}


  void SlamSystem::start(){
    if(verbose_){std::cout << " - SlamSystem start() ... " << std::endl;}

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


    std::vector<double> chi2_se2;
    std::vector<double> chi2_se2_xy;
    std::vector<double> chi2_rb;

    for (const auto& edge : optimizer_->edges()) {
        //if (auto e = dynamic_cast<EdgeSE2*>(edge)) {
        if (auto e = dynamic_cast<EdgeSE2*>(edge)) {
            chi2_se2.push_back(e->chi2());
        }
        if (auto e = dynamic_cast<EdgeVelocitySE2*>(edge)) {
            chi2_se2.push_back(e->chi2());
        }
        else if (auto e = dynamic_cast<EdgeSE2PointXY*>(edge)) {
            chi2_se2_xy.push_back(e->chi2());
        }
        else if (auto e = dynamic_cast<EdgeRangeBearing*>(edge)) {
            chi2_rb.push_back(e->chi2());
        }
    }

    auto computeStats = [](const std::vector<double>& chi2_values, const std::string& label) {
        if (chi2_values.empty()) {
            std::cout << label << ": No edges found." << std::endl;
            return;
        }

        double sum = 0.0;
        for (double c : chi2_values) {
            sum += c;
        }
        double mean = sum / chi2_values.size();

        std::vector<double> sorted = chi2_values;
        std::sort(sorted.begin(), sorted.end());
        double median = sorted[sorted.size() / 2];
        if (sorted.size() % 2 == 0) {
            median = 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
        }

        std::cout << label << " stats:" << std::endl;
        std::cout << "  Count:  " << chi2_values.size() << std::endl;
        std::cout << "  Mean:   " << mean << std::endl;
        std::cout << "  Median: " << median << std::endl;
    };

    computeStats(chi2_se2, "EdgeSE2");
    computeStats(chi2_se2_xy, "EdgeSE2PointXY");
    computeStats(chi2_rb, "EdgeRangeBearing");
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

    if(verbose_){std::cout << " - Optimizing before data extraction ..." << std::endl;}
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
    if(verbose_){std::cout << " - processing event with type:" << static_cast<int>(event.type()) << std::endl;}
    switch(event.type()){
      case Event::EventType::HeartBeat:
        break;
      case Event::EventType::LandmarkObservations:
        handleSLAMObservationEvent(static_cast<LandmarkObservationsEvent&>(event));
        break;
      case Event::EventType::LMRangeBearingObservations:
        handleRangeBearingObservationEvent(static_cast<LMRangeBearingObservationsEvent&>(event));
        break;
      case Event::EventType::GPSObservation:
        handleGPSObservationEvent(static_cast<GPSObservationEvent&>(event));
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
    EdgeVelocitySE2* odometry = new EdgeVelocitySE2(dT);
    odometry->setVertex(0, platformVertices_[platformVertices_.size() - 2]);
    odometry->setVertex(1, currentPlatformVertex_);
    if(verbose_){std::cout << " - Vertex set, setting measurments ..." << std::endl;}
    if(verbose_){std::cout << (lastpredX.inverse() * newX).toVector() << std::endl;}
    odometry->setMeasurement(u_);
    assert(odometry->information().rows() == 3);
    if(verbose_){std::cout << " - measurements set, setting information ..." << std::endl;}
    if(verbose_){std::cout << (sigmaU_) << std::endl;}
    odometry->setInformation((sigmaU_).inverse());
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
    
    //Matrix2d P;
    if(verbose_){std::cout << " - Estimating platform position ..." << std::endl;}
    SE2 curvtxEst = currentPlatformVertex_->estimate();

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
        lmVertex->setEstimate(curvtxEst * lmObs.value );  // Initial guess
        if(verbose_){std::cout<< lmVertex->estimate() <<std::endl;}
      }

      if(verbose_){std::cout << " - Creating Observation Edge ..." << std::endl;}
      EdgeSE2PointXY* landmarkObservation = new EdgeSE2PointXY;
      //landmarkObservation->resize(2);
      landmarkObservation->setVertex(0,currentPlatformVertex_);
      if(verbose_){std::cout<< (dynamic_cast<VertexSE2*>(landmarkObservation->vertices()[0])->estimate()).toVector() <<std::endl;}
      landmarkObservation->setVertex(1, lmVertex);
      if(verbose_){std::cout<< (dynamic_cast<VertexPointXY*>(landmarkObservation->vertices()[1])->estimate()) <<std::endl;}

      if(verbose_){std::cout << " = Setting measurments" << std::endl;}
      landmarkObservation->setMeasurement(lmObs.value);
      if(verbose_){std::cout<< landmarkObservation->measurement() <<std::endl;}

      if(verbose_){std::cout << " = Setting information" << std::endl;}

      landmarkObservation->setInformation(lmObs.covariance.inverse());
      if(verbose_){std::cout<< landmarkObservation->information() <<std::endl;}

      if(verbose_){std::cout << " - Setting parameter id" << std::endl;}
      if(verbose_){std::cout << sensorOffset_->id() << std::endl;}
      landmarkObservation->setParameterId(0, sensorOffset_->id());
      if(verbose_){std::cout << " - Adding edge to factor graph..." << std::endl;}

      //checkTypeRegistration();
      //landmarkObservation->linearizeOplus();

      optimizer_->addEdge(landmarkObservation);
    }
    if(verbose_){std::cout << " - SlamSystem handleSLAMObservationEvent end ..." << std::endl;}

  }



  void SlamSystem::handleRangeBearingObservationEvent(LMRangeBearingObservationsEvent event){
    if(verbose_){std::cout << " - SlamSystem handleSLAMObservationEvent start ..." << std::endl;}
    
    //Matrix2d P;
    if(verbose_){std::cout << " - Estimating platform position ..." << std::endl;}
    Vector3d curvtxEst = (currentPlatformVertex_->estimate()).toVector();

    for(const auto& lmObs : event.landmarkObservations){
      assert(lmObs.value.size() == 2);
      assert(lmObs.covariance.rows() == 2 && lmObs.covariance.cols() == 2);
      if(verbose_){std::cout << " - Processing Range bearing observation" << std::endl;}
      VertexPointXY* lmVertex;
      if(verbose_){std::cout << " - Creating/Getting Landmark Vertex..." << std::endl;}
      bool vtxCreated = createOrGetLandmark(lmObs.landmark_id, lmVertex);
      if(verbose_){std::cout << lmVertex << ",  id = " << lmVertex->id()  << std::endl;}
      if(vtxCreated){
        if(verbose_){std::cout<<" = New vertex created, setting estimates"<<std::endl;}
        double trueBearing = lmObs.value[1] + curvtxEst[2];
        Vector2d disp = Vector2d(lmObs.value[0] * cos(trueBearing) + curvtxEst[0], lmObs.value[0] * sin(trueBearing) + curvtxEst[1]);

        lmVertex->setEstimate(disp);  // Initial guess
        if(verbose_){std::cout<< lmVertex->estimate() <<std::endl;}
      }

      if(verbose_){std::cout << " - Creating Observation Edge ..." << std::endl;}
      
      EdgeRangeBearing* landmarkObservation = new EdgeRangeBearing;
      //landmarkObservation->resize(2);
      landmarkObservation->setVertex(0,currentPlatformVertex_);
      if(verbose_){std::cout<< (dynamic_cast<VertexSE2*>(landmarkObservation->vertices()[0])->estimate()).toVector() <<std::endl;}
      landmarkObservation->setVertex(1, lmVertex);
      if(verbose_){std::cout<< (dynamic_cast<VertexPointXY*>(landmarkObservation->vertices()[1])->estimate()) <<std::endl;}

      if(verbose_){std::cout << " = Setting measurments" << std::endl;}
      landmarkObservation->setMeasurement(lmObs.value);
      if(verbose_){std::cout<< landmarkObservation->measurement() <<std::endl;}

      if(verbose_){std::cout << " = Setting information" << std::endl;}

      landmarkObservation->setInformation(lmObs.covariance.inverse());
      if(verbose_){std::cout<< landmarkObservation->information() <<std::endl;}

      if(verbose_){std::cout << " - Setting parameter id" << std::endl;}
      if(verbose_){std::cout << sensorOffset_->id() << std::endl;}
      landmarkObservation->setParameterId(0, sensorOffset_->id());
      if(verbose_){std::cout << " - Adding edge to factor graph..." << std::endl;}

      //checkTypeRegistration();
      //landmarkObservation->linearizeOplus();

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
