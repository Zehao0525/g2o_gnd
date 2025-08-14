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


static bool isValidInformationMatrix(const Eigen::Matrix<double,6,6>& info) {
  auto eig = info.selfadjointView<Eigen::Upper>().eigenvalues();
  if (!info.allFinite() || (eig.array() <= 0).any() ||
    eig.maxCoeff() > 1e6){
      return false;
    }
  return true;
}

using namespace Eigen;

using VertexContainer = g2o::OptimizableGraph::VertexContainer;



  FileSlamSystem::FileSlamSystem(int id, const std::string& filename)
    :SlamSystemBase<VertexSE3, EdgeSE3>(filename), robotId_(id), gndActive_(false), haveUninitializedObs_(false){

  }

  FileSlamSystem::~FileSlamSystem()=default;


  void FileSlamSystem::start(){
    graphChanged_ = false;

    auto* offset = new g2o::ParameterSE3Offset; // identity offset	
    offset->setOffset(Isometry3d::Identity());	
    offset->setId(0); // ID = 0		
    optimizer_->addParameter(offset); // <-- only once
  }


  void FileSlamSystem::stop(){
    optimize(optCountStop_);

    //if (fixOlderPlatformVertices_ == true){
    // TODO We are doing id = 0 for now.
    for (const auto& vertex : optimizer_->vertices()) {
      g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(vertex.second);
      if(VertexIdMap_[v->id()] > 0){
        v->setFixed(false);
      }
    }
    optimize(optCountStopFix_);

    for (size_t i = 0; i < observations_.size(); ++i) {
      const auto& obs = observations_[i];
      std::cout << "Observation[" << i << "] (vertex " << obs.observedVertexId
                << ") initialized? " << (obs.initialized ? "Yes" : "No") << std::endl;
    }

    std::cout << "Number of intra: " << intraRobotCount_ << std::endl;
  }

  void FileSlamSystem::processEvent(Event& event){
    graphChanged_ = true;
    switch(event.type()){
      case Event::EventType::FileObservation:
        handleObservationEvent(static_cast<FileObsEvent&>(event));
        break;
      case Event::EventType::FileIntraObservation:
        handleIntraObservationEvent(static_cast<FileIntraObsEvent&>(event));
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
    if(verbose_){std::cout << " - SlamSystem handleObservationEvent start for robot: " << robotId_ << std::endl;}

    if(verbose_){std::cout << " - overvation from: " << event.vtxIdFrom << " To: " << event.vtxIdTo << std::endl;}
    // Initialize Vertex
    // place the id into the vertex id map

    // We add a vertex representing the pose frame transform between this robot and other bots. 
    if (relativeTransforms_.find(event.idTo) == relativeTransforms_.end()) {
      VertexSE3* v = new VertexSE3();
      // We assume that there's no more than 20000 vertecies in the map
      v->setId(20000 + event.idTo);
      v->setEstimate(Isometry3d::Identity());
      relativeTransformsFrom_[event.idTo] = v;
    }

    VertexSE3* observedVtx;
    EdgeSE3* observationPrior;
    VertexSE3* v0 = platformVertices_[VertexIdMap_[event.vtxIdFrom]];
    if (externalVertices_.find(event.vtxIdTo) == externalVertices_.end()) {
      observedVtx = new VertexSE3;
      observedVtx->setEstimate(v0->estimate() * event.value);
      observedVtx->setId(event.vtxIdTo);

      // Place in mapping
      externalVertices_[event.vtxIdTo] = observedVtx;
      bool ok = optimizer_->addVertex(observedVtx);

       // Initialize Observed Prior Edge (uninitialized)
      if (!optimizer_->parameter(0)) {
        ParameterSE3Offset* offset = new ParameterSE3Offset();
        offset->setOffset(Eigen::Isometry3d::Identity());  // Or a valid transform
        offset->setId(0);
        optimizer_->addParameter(offset);
      }
      observationPrior = new EdgeSE3Prior;
      observationPrior->setVertex(0,observedVtx);
      observationPrior->setMeasurement(v0->estimate() * event.value);
      observationPrior->setInformation(Eigen::Matrix<double,6,6>::Identity());
      observationPrior->setParameterId(0, 0);
      bool ok2 = optimizer_->addEdge(observationPrior);
      // std::cout << "Registered types:\n";
      // g2o::Factory::instance()->printRegisteredTypes(std::cout);
      if(verbose_){std::cout << " - Adding this prior: " << ok << ok2 << std::endl;}
      //assert(false);
      // Attach a GND kernel
      auto rk = new g2o::ToggelableGNDKernel(2.0, 6, 1e-3, 2.0*2.0, &gndActive_);
      observationPrior->setRobustKernel(rk);

      //Place in Mapping
      externalVerticesPrior_[event.vtxIdTo] = observationPrior;
    }
    else{
      observedVtx = externalVertices_[event.vtxIdTo];
      observationPrior = externalVerticesPrior_[event.vtxIdTo];
    }
    assert(v0 && "platform vertex lookup failed for vtxIdFrom");
    assert(observedVtx && "external vertex lookup failed for vtxIdTo");
    

    // Initialize Observation Edge
    EdgeSE3* observation = new EdgeSE3();
    observation->setVertex(0,v0);
    observation->setVertex(1,observedVtx);
    observation->setMeasurement(event.value);
    observation->setInformation(event.information);


    observations_.emplace_back(robotId_, event.robotIdTo, event.vtxIdFrom, event.vtxIdTo,
            observationPrior, observation, observedVtx);
    haveUninitializedObs_ = true;

  }


  void FileSlamSystem::handleIntraObservationEvent(FileIntraObsEvent event){
    //TODO implement
    intraRobotCount_+=1;
    if(verbose_){std::cout << " - robot " << robotId_<< "SlamSystem handleIntraObservationEvent start ..." << std::endl;}
    // lookup vertices
    auto* vFrom = platformVertices_[ VertexIdMap_[event.vtxIdFrom] ];
    auto* vTo   = platformVertices_[ VertexIdMap_[event.vtxIdTo]   ];

    // null checks
    if (!vFrom || !vTo) {
        std::cerr << "Error in handleIntraObservationEvent: "
                  << (vFrom ? "" : "vtxIdFrom lookup returned null ")
                  << (vTo   ? "" : "vtxIdTo   lookup returned null ")
                  << std::endl;
        throw std::runtime_error("handleIntraObservationEvent: vertex pointer is null");
    }

    EdgeSE3* observation = new EdgeSE3();
    observation->setVertex(0, vFrom);
    observation->setVertex(1, vTo);

    if(verbose_){std::cout << " - Setting measurments ..." << std::endl;}
    observation->setMeasurement(event.value);
    if(verbose_){std::cout << " - measurements set, setting information ..." << std::endl;}
    observation->setInformation((event.information));
    if(verbose_){std::cout << " - Adding edge to optimizer ..." << std::endl;}
    optimizer_->addEdge(observation);

    if(verbose_){
      // Optional: Set formatting for better readability
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

  }


  FileSlamSystem::ObsSyncMessage FileSlamSystem::broadcastObsSyncMessage() const{
    // Pre-allocate the vector for efficiency
    std::vector<FileSlamSystem::ObsSyncRequest> syncReqs;
    syncReqs.reserve(observations_.size());

    // Convert each Observation into an ObsSyncRequest
    for (const auto& obs : observations_) {
      // 1) Use the existing ID-only constructor
      FileSlamSystem::ObsSyncRequest req(
        /* selfId */        obs.observerRobotId,
        /* robotId */       obs.observedRobotId,
        /* selfVtxId */     obs.observerVertexId,
        /* vertexId */      obs.observedVertexId
      );

      // std::cout << "id:" << robotId_ << "(" << obs.observerRobotId <<") -> (" << obs.observedRobotId 
      //     << "), obs.observedVertexId " << obs.observedVertexId << std::endl;

      // 2) Fill in the actual pose + information from the edge
      // req.observedVtxLocation    = obs.observationEdge->measurement();
      // req.observedVtxInformation = obs.observationEdge->information();

      syncReqs.push_back(std::move(req));
    }

    // Build & return the final message; mark it as outgoing
    return FileSlamSystem::ObsSyncMessage(
      /* sender    */ robotId_,
      /* outGoing  */ true,
      /* syncReqs  */ std::move(syncReqs)
    );
  }

  // recieveObsSyncMessage
  FileSlamSystem::ObsSyncMessage FileSlamSystem::handleObservationSyncRequest(FileSlamSystem::ObsSyncMessage& request){
    // Step 1: find all ObsSyncReqs with observedRobotId == robotId_
    // if there are requests we ned to deal with, do:
    //   Stwp 2: call "optimize(10)"
    //   Step 3: Find the MAP estimates and Covriance of the vertex with "observerVertexId" for each by:
    //      Step 3.1: Extract estimates by qurying the vertex. It is the jth vertex of platformVertices_, where j = VertexIdMap_[observerVertexId]
    //      Step 3.2: Solve for the marginals of all vtxs. Fill in the Matrix<double,6,6> "observedVtxCovariance" for each request.
    //   Step 4: Create a ObsSyncMessage for these requests, set outGoing = false, return this message

      // Step 1: collect requests addressed to this robot
    std::vector<FileSlamSystem::ObsSyncRequest> localRequests;
    localRequests.reserve(request.syncRequests.size());
    std::vector<g2o::OptimizableGraph::Vertex*> verticesToMarginalize;
    verticesToMarginalize.reserve(localRequests.size());
    for (auto const& req : request.syncRequests) {
      if (req.observedRobotId != robotId_){
        continue;
      } 
      auto it = VertexIdMap_.find(req.observedVertexId);
      if (it == VertexIdMap_.end()) {
        if (verbose_) {
          std::cerr << "  [WARN] no mapping for observerVertexId="
                    << req.observedVertexId << "\n";
        }
        continue;
      }
      int idx = it->second;
      if (idx < 0 || idx >= int(platformVertices_.size())) {
        std::cerr << "  [WARN] idx " << idx 
                  << " out of range for platformVertices_\n";
        continue;
      }
      auto* v = platformVertices_[idx];
      //std::cerr << "Hessian Index" << v->hessianIndex() << std::endl;
      if (!v) {
        std::cerr << "  [WARN] platformVertices_[" << idx 
                  << "] is null\n";
        continue;
      }
      // both req and vertex are good—keep them
      localRequests.push_back(req);
      verticesToMarginalize.push_back(v);
    }

    // 2) if nothing to do, early out immediately
    if (localRequests.empty()) {
      if (verbose_) {
        std::cout << "[REQ] no observations to sync for robot "
                  << robotId_ << " – skipping optimize+marginals\n";
      }
      return ObsSyncMessage(robotId_, /*outGoing=*/false, {});
    }

    // Step 2: optimize our graph so that we have up-to-date estimates
    if(graphChanged_){
      if(verbose_){std::cout << "Optimizing before marginalization:\n";}
      optimizer_->initializeOptimization();
      optimizer_->optimize(20);
      graphChanged_ = false;
    }

    // Step 3: assemble the list of vertex pointers for marginal covariances
    // WARNING: no fail proofs
    

    // compute marginals
    // for (auto& [id, v] : externalVertices_) {
    //   if (v) {
    //     v->setFixed(true);
    //   }
    // }
    if(verbose_){std::cout << "Marginalizing\n";}
    g2o::SparseBlockMatrix<Eigen::MatrixXd> margCov;
    bool margSuccess = optimizer_->computeMarginals(margCov, verticesToMarginalize);
    if(verbose_){std::cout << "Marginalization success: " << margSuccess << "\n";}
    if(!margSuccess){return ObsSyncMessage(robotId_, /*outGoing=*/false, {});}
    // for (auto& [id, v] : externalVertices_) {
    //   if (v) {
    //     v->setFixed(false);
    //   }
    // }
    // Step 4: fill in each ObsSyncRequest with measurement+information
    std::vector<FileSlamSystem::ObsSyncRequest> validResponses;
    validResponses.reserve(localRequests.size());
    for (size_t i = 0; i < localRequests.size(); ++i) {
      //if(verbose_){std::cout << "entering for loop \n";}
      auto& req = localRequests[i];
      auto* v = dynamic_cast<g2o::VertexSE3*>(verticesToMarginalize[i]);
      int vhIdx = v->hessianIndex();
      // 3.1: MAP estimate
      assert(v && "v to marginalize is nullptr");
      req.observedVtxLocation = v->estimate();

      // 3.2: covariance → information = inverse(covariance)
      assert(margCov.block(vhIdx, vhIdx) && "Marg block is nullptr");
      //if(verbose_){std::cout << "marg cov block extraction start\n";}
      const Eigen::Matrix<double,6,6>& cov = *margCov.block(vhIdx, vhIdx);
      Eigen::Matrix<double,6,6> info = cov.inverse();
      // **Validation**: only keep if the info‐matrix is well‐formed
      if (isValidInformationMatrix(info)) {
        req.observedVtxInformation = std::move(info);
        validResponses.push_back(std::move(req));
        
      } else {
        std::cerr << "Dropping sync for vertex " << req.observedVertexId
                  << " due to invalid information matrix\n";
                  if(verbose_){std::cout << "cov (6x6):\n" << cov << "\n";}
      }
    }
    if(verbose_){std::cout << "handleObervationSynRequest Complete\n";}

    // Step 5: return only the valid responses
    return FileSlamSystem::ObsSyncMessage(
      /* sourceId  */ request.sourceId,
      /* outGoing  */ false,
      /* syncReqs  */ std::move(validResponses)
    );
  }



  void FileSlamSystem::handleObservationSyncResponse(const FileSlamSystem::ObsSyncMessage& message) {
    // 1) Update all matching SE3Prior edges from the external cache
    if(verbose_){std::cout << "handleObervationSynResponse Start\n";}
    for (const auto& req : message.syncRequests) {
      auto it = externalVerticesPrior_.find(req.observedVertexId);
      if (it != externalVerticesPrior_.end()) {
        g2o::EdgeSE3Prior* priorEdge = it->second;
        // // overwrite the cached prior with the new measurement + information

        // // WARNING: This does not check information validity.it assumes the informatio nis valid.
        priorEdge->setMeasurement(req.observedVtxLocation);
        priorEdge->setInformation(req.observedVtxInformation / 4);
        //optimizer_->addEdge(priorEdge);
      }
    }

    if(verbose_){std::cout << "step 2 Start\n";}
    // 2) If we still have uninitialized observations, try to add them now
    if (haveUninitializedObs_) {
      bool foundAny = false;
      for (auto& obs : observations_) {
        if (!obs.initialized) {
          // see if we just got a syncResponse for this observedVertexId
          // (we assume syncRequests covered it above)
          auto wasUpdated = std::any_of(
            message.syncRequests.begin(), message.syncRequests.end(),
            [&](auto const& req) {
              return req.observedVertexId == obs.observedVertexId;
            }
          );
          if (wasUpdated) {

            assert(obs.observationVertex != nullptr);
            assert(obs.observationEdge != nullptr);
            assert(obs.observationPriorEdge != nullptr);
            assert(VertexIdMap_.find(obs.observationVertex->id()) == VertexIdMap_.end());
            assert(obs.observationVertex == obs.observationPriorEdge->vertices()[0]);
            assert(obs.observationVertex == obs.observationEdge->vertices()[1]);
            assert(obs.observationPriorEdge->information().rows() == 6); // or whatever dimension
            assert(obs.observationPriorEdge->information().determinant() > 0);


            auto* vertex = obs.observationVertex;
            auto* prior = obs.observationPriorEdge;

            // std::cout << "=== PRIOR EDGE DIAGNOSTICS ===\n";
            // std::cout << "Edge ptr: " << prior << "\n";
            // std::cout << "Edge vertices: " << prior->vertices().size() << "\n";
            // for (size_t i = 0; i < prior->vertices().size(); ++i) {
            //   std::cout << "  vertex[" << i << "] ptr: " << prior->vertex(i) << "\n";
            // }
            // std::cout << "optimizer has edge? " << (optimizer_->edges().count(prior) ? "yes" : "no") << "\n";
            // std::cout << "Vertex ID: " << vertex->id() << ", in optimizer? " 
            //           << (optimizer_->vertex(vertex->id()) ? "yes" : "NO") << "\n";

            // std::cout << "Prior vertex(0) ptr: " << prior->vertex(0) 
            //           << ", ID: " << (prior->vertex(0) ? prior->vertex(0)->id() : -1) << "\n";

            // std::cout << "Measurement matrix:\n" << prior->measurement().matrix() << "\n";
            // std::cout << "Information matrix:\n" << prior->information() << "\n";
            // std::cout << "Determinant: " << prior->information().determinant() << "\n";

            // Manual check for allVerticesValid()
            bool all_valid = true;
            for (int i = 0; i < prior->vertices().size(); ++i) {
                auto* v = prior->vertex(i);
                if (!v) {
                    std::cout << "Vertex " << i << " is nullptr\n";
                    all_valid = false;
                } else if (!optimizer_->vertex(v->id())) {
                    std::cout << "Vertex " << i << " (ID=" << v->id() << ") NOT in optimizer\n";
                    all_valid = false;
                } else {
                    std::cout << "Vertex " << i << " (ID=" << v->id() << ") OK\n";
                }
            }

            if (!all_valid) {
                std::cout << "→ allVerticesValid() would return FALSE.\n";
            } else {
                std::cout << "→ allVerticesValid() would return TRUE.\n";
            }




            bool ok = optimizer_->addVertex(obs.observationVertex);

            assert(optimizer_->vertex(obs.observationEdge->vertices()[0]->id()) != nullptr);
            assert(optimizer_->vertex(obs.observationEdge->vertices()[1]->id()) != nullptr);
            assert(optimizer_->vertex(obs.observationPriorEdge->vertices()[0]->id()) != nullptr);

            //bool ok2 = false;
            bool ok2 = optimizer_->addEdge(obs.observationPriorEdge);
            bool ok3 = optimizer_->addEdge(obs.observationEdge);
            std::cout << "AddVertex, AddPrior, AddObservation " << ok << " " << ok2 << " " << ok3 << std::endl;
            std::cout << "obs.observationVertex->id(): " << obs.observationVertex->id() << std::endl;
            obs.initialized = true;
          } else {
            // still have at least one uninitialized left
            foundAny = true;
          }
        }
      }
      // 3) if none left uninitialized, clear the flag
      if (!foundAny) {
        haveUninitializedObs_ = false;
      }
    }
    if(verbose_){std::cout << "handleObervationSynResponse Complete\n";}
  }

  void FileSlamSystem::platformEstimate2d(Eigen::Vector3d& x, Eigen::Matrix2d& P){

  }

  void FileSlamSystem::platformEstimate(Eigen::Isometry3d& x, Eigen::Matrix<double,6,6>& P){

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
