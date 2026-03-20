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

#include "drone_slam_system.h"

#include <map>
#include <vector>
#include <fstream>
#include <iomanip>

#include <nlohmann/json.hpp>
#include <Eigen/Geometry>



namespace g2o {
namespace tutorial {
namespace multibotsim{

static bool isValidInformationMatrix(const Eigen::Matrix<double,6,6>& info) {
  auto eig = info.selfadjointView<Eigen::Upper>().eigenvalues();
  if (!info.allFinite() || (eig.array() <= 0).any() ||
    eig.maxCoeff() > 1e10){
      std::cout << "is Valid information failed, " << (!info.allFinite()) << (eig.array() <= 0) << (eig.maxCoeff() > 1e6) << std::endl;
      return false;
    }
  return true;
}

using namespace Eigen;

using VertexContainer = g2o::OptimizableGraph::VertexContainer;



  MultiDroneSLAMSystem::MultiDroneSLAMSystem(const std::string& id, const std::string& filename)
    :SlamSystemBase<VertexSE3, EdgeSE3>(filename), robotId_(id), gndActive_(false), haveUninitializedObs_(false){

  }

  MultiDroneSLAMSystem::~MultiDroneSLAMSystem()=default;


  void MultiDroneSLAMSystem::start(){
    graphChanged_ = false;

    auto* offset = new g2o::ParameterSE3Offset; // identity offset	
    offset->setOffset(Isometry3d::Identity());	
    offset->setId(0); // ID = 0		
    optimizer_->addParameter(offset); // <-- only once
  }


  void MultiDroneSLAMSystem::stop(){
    optimize(optCountStop_);

    //if (fixOlderPlatformVertices_ == true){
    // TODO We are doing id = 0 for now.
    for (const auto& vertex : optimizer_->vertices()) {
      g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(vertex.second);
    }
    optimize(optCountStopFix_);

    for (size_t i = 0; i < observations_.size(); ++i) {
      const auto& obs = observations_[i];
      std::cout << "Observation[" << i << "] (observationId " << obs.observationId
                << ") initialized? " << (obs.initialized ? "Yes" : "No") << std::endl;
    }

      // std::cout << "Number of intra: " << intraRobotCount_ << std::endl;  // Commented out - variable not defined
  }

  void MultiDroneSLAMSystem::processEvent(Event& event){
    graphChanged_ = true;
    switch(event.type()){
      case Event::EventType::DataObservation:
        handleObservationEvent(static_cast<DataObsEvent&>(event));
        break;
      case Event::EventType::DataOdometry:
        stepNumber_ +=1;
        handleOdometryEvent(static_cast<DataOdomEvent&>(event));
        break;
      case Event::EventType::DataInitialization:
        handleInitializationEvent(static_cast<DataInitEvent&>(event));
        break;
      default:
        if(verbose_){std::cout << " - Unknown Event ..." << std::endl;}
        ignoreUnknownEventType();
        break;
    }
  }



  void MultiDroneSLAMSystem::ignoreUnknownEventType(){}


  void MultiDroneSLAMSystem::handleInitializationEvent(DataInitEvent event){
    if(verbose_){std::cout << " - Bot " << robotId_ << " - SlamSystem handleInitializationEvent start ..." << std::endl;}
    if(verbose_){std::cout << " - Creating vertex ..." << std::endl;}

    currentPlatformVertex_ = new VertexSE3();
    int vid = platformVertices_.size();
    currentPlatformVertex_->setId(vid);
    currentPlatformVertex_->setEstimate(event.value);
    vertexStampMap_.add(event.time, vid);

    if(verbose_){std::cout << " - Adding vertex to optimizer ..." << std::endl;}

    optimizer_->addVertex(currentPlatformVertex_);
    platformVertices_.emplace_back(currentPlatformVertex_);
    // place the id into the vertex id map

    if (event.posFixed) {
      if(verbose_){std::cout << " - Fixing initial vertex ..." << std::endl;}
      currentPlatformVertex_->setFixed(true);
    } else {
      // Anchor the initial pose with a prior instead of fixing the vertex.
      // This keeps the graph well-posed while still allowing refinement.
      if(verbose_){std::cout << " - Adding initialization prior edge ..." << std::endl;}
      auto* prior = new g2o::EdgeSE3Prior();
      prior->setVertex(0, currentPlatformVertex_);
      prior->setMeasurement(event.value);
      prior->setInformation(event.information);
      optimizer_->addEdge(prior);
      currentPlatformVertex_->setFixed(false);
    }
    initialized_ = true;

    if(verbose_){std::cout << " - SlamSystem handleInitializationEvent end ..." << std::endl;}
  }


  void MultiDroneSLAMSystem::handleOdometryEvent(DataOdomEvent event){
    if(verbose_){std::cout << " - Bot " << robotId_ << " - SlamSystem handleOdometryEvent start ..." << std::endl;}
    // Odom event carries velocity-like values. Convert to displacement using odom dt.
    if (!hasLastOdomTime_) {
      hasLastOdomTime_ = true;
      lastOdomTime_ = event.time;
      if (verbose_) {
        std::cout << " - First odom event received, caching timestamp only." << std::endl;
      }
      return;
    }
    const double odomDt = event.time - lastOdomTime_;
    lastOdomTime_ = event.time;
    if (odomDt < 1e-6 || !std::isfinite(odomDt)) {
      if (verbose_) {
        std::cout << " - Odom dt too small/invalid (" << odomDt << "), skipping event." << std::endl;
      }
      return;
    }

    // Body frame (Z-up world, yaw about Z): translation x = v_fwd, y = 0, z = v_up (vertical).
    const double vel_fw = event.value.translation().x();
    const double vel_z = event.value.translation().z();
    // Angular rate must come from the scalar field: value.linear() was wrongly used as R_z(ω),
    // which only determines ω modulo 2π (e.g. ω=2π → identity matrix → zero rate after atan2).
    const double omega = event.omegaZ;

    Isometry3 delta = Isometry3::Identity();
    delta.translation() = Eigen::Vector3d(vel_fw * odomDt, 0.0, vel_z * odomDt);
    delta.linear() = Eigen::AngleAxisd(omega * odomDt, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    Eigen::Matrix<double, 6, 6> deltaInfo = event.information;
    // Incoming information is in velocity space for (x,z,yaw). Convert only those
    // diagonal terms to displacement space by scaling with 1/dt^2.
    deltaInfo *= 1.0 / (odomDt * odomDt);
    // Keep information bounded for solver stability and isValidInformationMatrix checks.
    const double maxDiag = 9.0e9;
    for (int i = 0; i < 6; ++i) {
      if (!std::isfinite(deltaInfo(i, i)) || deltaInfo(i, i) <= 0.0) {
        deltaInfo(i, i) = 1.0;
      } else if (deltaInfo(i, i) > maxDiag) {
        deltaInfo(i, i) = maxDiag;
      }
    }

    if(verbose_){std::cout << " - Creating vertex ..." << std::endl;}
    currentPlatformVertex_ = new VertexSE3();

    int vid = platformVertices_.size();
    currentPlatformVertex_->setId(vid);
    vertexStampMap_.add(event.time, vid);
    if(verbose_){std::cout << " - Adding vertex to optimizer ..." << std::endl;}
    optimizer_->addVertex(currentPlatformVertex_);

    platformVertices_.emplace_back(currentPlatformVertex_);

    // TODO replace with initialization prior
    if(verbose_){std::cout << " - Adding Edge ..." << std::endl;}
    EdgeSE3* odometry = new EdgeSE3();
    VertexSE3* v0 = platformVertices_[platformVertices_.size() - 2];
    odometry->setVertex(0, v0);
    odometry->setVertex(1, currentPlatformVertex_);

    OptimizableGraph::VertexSet fromSet;
    fromSet.insert(v0);
    //odometry->initialEstimate(fromSet, v0);
    currentPlatformVertex_->setEstimate(v0->estimate() * delta);

    if(verbose_){std::cout << " - Vertex set, setting measurements ..." << std::endl;}
    odometry->setMeasurement(delta);
    //assert(odometry->information().rows() == 3);
    if(verbose_){std::cout << " - measurements set, setting information ..." << std::endl;}
    assert(isValidInformationMatrix(deltaInfo));
    odometry->setInformation((deltaInfo));
    if(verbose_){std::cout << " - Adding edge to optimizer ..." << std::endl;}
    optimizer_->addEdge(odometry);

    currentTime_ = event.time;

    if(verbose_ && false){
      // Optional: Set formatting for better readability
      std::cout << std::fixed << std::setprecision(6);
      std::cout << "\n\n\n\n Current Vertex\n";
      Isometry3 v0Iso  = v0->estimate();
      std::cout << "Translation (x y z):\n" << v0Iso.translation().transpose() << "\n\n";
      std::cout << "Rotation matrix (3x3):\n" << v0Iso.rotation() << "\n\n";
      // Print translation
      std::cout << "Odom Edge Vertex\n";
      std::cout << "Translation (x y z):\n" << delta.translation().transpose() << "\n\n";
      // Print rotation matrix
      std::cout << "Rotation matrix (3x3):\n" << delta.rotation() << "\n\n";
      // Print quaternion form
      Quaterniond q(delta.rotation());
      std::cout << "Quaternion (w x y z):\n" 
          << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "\n\n";
      // Print full isometry matrix
      std::cout << "Isometry3 (4x4 homogeneous transformation):\n" << delta.matrix() << "\n\n";
      // Print 6x6 information matrix
      std::cout << "Information matrix (6x6):\n" << deltaInfo << "\n";
    }


    if(verbose_){std::cout << " - SlamSystem handleOdometryEvent end ..." << std::endl;}
  }


  void MultiDroneSLAMSystem::handleObservationEvent(DataObsEvent event){
    //TODO implement
    if(verbose_){std::cout << " - SlamSystem handleObservationEvent start for robot: " << robotId_ << std::endl;}

    if(verbose_){std::cout << " - observation from robot: " << robotId_ << " To: " << event.robotIdTo << std::endl;}
    // Initialize Vertex
    // place the id into the vertex id map

    // We add a vertex representing the pose frame transform between this robot and other bots. 
    if (relativeTransforms_.find(event.robotIdTo) == relativeTransforms_.end()) {
      // Assign a consecutive integer to this robot ID if not already assigned
      if (robotIdToIntMap_.find(event.robotIdTo) == robotIdToIntMap_.end()) {
        robotIdToIntMap_[event.robotIdTo] = nextRobotIdInt_;
        nextRobotIdInt_++;
      }
      int robotIdInt = robotIdToIntMap_[event.robotIdTo];
      
      VertexSE3* v = new VertexSE3();
      // We assume that there's no more than 20000 vertices in the map
      // TODO Change
      v->setId(200000 + robotIdInt);
      v->setEstimate(Isometry3d::Identity());
      relativeTransforms_[event.robotIdTo] = v;
      optimizer_->addVertex(v);
    }

    VertexSE3* observedVtx;
    EdgeSE3* observationPrior;
    // NOTE: Here's another problem, the observation event must happen after the odometry. We'll enforce this at the data generation phase for now.
    // Again, pain in the butt to deal with. Not nessesary for our experiment for now.
    VertexSE3* v0 = platformVertices_[vertexStampMap_.query(event.time)];

    observedVtx = new VertexSE3;
    observedVtx->setEstimate(v0->estimate() * event.value);
    // NOTE: this will cause failure if the odom entries exceed 10000, but for the purposes of this it doesn't matter.
    // We take this short cut for simplicity. Otherwise we will have the whole ID management pain in the butt.
    observedVtx->setId(100000 + observations_.size());

    // Each observation is recorded once.
    // So the vertex prior and vertex mapping is removed: There's no way to do these kinds of matching

      // Initialize Observed Prior Edge (uninitialized)
    if (!optimizer_->parameter(0)) {
      ParameterSE3Offset* offset = new ParameterSE3Offset();
      offset->setOffset(Eigen::Isometry3d::Identity());  // Or a valid transform
      offset->setId(0);
      optimizer_->addParameter(offset);
    }
    observationPrior = new EdgeSE3;
    observationPrior->setVertex(0,relativeTransforms_[event.robotIdTo]);
    observationPrior->setVertex(1,observedVtx);
    observationPrior->setMeasurement(v0->estimate() * event.value);
    observationPrior->setInformation(Eigen::Matrix<double,6,6>::Identity());
    auto rk = new g2o::ToggelableGNDKernel(2.0, 6, 1e-3, 2.0*2.0, &gndActive_);
    observationPrior->setRobustKernel(rk);

    assert(v0 && "platform vertex lookup failed for vtxIdFrom");
    assert(observedVtx && "external vertex lookup failed for vtxIdTo");
    

    // Initialize Observation Edge
    EdgeSE3* observation = new EdgeSE3();
    observation->setVertex(0,v0);
    observation->setVertex(1,observedVtx);
    observation->setMeasurement(event.value);
    observation->setInformation(event.information);


    observations_.emplace_back(robotId_, event.time, exVtxCount_, event.robotIdTo,
            observationPrior, observation, observedVtx);
    exVtxCount_ ++;
    haveUninitializedObs_ = true;

  }



  DSMessage MultiDroneSLAMSystem::broadcastDSMessage() const{
    // Pre-allocate the vector for efficiency
    std::vector<PoseStampEntry> syncReqs;
    syncReqs.reserve(observations_.size());

    // Convert each Observation into an ObsSyncRequest
    for (const auto& obs : observations_) {
      // 1) Use the existing ID-only constructor
      PoseStampEntry entry(
        /* time */          obs.observationTime,                
        /* sourceId */      robotId_,                   // who originated this query
        /* observationId */ obs.observationId,          // local obs id (for matching on return)
        /* subjectId */     obs.observedRobotId         // which robot / subject this pose refers to
      );
      // 2) Fill in the actual pose + information from the edge
      syncReqs.push_back(std::move(entry));
    }

    // Build & return the final message; mark it as outgoing
    return DSMessage(
      /* sender    */ robotId_,
      /* loaded */    false,
      /* syncReqs  */ std::move(syncReqs)
    );
  }

  // receiveObsSyncMessage
  DSMessage MultiDroneSLAMSystem::handleObservationSyncRequest(DSMessage& msg){
    // NOTE:
    // We have an assumption here that querying the temporally "closest" vertex is good enough.
    // This holds in the experiment. But to generalise something need to be implemented. 
    // Something that runs a weighted average of the pose and covariance of the two closest vertex will do.
    std::vector<PoseStampEntry> localEntries;
    localEntries.reserve(msg.poseEntries.size());
    std::vector<g2o::OptimizableGraph::Vertex*> verticesToMarginalize;
    verticesToMarginalize.reserve(localEntries.size());
    for (auto const& pe : msg.poseEntries) {
      if (pe.subjectId != robotId_){
        continue;
      } 


      int idx = vertexStampMap_.query(pe.time);
      auto* v = platformVertices_[idx];
      //std::cerr << "Hessian Index" << v->hessianIndex() << std::endl;
      if (!v) {
        std::cerr << "  [WARN] platformVertices_[" << idx 
                  << "] is null\n";
        continue;
      }
      // both req and vertex are good—keep them
      localEntries.push_back(pe);
      verticesToMarginalize.push_back(v);
    }

    // 2) if nothing to do, early out immediately
    if (localEntries.empty()) {
      if (verbose_) {
        std::cout << "[REQ] no observations to sync for robot "
                  << robotId_ << " – skipping optimize+marginals\n";
      }
      return DSMessage(robotId_, /*outGoing=*/false, {});
    }

    // Step 2: optimize our graph so that we have up-to-date estimates
    if(graphChanged_ && false){
      if(verbose_){std::cout << "Optimizing before marginalization:\n";}
      optimizer_->initializeOptimization();
      optimizer_->optimize(10);
      graphChanged_ = false;
    }

    // Step 3: assemble the list of vertex pointers for marginal covariances
    // WARNING: no fail proofs
    for (auto* v : verticesToMarginalize) {
      assert(v);
      assert(!v->fixed());
      if (v->hessianIndex() < 0) {
        std::cerr << "[marg] skip v id=" << v->id()
                  << " fixed=" << v->fixed()
                  << " hIdx=" << v->hessianIndex() << "\n";
      }
      assert(v->hessianIndex() >= 0);

    }
    int numVertices = optimizer_->vertices().size();
    int numEdges    = optimizer_->edges().size();
    if(verbose_){std::cout << "Graph contains: " << (numVertices) << " vertices and " << numEdges << " edges\n";}
    if(verbose_){std::cout << "Marginalizing " << verticesToMarginalize.size() << " vertices\n";}
    g2o::SparseBlockMatrix<Eigen::MatrixXd> margCov;
    
    //assert(false);
    optimizer_->initializeOptimization();
    optimizer_->optimize(10);
    bool margSuccess = optimizer_->computeMarginals(margCov, verticesToMarginalize);
    if(verbose_){std::cout << "Marginalization success: " << margSuccess << "\n";}
    if(!margSuccess){return DSMessage(robotId_, /*outGoing=*/false, {});}


    // Step 4: fill in each ObsSyncRequest with measurement+information
    std::vector<PoseStampEntry> validResponses;
    validResponses.reserve(localEntries.size());
    for (size_t i = 0; i < localEntries.size(); ++i) {
      //if(verbose_){std::cout << "entering for loop \n";}
      auto& req = localEntries[i];
      auto* v = dynamic_cast<g2o::VertexSE3*>(verticesToMarginalize[i]);
      int vhIdx = v->hessianIndex();
      // 3.1: MAP estimate
      assert(v && "v to marginalize is nullptr");
      req.pose = v->estimate();
      req.hasPose = true;

      // 3.2: covariance → information = inverse(covariance)
      assert(margCov.block(vhIdx, vhIdx) && "Marg block is nullptr");
      //if(verbose_){std::cout << "marg cov block extraction start\n";}
      const Eigen::Matrix<double,6,6>& cov = *margCov.block(vhIdx, vhIdx);
      Eigen::Matrix<double,6,6> info = cov.inverse();
      // **Validation**: only keep if the info‐matrix is well‐formed
      if (isValidInformationMatrix(info)) {
        req.information = std::move(info);
        validResponses.push_back(std::move(req));
        
      } else {
        std::cerr << "Dropping sync for observation " << req.observationId
                  << " due to invalid information matrix\n";
                  if(verbose_){std::cout << "cov (6x6):\n" << cov << "\n";}
      }
    }
    if(verbose_){std::cout << "handleObservationSyncRequest Complete\n";}

    // Step 5: return only the valid responses
    return DSMessage(
      /* sourceId  */ msg.sourceId,
      /* loaded  */ true,
      /* poseEntries  */ std::move(validResponses)
    );
  }



  void MultiDroneSLAMSystem::handleObservationSyncResponse(const DSMessage& message) {
    // 1) Update all matching SE3Prior edges from the external cache
    if(verbose_){std::cout << "handleObservationSyncResponse Start\n";}
    for (const auto& pe : message.poseEntries) {
      // Only handle entries that belong to *this* robot's outstanding requests.
      // `PoseStampEntry.sourceId` is the robot that originated the query.
      if (pe.sourceId != robotId_) {
        continue;
      }
      // A response entry must refer to the robot pose we requested (the observed robot id)
      if (pe.observationId < 0 || pe.observationId >= static_cast<int>(observations_.size())) {
        std::cerr << "Invalid observationId: " << pe.observationId << std::endl;
        continue;
      }
      if (observations_[pe.observationId].observedRobotId != pe.subjectId) {
        std::cerr << "Mismatched subjectId for observationId " << pe.observationId
                  << " (expected " << observations_[pe.observationId].observedRobotId
                  << ", got " << pe.subjectId << ")\n";
        continue;
      }
      g2o::EdgeSE3* priorEdge = observations_[pe.observationId].observationPriorEdge;
      // // overwrite the cached prior with the new measurement + information

      // // WARNING: This does not check information validity. It assumes the information is valid.
      priorEdge->setMeasurement(pe.pose);
      priorEdge->setInformation(pe.information / 4);
    }

    if(verbose_){std::cout << "step 2 Start\n";}
    // 2) If we still have uninitialized observations, try to add them now
    if (haveUninitializedObs_) {
      bool foundAny = false;
      for (size_t i=0; i < observations_.size(); i++) {
        if (!observations_[i].initialized) {
          // see if we just got a syncResponse for this observationId
          // (we assume syncRequests covered it above)
          auto wasUpdated = std::any_of(
            message.poseEntries.begin(), message.poseEntries.end(),
            [&](auto const& req) {
              return req.observationId == i;
            }
          );
          if (wasUpdated) {
            // assert(obs.observationVertex != nullptr);
            // assert(obs.observationEdge != nullptr);
            // assert(obs.observationPriorEdge != nullptr);
            // assert(VertexIdMap_.find(obs.observationVertex->id()) == VertexIdMap_.end());
            // assert(obs.observationVertex == obs.observationPriorEdge->vertices()[1]);
            // assert(obs.observationVertex == obs.observationEdge->vertices()[1]);
            // assert(obs.observationPriorEdge->information().rows() == 6); // or whatever dimension
            // assert(obs.observationPriorEdge->information().determinant() > 0);


            auto* vertex = observations_[i].observationVertex;
            auto* prior = observations_[i].observationPriorEdge;
            bool all_valid = true;
            bool ok = optimizer_->addVertex(observations_[i].observationVertex);

            // assert(optimizer_->vertex(obs.observationEdge->vertices()[0]->id()) != nullptr);
            // assert(optimizer_->vertex(obs.observationEdge->vertices()[1]->id()) != nullptr);
            // assert(optimizer_->vertex(obs.observationPriorEdge->vertices()[0]->id()) != nullptr);

            //bool ok2 = false;
            bool ok2 = optimizer_->addEdge(observations_[i].observationPriorEdge);
            bool ok3 = optimizer_->addEdge(observations_[i].observationEdge);
            std::cout << "AddVertex, AddPrior, AddObservation " << ok << " " << ok2 << " " << ok3 << std::endl;
            std::cout << "obs.observationVertex->id(): " << observations_[i].observationVertex->id() << std::endl;
            observations_[i].initialized = true;
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
    if(verbose_){std::cout << "handleObservationSyncResponse Complete\n";}
  }

  void MultiDroneSLAMSystem::platformEstimate(Eigen::Isometry3d& x, Eigen::Matrix<double,6,6>& P){

  }

  void MultiDroneSLAMSystem::platformEstimate(Eigen::Isometry3d& pose) const {
    // 1) translation
    pose = currentPlatformVertex_->estimate();
    // const Eigen::Vector3d& t3 = pos.translation();
    // double x = t3.x();
    // double y = t3.y();

    // // 2) extract yaw from rotation matrix R
    // Eigen::Matrix3d R = pos.rotation();
    // // yaw = atan2(sinθ, cosθ) = atan2(R(1,0), R(0,0))
    // double yaw = std::atan2(R(1,0), R(0,0));

    // pose = Eigen::Vector3d(x, y, yaw);
  }

// NOTE: Another look
  std::vector<std::pair<double, Eigen::Isometry3d>> MultiDroneSLAMSystem::getTrajectory() const {
    std::vector<std::pair<double, Eigen::Isometry3d>> trajectory;
    
    // Get the timestamp-vertex mapping from vertexStampMap_
    const auto& stampData = vertexStampMap_.data();
    
    // Reserve space for efficiency
    trajectory.reserve(stampData.size());
    
    // Iterate through all timestamp entries
    for (const auto& entry : stampData) {
      double timestamp = entry.t;
      int vertexId = entry.id;
      
      // Check if vertex ID is valid
      if (vertexId >= 0 && vertexId < static_cast<int>(platformVertices_.size())) {
        VertexSE3* vertex = platformVertices_[vertexId];
        if (vertex) {
          Eigen::Isometry3d pose = vertex->estimate();
          trajectory.emplace_back(timestamp, pose);
        }
      }
    }
    
    return trajectory;
  }

  void MultiDroneSLAMSystem::saveTrajectoryTUM(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Cannot open file for writing trajectory: " + filename);
    }
    
    // Set precision for output
    out << std::fixed << std::setprecision(9);
    
    // Get trajectory
    auto trajectory = getTrajectory();
    
    // Write TUM format: timestamp x y z qx qy qz qw
    for (const auto& [timestamp, pose] : trajectory) {
      Eigen::Vector3d translation = pose.translation();
      Eigen::Quaterniond quaternion(pose.rotation());
      
      out << timestamp << " "
          << translation.x() << " " << translation.y() << " " << translation.z() << " "
          << quaternion.x() << " " << quaternion.y() << " " << quaternion.z() << " " << quaternion.w()
          << std::endl;
    }
    
    out.close();
    std::cout << "Saved trajectory with " << trajectory.size() << " poses to " << filename << std::endl;
  }

}
}  // namespace tutorial
}  // namespace g2o
