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
#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <Eigen/Geometry>



#include "g2o/core/optimization_algorithm_with_hessian.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"

namespace g2o {
namespace tutorial {
namespace multibotsim{

using json = nlohmann::json;

int MultiDroneSLAMSystem::preOptTrajectoryBatchCounter_ = 0;

void MultiDroneSLAMSystem::resetPreOptTrajectoryBatchCounter() {
  preOptTrajectoryBatchCounter_ = 0;
}

int MultiDroneSLAMSystem::takeNextPreOptTrajectoryBatchIndex() {
  return preOptTrajectoryBatchCounter_++;
}

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
    : SlamSystemBase<VertexSE3, EdgeSE3>(filename),
      robotId_(id),
      exVtxCount_(0),
      gndActive_(false),
      haveUninitializedObs_(false),
      graphChanged_(false) {
    std::ifstream f(filename);
    if (!f) {
      throw std::runtime_error("MultiDroneSLAMSystem: cannot open SLAM config: " + filename);
    }
    json j;
    f >> j;
    gndActiveConfig_ = j.value("gndActive_", true);
    gndActive_ = gndActiveConfig_;

    gndBound_ = j.value("gndBound_", gndBound_);
    gndPower_ = j.value("gndPower_", gndPower_);
    gndLnc_ = j.value("gndLnc_", gndLnc_);
    gndTailPenaltyStd_ = j.value("gndTailPenaltyStd_", gndTailPenaltyStd_);

    // Fix relative-transform vertices to identity (debug/robustness switch).
    fixRelativetransform_ = j.value("fixRelativetransform_", false);
  }

  g2o::ToggelableGNDKernel* MultiDroneSLAMSystem::newPriorToggelableGndKernel() {
    return new g2o::ToggelableGNDKernel(
        gndBound_, gndPower_, gndLnc_, gndTailPenaltyStd_, &gndActiveAlwaysFalse_);
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
    if(verbose_){std::cout << "Bot " << robotId_ << " stop start\n";}
    
    optimize(optCountStop_);
    if (gndActiveConfig_) {
      gndActive_ = true;
    }
    //if (fixOlderPlatformVertices_ == true){
    // TODO We are doing id = 0 for now.
    for (const auto& vertex : optimizer_->vertices()) {
      g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(vertex.second);
    }
    optimize(optCountStopFix_);

    // Debug: dump relative-transform vertices (should be near identity if everything is consistent).
    if (preOptTrajectoryDumpEnabled_) {
      std::cout << "\n[Bot " << robotId_ << "] relativeTransforms_ dump @ stop()\n";
      if (relativeTransforms_.empty()) {
        std::cout << "  (none)\n";
      } else {
        for (const auto& kv : relativeTransforms_) {
          const std::string& observedRobotKey = kv.first;
          const VertexSE3* v = kv.second;
          if (!v) continue;
          const Isometry3d T = v->estimate();
          const Eigen::Vector3d t = T.translation();
          const double rotAngle = Eigen::AngleAxisd(T.rotation()).angle();
          std::cout << "  observedRobotId=" << observedRobotKey
                    << " | t=(" << t.transpose() << ")"
                    << " | rotAngle(rad)=" << rotAngle << "\n";
        }
      }
      std::cout << std::endl;
    }

    if (preOptTrajectoryDumpEnabled_) {
      size_t initializedCount = 0;
      for (const auto& obs : observations_) {
        initializedCount += obs.initialized ? 1 : 0;
      }
      std::cout << "  observations_: " << observations_.size()
                << " | initialized=" << initializedCount << std::endl;

      // Optionally print a small prefix to spot issues without spamming.
      const size_t maxPrint = 8;
      for (size_t i = 0; i < observations_.size() && i < maxPrint; ++i) {
        const auto& obs = observations_[i];
        std::cout << "  Observation[" << i << "] (obsId " << obs.observationId
                  << ") initialized=" << (obs.initialized ? "Yes" : "No") << std::endl;
      }
    }

      // std::cout << "Number of intra: " << intraRobotCount_ << std::endl;  // Commented out - variable not defined
  }

  void MultiDroneSLAMSystem::onAfterOptimize() {
    if (pendingGndPriorEdges_.empty()) {
      return;
    }

    // First time each pending prior edge participates in an optimization, we
    // switch its GND kernel from `gndActiveAlwaysFalse_` to `gndActive_` (same for
    // inter-robot SE3 observation priors and landmark EdgeSE3PointXYZ estimate priors).
    for (auto* e : pendingGndPriorEdges_) {
      if (!e) continue;

      auto* rk = dynamic_cast<g2o::ToggelableGNDKernel*>(e->robustKernel());
      if (rk) {
        rk->setBoolPointer(&gndActive_);
      } else if (verbose_) {
        std::cerr << "[GND] Pending prior edge has no ToggelableGNDKernel; id="
                  << e->id() << std::endl;
      }
    }
    pendingGndPriorEdges_.clear();
  }

  void MultiDroneSLAMSystem::setPreOptTrajectoryOutputDir(const std::string& output_dir) {
    preOptTrajectoryOutputDir_ = output_dir;
  }

  void MultiDroneSLAMSystem::setPreOptTrajectoryDumpEnabled(bool enabled) {
    preOptTrajectoryDumpEnabled_ = enabled;
  }

  void MultiDroneSLAMSystem::dumpPreOptTrajectory(const std::string& run_directory) {
    if (preOptTrajectoryDumpEnabled_ && !run_directory.empty()) {
      std::filesystem::path out_dir(run_directory);
      std::error_code mkdir_ec;
      std::filesystem::create_directories(out_dir, mkdir_ec);
      if (mkdir_ec) {
        throw std::runtime_error(
            "Cannot create pre-opt trajectory directory '" + run_directory +
            "': " + mkdir_ec.message());
      }
      const std::string filename =
          (out_dir / ("trajectory_" + robotId_ + ".txt")).string();
      saveTrajectoryTUM(filename);
    }
  }

  void MultiDroneSLAMSystem::processEvent(Event& event){
    graphChanged_ = true;
    switch(event.type()){
      case Event::EventType::Other: {
        auto& dataEvent = static_cast<DataEventBase&>(event);
        switch (dataEvent.dataEventType()) {
          case DataEventType::Initialization:
            handleInitializationEvent(static_cast<DataInitEvent&>(event));
            break;
          case DataEventType::Odometry:
            stepNumber_ +=1;
            handleOdometryEvent(static_cast<DataOdomEvent&>(event));
            break;
          case DataEventType::Observation:
            handleObservationEvent(static_cast<DataObsEvent&>(event));
            break;
          case DataEventType::LandmarkObservation:
            handleLMObservationEvent(static_cast<DataLmObsEvent&>(event));
            break;
          default:
            ignoreUnknownEventType();
            break;
        }
        break;
      }
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


  void MultiDroneSLAMSystem::handleLMObservationEvent(DataLmObsEvent event) {
    if (!initialized_) {
      if (verbose_) {
        std::cout << " - handleLMObservationEvent: skip (not initialized)\n";
      }
      return;
    }

    if (event.landmarkId < 0) {
      std::cerr << "[LM] invalid landmark id (expected non-negative int)\n";
      return;
    }
    const int lmKey = event.landmarkId;

    VertexSE3* v0 = platformVertices_[vertexStampMap_.query(event.time)];
    if (!v0) {
      std::cerr << "[LM] platform vertex lookup failed at t=" << event.time << std::endl;
      return;
    }

    // Landmarks are points in world frame. Observations are body-frame relative position.
    // Do NOT use EdgeSE3 between two VertexSE3: that treats the landmark as a full pose and
    // (with rotation pinned via artificial info) forces the landmark rotation to track every
    // robot heading, which conflicts and wrecks the whole graph. EdgeSE3PointXYZ matches
    // the monocular / RGB-D convention: error = world point expressed in sensor frame minus meas.
    Eigen::Matrix3d info3 = event.information;
    if (!info3.allFinite()) {
      info3 = Eigen::Matrix3d::Identity();
    }

    VertexPointXYZ* lmVtx = nullptr;

    auto it = landmarks_.find(lmKey);
    if (it == landmarks_.end()) {
      lmVtx = new VertexPointXYZ();
      lmVtx->setId(400000 + nextLandmarkVertexSeq_++);
      const Eigen::Vector3d p_world = v0->estimate() * event.relPos;
      lmVtx->setEstimate(p_world);
      optimizer_->addVertex(lmVtx);

      Landmark L;
      L.lmId = lmKey;
      L.landmark = lmVtx;
      L.initialized = true;
      landmarks_[lmKey] = std::move(L);
    } else {
      lmVtx = it->second.landmark;
      if (!lmVtx) {
        std::cerr << "[LM] landmark vertex null for lmKey=" << lmKey << std::endl;
        return;
      }

      // Probably could be combined with the above control flow, but it is what it is. 
      if (!it->second.initialized) {
        lmVtx = new VertexPointXYZ();
        lmVtx->setId(400000 + nextLandmarkVertexSeq_++);
        const Eigen::Vector3d p_world = v0->estimate() * event.relPos;
        lmVtx->setEstimate(p_world);
        optimizer_->addVertex(lmVtx);

        // Add all communicated priors that reference this landmark.
        for (auto& [observerId, est] : it->second.landmarkEsts) {
          (void)observerId;
          if (!est.estimatePriorEdge) continue;
          est.estimatePriorEdge ->setVertex(1, lmVtx);
          if (optimizer_->addEdge(est.estimatePriorEdge)) {
            pendingGndPriorEdges_.push_back(est.estimatePriorEdge);
          }
        }

        it->second.initialized = true;
        graphChanged_ = true;
      }
    }

    //const Eigen::Vector3d lm_world = v0->estimate() * event.relPos;
    //if(verbose_){std::cout << " - LM: " <<lmKey << " " << "|" << lm_world.transpose()  << std::endl;}

    if (!optimizer_->parameter(0)) {
      auto* offset = new g2o::ParameterSE3Offset();
      offset->setOffset(Isometry3d::Identity());
      offset->setId(0);
      optimizer_->addParameter(offset);
    }

    auto* edge = new EdgeSE3PointXYZ();
    edge->setVertex(0, v0);
    edge->setVertex(1, lmVtx);
    edge->setMeasurement(event.relPos);
    edge->setInformation(info3);
    edge->setParameterId(0, 0);
    optimizer_->addEdge(edge);

    if (verbose_) {
      std::cout << " - handleLMObservationEvent: lmKey=" << lmKey
                << " edge added\n";
    }
  }

  void MultiDroneSLAMSystem::handleObservationEvent(DataObsEvent event){
    //TODO implement
    if(verbose_){std::cout << " - SlamSystem handleObservationEvent start for robot: " << robotId_ << std::endl;}

    if(verbose_){std::cout << " - observation from robot: " << robotId_ << " To: " << event.robotIdTo << std::endl;}
    // Initialize Vertex
    // place the id into the vertex id map

    // We add a vertex representing the pose frame transform between this robot and other bots. 
    // Create (or fetch) a single relative-transform vertex per observed robot id.
    const std::string& observedRobotKey = event.robotIdTo;
    if (relativeTransforms_.find(observedRobotKey) == relativeTransforms_.end()) {
      VertexSE3* v = new VertexSE3();
      v->setId(200000 + nextRelativeTransformVtxId_);
      v->setEstimate(Isometry3d::Identity());

      // Optional robustness switch: fix relative transforms to identity so the
      // optimizer cannot "explain away" comm errors by moving these vertices.
      // This is gated by `fixRelativetransform_` (editable in slam_system_config.json).
      if (fixRelativetransform_) {
        v->setFixed(true);
      }

      relativeTransforms_[observedRobotKey] = v;
      ++nextRelativeTransformVtxId_;
      relativeTransformInGraph_[observedRobotKey] = false;
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
    observationPrior->setVertex(0, relativeTransforms_[event.robotIdTo]);
    observationPrior->setVertex(1,observedVtx);
    observationPrior->setMeasurement(event.value);
    observationPrior->setInformation(Eigen::Matrix<double,6,6>::Identity());
    // Initially keep the GND kernel inactive for new prior edges.
    // After the edge has been optimized once, we switch its kernel to follow
    // this system's `gndActive_` via `onAfterOptimize()`.
    observationPrior->setRobustKernel(newPriorToggelableGndKernel());

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
    if(verbose_){std::cout << "Bot " << robotId_ << " broadcastDSMessage start\n";}
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
      /* lm_query  */ lmQueryEnabled_,
      /* poseEntries */ std::move(syncReqs)
    );
  }

  // receiveObsSyncMessage
  DSMessage MultiDroneSLAMSystem::handleObservationSyncRequest(DSMessage& msg){
    // NOTE:
    // We have an assumption here that querying the temporally "closest" vertex is good enough.
    // This holds in the experiment. But to generalise something need to be implemented.
    if (verbose_) { std::cout << "Bot " << robotId_ << " handleObservationSyncRequest start\n"; }

    // 1) Pose marginalization for any pose entries whose subject matches this robot.
    std::vector<PoseStampEntry> localPoseEntries;
    std::vector<g2o::OptimizableGraph::Vertex*> poseVerticesToMarginalize;
    localPoseEntries.reserve(msg.poseEntries.size());
    poseVerticesToMarginalize.reserve(msg.poseEntries.size());

    for (const auto& pe : msg.poseEntries) {
      if (pe.subjectId != robotId_) continue;

      const int idx = vertexStampMap_.query(pe.time);
      auto* v = platformVertices_[idx];
      if (!v) {
        std::cerr << "  [WARN] platformVertices_[" << idx << "] is null\n";
        continue;
      }

      localPoseEntries.push_back(pe);
      poseVerticesToMarginalize.push_back(v);
    }

    // 2) Landmark marginalization (optional).
    std::vector<g2o::OptimizableGraph::Vertex*> landmarkVerticesToMarginalize;
    std::vector<int> landmarkIdsToMarginalize;

    // Which robots asked for landmark priors? We infer from msg.poseEntries.sourceId.
    std::vector<std::string> observerIdsToRespondTo;
    if (msg.lm_query) {
      observerIdsToRespondTo.reserve(msg.poseEntries.size());
      for (const auto& pe : msg.poseEntries) {
        // Don't send landmark priors back to ourselves.
        if (pe.sourceId == robotId_) continue;
        if (std::find(observerIdsToRespondTo.begin(), observerIdsToRespondTo.end(), pe.sourceId) == observerIdsToRespondTo.end()) {
          observerIdsToRespondTo.push_back(pe.sourceId);
        }
      }
      if (observerIdsToRespondTo.empty()) {
        // Fallback: if we have no pose entries, respond to msg.sourceId.
        if (msg.sourceId != robotId_) {
          observerIdsToRespondTo.push_back(msg.sourceId);
        }
      }

      for (const auto& [lmId, lm] : landmarks_) {
        if (!lm.initialized) continue;
        if (!lm.landmark) continue;
        landmarkVerticesToMarginalize.push_back(lm.landmark);
        landmarkIdsToMarginalize.push_back(lmId);
      }
    }

    // 3) Early out if we have nothing to marginalize.
    if (localPoseEntries.empty() && landmarkVerticesToMarginalize.empty()) {
      if (verbose_) {
        std::cout << "[REQ] no observations/landmarks to sync for robot " << robotId_ << " – skipping\n";
      }
      return DSMessage(robotId_, /*loaded=*/false, msg.lm_query, /*poseEntries=*/{}, /*lmEntries=*/{});
    }

    // Step 2: optimize our graph so that we have up-to-date estimates.
    if (graphChanged_) {
      if (verbose_) { std::cout << "Optimizing before marginalization:\n"; }
      optimize(10);
      graphChanged_ = false;
    }

    // Assemble the list of vertices to marginalize.
    std::vector<g2o::OptimizableGraph::Vertex*> verticesToMarginalize;
    verticesToMarginalize.reserve(poseVerticesToMarginalize.size() + landmarkVerticesToMarginalize.size());
    verticesToMarginalize.insert(verticesToMarginalize.end(),
                                 poseVerticesToMarginalize.begin(),
                                 poseVerticesToMarginalize.end());
    verticesToMarginalize.insert(verticesToMarginalize.end(),
                                 landmarkVerticesToMarginalize.begin(),
                                 landmarkVerticesToMarginalize.end());

    for (auto* v : verticesToMarginalize) {
      assert(v);
      assert(!v->fixed());
      assert(v->hessianIndex() >= 0);
    }

    if (verbose_) {
      std::cout << "Graph contains: " << optimizer_->vertices().size() << " vertices and "
                << optimizer_->edges().size() << " edges\n";
      std::cout << "Marginalizing " << verticesToMarginalize.size() << " vertices\n";
    }

    g2o::SparseBlockMatrix<Eigen::MatrixXd> margCov;

    // Marginalization: disable GND robust kernel behavior during marginals.
    const bool oldGndActive = gndActive_;
    gndActive_ = false;

    optimizer_->initializeOptimization();
    optimizer_->computeActiveErrors();
    if (const auto* algoConst =
            dynamic_cast<const g2o::OptimizationAlgorithmWithHessian*>(optimizer_->algorithm())) {
      auto* algo = const_cast<g2o::OptimizationAlgorithmWithHessian*>(algoConst);
      algo->updateLinearSystem();
    }

    bool margSuccess = optimizer_->computeMarginals(margCov, verticesToMarginalize);

    gndActive_ = oldGndActive;
    if (verbose_) { std::cout << "Marginalization success: " << margSuccess << "\n"; }
    if (!margSuccess) {
      return DSMessage(robotId_, /*loaded=*/false, msg.lm_query, /*poseEntries=*/{}, /*lmEntries=*/{});
    }

    // Step 4: fill pose responses.
    std::vector<PoseStampEntry> validPoseResponses;
    validPoseResponses.reserve(localPoseEntries.size());

    for (size_t i = 0; i < localPoseEntries.size(); ++i) {
      auto& req = localPoseEntries[i];
      auto* v = dynamic_cast<g2o::VertexSE3*>(poseVerticesToMarginalize[i]);
      assert(v && "v to marginalize is nullptr");

      const int vhIdx = v->hessianIndex();
      req.pose = v->estimate();
      req.hasPose = true;

      assert(margCov.block(vhIdx, vhIdx) && "Marg block is nullptr");
      const Eigen::Matrix<double, 6, 6>& cov = *margCov.block(vhIdx, vhIdx);
      Eigen::Matrix<double, 6, 6> info = cov.inverse();

      if (isValidInformationMatrix(info)) {
        req.information = std::move(info);
        validPoseResponses.push_back(std::move(req));
      } else {
        std::cerr << "Dropping sync for observation " << req.observationId
                  << " due to invalid information matrix\n";
      }
    }

    // Step 5: fill landmark responses (if requested).
    std::vector<LMPoseEntry> validLmResponses;
    if (msg.lm_query) {
      validLmResponses.reserve(landmarkVerticesToMarginalize.size() * observerIdsToRespondTo.size());

      for (size_t k = 0; k < landmarkVerticesToMarginalize.size(); ++k) {
        auto* ptVtx = dynamic_cast<g2o::VertexPointXYZ*>(landmarkVerticesToMarginalize[k]);
        if (!ptVtx) continue;
        const int lmId = landmarkIdsToMarginalize[k];
        const int vhIdx = ptVtx->hessianIndex();

        auto* blk = margCov.block(vhIdx, vhIdx);
        if (!blk) continue;

        // VertexPointXYZ is 3D, so marginal cov is 3x3.
        if (blk->rows() != 3 || blk->cols() != 3) continue;
        const Eigen::Matrix3d cov3 = *blk;
        Eigen::Matrix3d info3 = cov3.inverse();
        if (!info3.allFinite()) continue;

        Eigen::Matrix<double, 6, 6> info6 = Eigen::Matrix<double, 6, 6>::Identity();
        info6.setZero();
        info6.topLeftCorner<3, 3>() = info3;
        info6.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity();

        Isometry3 pose = Isometry3::Identity();
        pose.translation() = ptVtx->estimate();

        for (const auto& observerId : observerIdsToRespondTo) {
          LMPoseEntry entry(lmId, observerId);
          entry.hasPose = true;
          entry.pose = pose;
          entry.information = info6;
          validLmResponses.push_back(std::move(entry));
        }
      }
    }

    if (verbose_) { std::cout << "handleObservationSyncRequest Complete\n"; }

    // Step 6: return both pose + landmark responses.
    return DSMessage(
      /* sender    */ msg.sourceId,
      /* loaded    */ true,
      /* lm_query  */ msg.lm_query,
      /* poseEntries */ std::move(validPoseResponses),
      /* lmEntries */ std::move(validLmResponses));
  }



  void MultiDroneSLAMSystem::handleObservationSyncResponse(const DSMessage& message) {
    // 1) Update all matching SE3Prior edges from the external cache
    if(verbose_){std::cout << "Bot " << robotId_ << " handleObservationSyncResponse start\n";}
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
      priorEdge->setInformation(pe.information);
      graphChanged_ = true;
    }

    // 1.5) Landmark sync: create/update landmark priors (EdgeSE3PointXYZ)
    // based on communicated MAP + covariance.
    if (message.lm_query && !message.lmEntries.empty()) {
      for (const auto& lme : message.lmEntries) {
        if (lme.landmarkId < 0) continue;

        const int lmId = lme.landmarkId;

        // Ensure landmark exists in mapping.
        auto it = landmarks_.find(lmId);
        if (it == landmarks_.end()) {
          Landmark L;
          L.lmId = lmId;
          L.initialized = false;
          it = landmarks_.emplace(lmId, std::move(L)).first;
        }

        auto& landmark = it->second;

        // (Insight, don't delete comment) landmark estimate vertex will be added during Landmark initialization, so no need to worry about that here 

        if (!currentPlatformVertex_) {
          if (verbose_) {
            std::cerr << "[LM sync] currentPlatformVertex_ is null; skipping prior creation.\n";
          }
          continue;
        }

        // Create/update the per-observer landmark estimate prior.
        // (Insight, don't delete comment): Id of the robot that produced this estimate/observed the landmark to form this prior
        const std::string& observerId = lme.observerId;  

        // SO.. This is techincally impossible, but I'll just keep this just in case. 
        if (relativeTransforms_.find(observerId) == relativeTransforms_.end()) {
          VertexSE3* rv = new VertexSE3();
          rv->setId(200000 + nextRelativeTransformVtxId_++);
          rv->setEstimate(Isometry3d::Identity());
          if (fixRelativetransform_) {
            rv->setFixed(true);
          }
          relativeTransforms_[observerId] = rv;
          relativeTransformInGraph_[observerId] = false;
        }
        
        if (!relativeTransformInGraph_[observerId]) {
          if (optimizer_->addVertex(relativeTransforms_[observerId])) {
            relativeTransformInGraph_[observerId] = true;
          } else {
            std::cerr << "[obs sync] failed to add relative transform vertex for observer "
                      << observerId << std::endl;
            continue;
          }
        }
        LandmarkEst& est = landmark.landmarkEsts[observerId];

        if (!est.estimatePriorEdge) {
          est.lmId = lmId;
          est.observerId = observerId;
          est.initialized = true;
          est.estimatePriorEdge = new EdgeSE3PointXYZ();

          // EdgeSE3PointXYZ requires the SE3 offset parameter.
          if (!optimizer_->parameter(0)) {
            auto* offset = new g2o::ParameterSE3Offset();
            offset->setOffset(Isometry3d::Identity());
            offset->setId(0);
            optimizer_->addParameter(offset);
          }

          // (Insight, don't delete comment):
          // The relative Transform is the transform from the observed robot's frame to this robot's frame.
          // The Prior is the landmark's pose in the observed robot's frame.
          est.estimatePriorEdge->setVertex(0, relativeTransforms_[observerId]);
          est.estimatePriorEdge->setVertex(1, landmark.landmark);
          est.estimatePriorEdge->setParameterId(0, 0);
          est.estimatePriorEdge->setRobustKernel(newPriorToggelableGndKernel());

          // (Insight, don't delete comment)
          // Add this edge only if the landmark has been activated in the factor graph.
          // If not added here, it'll be added during landmark initialization.
          if (landmark.initialized) {
            if (optimizer_->addEdge(est.estimatePriorEdge)) {
              pendingGndPriorEdges_.push_back(est.estimatePriorEdge);
            }
          }
        }

        // Update estimate prior edge measurement+information.
        const Eigen::Vector3d lmObserved = lme.pose.translation();

        Eigen::Matrix3d info3 = lme.information.topLeftCorner<3, 3>();
        if (!info3.allFinite()) {
          info3 = Eigen::Matrix3d::Identity();
        }

        // (NOTE, don't delete comment) OK so the divide by 4 thing is a bit of a hack.
        // It's to make ths bound of the kernel appear at 2 * std
        // It'd probably better if embedded in the kernel.
        est.estimatePriorEdge->setMeasurement(lmObserved);
        est.estimatePriorEdge->setInformation(info3);

        graphChanged_ = graphChanged_ || landmark.initialized;
      }
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
              // Important: `observationId` is local to *this* robot.
              // The response may contain entries coming from multiple robots,
              // so we must also match the query origin (`sourceId`) and the subject (`subjectId`).
              // Otherwise different robots can collide on the same numeric `observationId`,
              // causing us to add the wrong edges with stale/incorrect priors.
              return req.observationId == static_cast<int>(i) &&
                     req.sourceId == robotId_ &&
                     req.subjectId == observations_[i].observedRobotId;
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

            // EdgeSE3 constraint uses: from^{-1} * to = Z(measurement).
            // Here, `prior` connects:
            //   prior->vertices()[0] = relativeTransforms_* vertex (from)
            //   prior->vertices()[1] = observedVtx (to)
            // We check how far (from^{-1}*to) is from measurement Z.
            // if (verbose_) {
            //   const Isometry3 Tfrom = prior->vertices()[0]->estimate();
            //   const Isometry3 Tto = prior->vertices()[1]->estimate();
            //   const Isometry3 Z = prior->measurement();
            //   const Isometry3 T_expected = Tfrom.inverse() * Tto;
            //   const Eigen::Vector3d dTrans = T_expected.translation() - Z.translation();
            //   const Eigen::Matrix3d Rerr = Z.rotation().transpose() * T_expected.rotation();
            //   const double dRotRad = Eigen::AngleAxisd(Rerr).angle();
            //   if (i < 3) {
            //     std::cout << "[obsPrior SE3 check bot " << robotId_
            //               << " obsIdx=" << i
            //               << "] |dTrans|=" << dTrans.norm()
            //               << " dRot(rad)=" << dRotRad << std::endl;
            //   }
            // }

            //bool ok2 = false;
            bool ok2 = optimizer_->addEdge(observations_[i].observationPriorEdge);
            bool ok3 = optimizer_->addEdge(observations_[i].observationEdge);
            if (ok2) {
              pendingGndPriorEdges_.push_back(observations_[i].observationPriorEdge);
            }
            if (preOptTrajectoryDumpEnabled_ && se3PriorDiagPrinted_ < kSe3PriorDiagPrintMax) {
              std::cout << "AddVertex, AddPrior, AddObservation " << ok << " " << ok2 << " " << ok3 << std::endl;
              std::cout << "obs.observationVertex->id(): " << observations_[i].observationVertex->id() << std::endl;
              ++se3PriorDiagPrinted_;
            }
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
