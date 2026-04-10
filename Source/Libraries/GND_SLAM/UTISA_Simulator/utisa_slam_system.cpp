// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard

#include "utisa_slam_system.h"

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
#include "g2o/core/sparse_block_matrix.h"

namespace g2o {
namespace tutorial {
namespace multibotsim {

using json = nlohmann::json;

int UTISASlamSystem::preOptTrajectoryBatchCounter_ = 0;

void UTISASlamSystem::resetPreOptTrajectoryBatchCounter() {
  preOptTrajectoryBatchCounter_ = 0;
}

int UTISASlamSystem::takeNextPreOptTrajectoryBatchIndex() {
  return preOptTrajectoryBatchCounter_++;
}

static bool isValidInformationMatrix3(const Eigen::Matrix3d& info) {
  auto eig = info.selfadjointView<Eigen::Upper>().eigenvalues();
  if (!info.allFinite() || (eig.array() <= 0).any() || eig.maxCoeff() > 1e10) {
    return false;
  }
  return true;
}

static bool isValidInformationMatrix2(const Eigen::Matrix2d& info) {
  auto eig = info.selfadjointView<Eigen::Upper>().eigenvalues();
  if (!info.allFinite() || (eig.array() <= 0).any() || eig.maxCoeff() > 1e10) {
    return false;
  }
  return true;
}

static VertexSE2* platformVertexAtTimeSafe(const StampMap& stampMap,
                                           const std::vector<VertexSE2*>& platformVertices,
                                           double t) {
  const int idx = stampMap.query(t);
  if (idx < 0 || idx >= static_cast<int>(platformVertices.size())) {
    return nullptr;
  }
  return platformVertices[static_cast<size_t>(idx)];
}

using VertexContainer = g2o::OptimizableGraph::VertexContainer;

UTISASlamSystem::UTISASlamSystem(const std::string& id, const std::string& filename)
    : SlamSystemBase<VertexSE2, EdgeSE2>(filename),
      robotId_(id),
      exVtxCount_(0),
      gndActive_(false),
      haveUninitializedObs_(false),
      graphChanged_(false) {
  std::ifstream f(filename);
  if (!f) {
    throw std::runtime_error("UTISASlamSystem: cannot open SLAM config: " + filename);
  }
  json j;
  f >> j;
  gndActiveConfig_ = j.value("gndActive_", true);
  gndActive_ = gndActiveConfig_;

  gndBound_ = j.value("gndBound_", gndBound_);
  gndPower_ = j.value("gndPower_", gndPower_);
  gndLnc_ = j.value("gndLnc_", gndLnc_);
  gndTailPenaltyStd_ = j.value("gndTailPenaltyStd_", gndTailPenaltyStd_);

  fixRelativetransform_ = j.value("fixRelativetransform_", false);

  const auto sensorOff = j.value("sensor_offset", std::vector<double>{0.0, 0.0, 0.0});
  if (sensorOff.size() != 3) {
    throw std::runtime_error("UTISASlamSystem: sensor_offset must have exactly 3 values [x, y, theta]");
  }
  landmarkSensorOffset_ = SE2(sensorOff[0], sensorOff[1], sensorOff[2]);
}

g2o::ToggelableGNDKernel* UTISASlamSystem::newPriorToggelableGndKernel() {
  return new g2o::ToggelableGNDKernel(
      gndBound_, gndPower_, gndLnc_, gndTailPenaltyStd_, &gndActiveAlwaysFalse_);
}

UTISASlamSystem::~UTISASlamSystem() = default;

void UTISASlamSystem::start() {
  graphChanged_ = false;

  auto* poseFrame = new ParameterSE2Offset();
  poseFrame->setOffset(SE2());
  poseFrame->setId(kPoseFrameParameterId);
  if (!optimizer_->addParameter(poseFrame)) {
    throw std::runtime_error("UTISASlamSystem: failed to register pose-frame ParameterSE2Offset");
  }

  auto* lmSensor = new ParameterSE2Offset();
  lmSensor->setOffset(landmarkSensorOffset_);
  lmSensor->setId(kLandmarkSensorParameterId);
  if (!optimizer_->addParameter(lmSensor)) {
    throw std::runtime_error("UTISASlamSystem: failed to register landmark-sensor ParameterSE2Offset");
  }
}

void UTISASlamSystem::stop() {
  if (verbose_) {
    std::cout << "Bot " << robotId_ << " stop start\n";
  }

  const bool canOptimize = !optimizer_->edges().empty() && optimizer_->vertices().size() > 1;
  if (canOptimize) {
    optimize(optCountStop_);
  }
  if (gndActiveConfig_) {
    gndActive_ = true;
  }
  for (const auto& vertex : optimizer_->vertices()) {
    g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(vertex.second);
    (void)v;
  }
  if (canOptimize) {
    optimize(optCountStopFix_);
  }

  if (preOptTrajectoryDumpEnabled_) {
    std::cout << "\n[Bot " << robotId_ << "] relativeTransforms_ dump @ stop()\n";
    if (relativeTransforms_.empty()) {
      std::cout << "  (none)\n";
    } else {
      for (const auto& kv : relativeTransforms_) {
        const VertexSE2* v = kv.second;
        if (!v) continue;
        const SE2 T = v->estimate();
        std::cout << "  observedRobotId=" << kv.first << " | SE2 t=(" << T.translation().transpose()
                  << ") theta=" << T.rotation().angle() << "\n";
      }
    }
    std::cout << std::endl;
  }

  if (preOptTrajectoryDumpEnabled_) {
    size_t initializedCount = 0;
    for (const auto& obs : observations_) {
      initializedCount += obs.initialized ? 1 : 0;
    }
    std::cout << "  observations_: " << observations_.size() << " | initialized=" << initializedCount
              << std::endl;

    const size_t maxPrint = 8;
    for (size_t i = 0; i < observations_.size() && i < maxPrint; ++i) {
      const auto& obs = observations_[i];
      std::cout << "  Observation[" << i << "] (obsId " << obs.observationId
                << ") initialized=" << (obs.initialized ? "Yes" : "No") << std::endl;
    }
  }
}

void UTISASlamSystem::onAfterOptimize() {
  if (pendingGndPriorEdges_.empty()) {
    return;
  }

  for (auto* e : pendingGndPriorEdges_) {
    if (!e) continue;

    auto* rk = dynamic_cast<g2o::ToggelableGNDKernel*>(e->robustKernel());
    if (rk) {
      rk->setBoolPointer(&gndActive_);
    } else if (verbose_) {
      std::cerr << "[GND] Pending prior edge has no ToggelableGNDKernel; id=" << e->id() << std::endl;
    }
  }
  pendingGndPriorEdges_.clear();
}

void UTISASlamSystem::setPreOptTrajectoryOutputDir(const std::string& output_dir) {
  preOptTrajectoryOutputDir_ = output_dir;
}

void UTISASlamSystem::setPreOptTrajectoryDumpEnabled(bool enabled) {
  preOptTrajectoryDumpEnabled_ = enabled;
}

void UTISASlamSystem::dumpPreOptTrajectory(const std::string& run_directory) {
  if (preOptTrajectoryDumpEnabled_ && !run_directory.empty()) {
    std::filesystem::path out_dir(run_directory);
    std::error_code mkdir_ec;
    std::filesystem::create_directories(out_dir, mkdir_ec);
    if (mkdir_ec) {
      throw std::runtime_error("Cannot create pre-opt trajectory directory '" + run_directory +
                               "': " + mkdir_ec.message());
    }
    const std::string filename = (out_dir / ("trajectory_" + robotId_ + ".txt")).string();
    saveTrajectoryTUM(filename);
  }
}

void UTISASlamSystem::processEvent(Event& event) {
  graphChanged_ = true;
  switch (event.type()) {
    case Event::EventType::Other: {
      auto& utisaEvent = static_cast<UTISAEventBase&>(event);
      switch (utisaEvent.utisaEventType()) {
        case UTISAEventType::Initialization:
          handleInitializationEvent(static_cast<UTISAInitEvent&>(event));
          break;
        case UTISAEventType::Odometry:
          stepNumber_ += 1;
          handleOdometryEvent(static_cast<UTISAOdomEvent&>(event));
          break;
        case UTISAEventType::Observation:
          handleObservationEvent(static_cast<UTISAObsEvent&>(event));
          break;
        case UTISAEventType::LandmarkObservation:
          handleLMObservationEvent(static_cast<UTISALmObsEvent&>(event));
          break;
        default:
          ignoreUnknownEventType();
          break;
      }
      break;
    }
    default:
      if (verbose_) {
        std::cout << " - Unknown Event ..." << std::endl;
      }
      ignoreUnknownEventType();
      break;
  }
}

void UTISASlamSystem::ignoreUnknownEventType() {}

void UTISASlamSystem::handleInitializationEvent(UTISAInitEvent event) {
  if (verbose_) {
    std::cout << " - Bot " << robotId_ << " - SlamSystem handleInitializationEvent start ..." << std::endl;
  }

  currentPlatformVertex_ = new VertexSE2();
  int vid = static_cast<int>(platformVertices_.size());
  currentPlatformVertex_->setId(vid);
  currentPlatformVertex_->setEstimate(event.value);
  vertexStampMap_.add(event.time, vid);

  optimizer_->addVertex(currentPlatformVertex_);
  platformVertices_.emplace_back(currentPlatformVertex_);

  if (event.posFixed) {
    if (verbose_) {
      std::cout << " - Fixing initial vertex ..." << std::endl;
    }
    currentPlatformVertex_->setFixed(true);
  } else {
    if (verbose_) {
      std::cout << " - Adding initialization prior edge ..." << std::endl;
    }
    auto* prior = new EdgePlatformPosePrior();
    prior->setVertex(0, currentPlatformVertex_);
    prior->setMeasurement(event.value.toVector());
    prior->setInformation(event.information);
    prior->setParameterId(0, kPoseFrameParameterId);
    if (!optimizer_->addEdge(prior)) {
      // g2o inserts edge before cache/parameter resolution; reject unresolved prior explicitly.
      optimizer_->removeEdge(prior);
      delete prior;
      currentPlatformVertex_->setFixed(true);
      if (verbose_) {
        std::cerr << "[Init] failed to add prior edge; falling back to fixed initial pose.\n";
      }
    } else {
      currentPlatformVertex_->setFixed(false);
    }
  }
  initialized_ = true;

  if (verbose_) {
    std::cout << " - SlamSystem handleInitializationEvent end ..." << std::endl;
  }
}

void UTISASlamSystem::handleOdometryEvent(UTISAOdomEvent event) {
  if (verbose_) {
    std::cout << " - Bot " << robotId_ << " - SlamSystem handleOdometryEvent start ..." << std::endl;
  }
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

  const double vx = event.velBody.x();
  const double vy = event.velBody.y();
  const double omega = event.velBody.z();

  const SE2 delta(vx * odomDt, vy * odomDt, omega * odomDt);

  Eigen::Matrix3d deltaInfo = event.information;
  deltaInfo *= 1.0 / (odomDt * odomDt);
  const double maxDiag = 9.0e9;
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(deltaInfo(i, i)) || deltaInfo(i, i) <= 0.0) {
      deltaInfo(i, i) = 1.0;
    } else if (deltaInfo(i, i) > maxDiag) {
      deltaInfo(i, i) = maxDiag;
    }
  }

  if (verbose_) {
    std::cout << " - Creating vertex ..." << std::endl;
  }
  currentPlatformVertex_ = new VertexSE2();

  int vid = static_cast<int>(platformVertices_.size());
  currentPlatformVertex_->setId(vid);
  vertexStampMap_.add(event.time, vid);
  optimizer_->addVertex(currentPlatformVertex_);

  platformVertices_.emplace_back(currentPlatformVertex_);

  auto* odometry = new EdgeSE2();
  VertexSE2* v0 = platformVertices_[platformVertices_.size() - 2];
  odometry->setVertex(0, v0);
  odometry->setVertex(1, currentPlatformVertex_);

  currentPlatformVertex_->setEstimate(v0->estimate() * delta);

  odometry->setMeasurement(delta);
  if (!isValidInformationMatrix3(deltaInfo)) {
    deltaInfo = Eigen::Matrix3d::Identity();
  }
  odometry->setInformation(deltaInfo);
  optimizer_->addEdge(odometry);

  currentTime_ = event.time;

  if (verbose_) {
    std::cout << " - SlamSystem handleOdometryEvent end ..." << std::endl;
  }
}

void UTISASlamSystem::handleLMObservationEvent(UTISALmObsEvent event) {
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

  VertexSE2* v0 = platformVertexAtTimeSafe(vertexStampMap_, platformVertices_, event.time);
  if (!v0) {
    std::cerr << "[LM] platform vertex lookup failed at t=" << event.time << std::endl;
    return;
  }

  Eigen::Matrix2d info2 = event.information;
  if (!info2.allFinite()) {
    info2 = Eigen::Matrix2d::Identity();
  }

  const Eigen::Vector2d xy_body(event.range * std::cos(event.bearing),
                                 event.range * std::sin(event.bearing));
  const Eigen::Vector2d p_frame = v0->estimate() * xy_body;

  VertexPointXY* lmVtx = nullptr;
  auto it = landmarks_.find(lmKey);
  if (it == landmarks_.end()) {
    lmVtx = new VertexPointXY();
    lmVtx->setId(40000000 + nextLandmarkVertexSeq_++);
    lmVtx->setEstimate(p_frame);
    optimizer_->addVertex(lmVtx);

    Landmark L;
    L.lmId = lmKey;
    L.landmark = lmVtx;
    L.initialized = true;
    landmarks_[lmKey] = std::move(L);
  } else {
    Landmark& slot = it->second;
    if (!slot.initialized || !slot.landmark) {
      lmVtx = new VertexPointXY();
      lmVtx->setId(40000000 + nextLandmarkVertexSeq_++);
      lmVtx->setEstimate(p_frame);
      optimizer_->addVertex(lmVtx);
      slot.landmark = lmVtx;
      slot.initialized = true;

      for (auto& [observerId, est] : slot.landmarkEsts) {
        (void)observerId;
        if (!est.estimatePriorEdge) continue;
        est.estimatePriorEdge ->setVertex(1, lmVtx);
        if (optimizer_->addEdge(est.estimatePriorEdge)) {
          pendingGndPriorEdges_.push_back(est.estimatePriorEdge);
        }
      }
      graphChanged_ = true;
    } else {
      lmVtx = slot.landmark;
    }
  }
  //std::cout << " - lm " << lmKey << " process over\n";

  auto* edge = new EdgeRangeBearing();
  edge->setVertex(0, v0);
  edge->setVertex(1, lmVtx);
  edge->setMeasurement(Eigen::Vector2d(event.range, event.bearing));
  edge->setInformation(info2);
  edge->setParameterId(0, kLandmarkSensorParameterId);
  optimizer_->addEdge(edge);

  if (verbose_) {
    std::cout << " - handleLMObservationEvent: lmKey=" << lmKey << " edge added\n";
  }
}

void UTISASlamSystem::handleObservationEvent(UTISAObsEvent event) {
  if (verbose_) {
    std::cout << " - SlamSystem handleObservationEvent start for robot: " << robotId_ << std::endl;
  }


  const std::string& observedRobotKey = event.robotIdTo;
  if (relativeTransforms_.find(observedRobotKey) == relativeTransforms_.end()) {
    auto* v = new VertexSE2();
    v->setId(20000000 + nextRelativeTransformVtxId_++);
    v->setEstimate(SE2());
    if (fixRelativetransform_) {
      v->setFixed(true);
    }
    relativeTransforms_[observedRobotKey] = v;
    relativeTransformInGraph_[observedRobotKey] = false;
  }

  VertexSE2* v0 = platformVertexAtTimeSafe(vertexStampMap_, platformVertices_, event.time);
  if (!v0) {
    std::cerr << "[obs] platform vertex lookup failed\n";
    return;
  }

  const Eigen::Vector2d xy_body(event.range * std::cos(event.bearing),
                                 event.range * std::sin(event.bearing));

  auto* observedVtx = new VertexPointXY();
  observedVtx->setEstimate(v0->estimate() * xy_body);
  observedVtx->setId(10000000 + static_cast<int>(observations_.size()));

  auto* observationPrior = new EdgeSE2PointXY();
  observationPrior->setVertex(0, relativeTransforms_[event.robotIdTo]);
  observationPrior->setVertex(1, observedVtx);
  observationPrior->setMeasurement(relativeTransforms_[event.robotIdTo]->estimate() * observedVtx->estimate());
  observationPrior->setInformation(Eigen::Matrix2d::Identity());
  observationPrior->setParameterId(0, kPoseFrameParameterId);
  observationPrior->setRobustKernel(newPriorToggelableGndKernel());

  auto* observation = new EdgeRangeBearing();
  observation->setVertex(0, v0);
  observation->setVertex(1, observedVtx);
  observation->setMeasurement(Eigen::Vector2d(event.range, event.bearing));
  observation->setInformation(event.information);
  observation->setParameterId(0, kLandmarkSensorParameterId);

  observations_.emplace_back(robotId_, event.time, exVtxCount_, event.robotIdTo, observationPrior,
                             observation, observedVtx);
  exVtxCount_++;
  haveUninitializedObs_ = true;
}

UTSIAMessage UTISASlamSystem::broadcastUTSIAMessage() const {
  
  if (verbose_) {
    std::cout << "Bot " << robotId_ << " broadcastUTSIAMessage start\n";
  }
  std::vector<PoseStampEntry> syncReqs;
  syncReqs.reserve(observations_.size());

  for (const auto& obs : observations_) {
    PoseStampEntry entry(obs.observationTime, robotId_, obs.observationId, obs.observedRobotId);
    syncReqs.push_back(std::move(entry));
  }

  return UTSIAMessage(robotId_, false, lmQueryEnabled_, std::move(syncReqs), {});
}

UTSIAMessage UTISASlamSystem::handleObservationSyncRequest(UTSIAMessage& msg) {
  if (verbose_) {
    std::cout << "Bot " << robotId_ << " handleObservationSyncRequest start\n";
  }

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

  std::vector<g2o::OptimizableGraph::Vertex*> landmarkVerticesToMarginalize;
  std::vector<int> landmarkIdsToMarginalize;

  std::vector<std::string> observerIdsToRespondTo;
  if (msg.lm_query) {
    observerIdsToRespondTo.reserve(msg.poseEntries.size());
    for (const auto& pe : msg.poseEntries) {
      if (pe.sourceId == robotId_) continue;
      if (std::find(observerIdsToRespondTo.begin(), observerIdsToRespondTo.end(), pe.sourceId) ==
          observerIdsToRespondTo.end()) {
        observerIdsToRespondTo.push_back(pe.sourceId);
      }
    }
    if (observerIdsToRespondTo.empty()) {
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

  if (localPoseEntries.empty() && landmarkVerticesToMarginalize.empty()) {
    if (verbose_) {
      std::cout << "[REQ] no observations/landmarks to sync for robot " << robotId_ << " – skipping\n";
    }
    return UTSIAMessage(robotId_, false, msg.lm_query, {}, {});
  }

  if (graphChanged_) {
    if (verbose_) {
      std::cout << "Optimizing before marginalization:\n";
    }
    optimize(10);
    graphChanged_ = false;
  }

  std::vector<g2o::OptimizableGraph::Vertex*> verticesToMarginalize;
  verticesToMarginalize.reserve(poseVerticesToMarginalize.size() + landmarkVerticesToMarginalize.size());
  verticesToMarginalize.insert(verticesToMarginalize.end(), poseVerticesToMarginalize.begin(),
                                 poseVerticesToMarginalize.end());
  verticesToMarginalize.insert(verticesToMarginalize.end(), landmarkVerticesToMarginalize.begin(),
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
  if (verbose_) {
    std::cout << "Marginalization success: " << margSuccess << "\n";
  }
  if (!margSuccess) {
    return UTSIAMessage(robotId_, false, msg.lm_query, {}, {});
  }

  std::vector<PoseStampEntry> validPoseResponses;
  validPoseResponses.reserve(localPoseEntries.size());

  for (size_t i = 0; i < localPoseEntries.size(); ++i) {
    auto req = localPoseEntries[i];
    auto* v = dynamic_cast<VertexSE2*>(poseVerticesToMarginalize[i]);
    assert(v && "v to marginalize is nullptr");

    const int vhIdx = v->hessianIndex();
    req.position = v->estimate().translation();
    req.hasPose = true;

    assert(margCov.block(vhIdx, vhIdx) && "Marg block is nullptr");
    const Eigen::Matrix<double, 3, 3>& cov = *margCov.block(vhIdx, vhIdx);
    const Eigen::Matrix2d cov_xy = cov.topLeftCorner<2, 2>();
    Eigen::Matrix2d info2 = cov_xy.inverse();

    if (isValidInformationMatrix2(info2)) {
      req.information = std::move(info2);
      validPoseResponses.push_back(std::move(req));
    } else {
      std::cerr << "Dropping sync for observation " << req.observationId
                << " due to invalid information matrix\n";
    }
  }

  std::vector<LMPoseEntry> validLmResponses;
  if (msg.lm_query) {
    validLmResponses.reserve(landmarkVerticesToMarginalize.size() * observerIdsToRespondTo.size());

    for (size_t k = 0; k < landmarkVerticesToMarginalize.size(); ++k) {
      auto* ptVtx = dynamic_cast<VertexPointXY*>(landmarkVerticesToMarginalize[k]);
      if (!ptVtx) continue;
      const int lmId = landmarkIdsToMarginalize[k];
      const int vhIdx = ptVtx->hessianIndex();

      auto* blk = margCov.block(vhIdx, vhIdx);
      if (!blk) continue;

      if (blk->rows() != 2 || blk->cols() != 2) continue;
      const Eigen::Matrix2d cov2 = *blk;
      Eigen::Matrix2d info2 = cov2.inverse();
      if (!info2.allFinite()) continue;

      const Eigen::Vector2d pos = ptVtx->estimate();

      for (const auto& observerId : observerIdsToRespondTo) {
        LMPoseEntry entry(lmId, observerId);
        entry.hasPose = true;
        entry.position = pos;
        entry.information = info2;
        validLmResponses.push_back(std::move(entry));
      }
    }
  }

  if (verbose_) {
    std::cout << "handleObservationSyncRequest Complete\n";
  }

  return UTSIAMessage(msg.sourceId, true, msg.lm_query, std::move(validPoseResponses),
                   std::move(validLmResponses));
}

void UTISASlamSystem::handleObservationSyncResponse(const UTSIAMessage& message) {
  if (verbose_) {
    std::cout << "Bot " << robotId_ << " handleObservationSyncResponse start\n";
  }
  for (const auto& pe : message.poseEntries) {
    if (pe.sourceId != robotId_) {
      continue;
    }
    if (pe.observationId < 0 || pe.observationId >= static_cast<int>(observations_.size())) {
      std::cerr << "Invalid observationId: " << pe.observationId << std::endl;
      continue;
    }
    if (observations_[pe.observationId].observedRobotId != pe.subjectId) {
      std::cerr << "Mismatched subjectId for observationId " << pe.observationId << std::endl;
      continue;
    }
    EdgeSE2PointXY* priorEdge = observations_[pe.observationId].observationPriorEdge;
    priorEdge->setMeasurement(pe.position);
    priorEdge->setInformation(pe.information);
    graphChanged_ = true;
  }

  if (message.lm_query && !message.lmEntries.empty()) {
    for (const auto& lme : message.lmEntries) {
      if (lme.landmarkId < 0) continue;

      const int lmId = lme.landmarkId;

      auto it = landmarks_.find(lmId);
      if (it == landmarks_.end()) {
        Landmark L;
        L.lmId = lmId;
        L.initialized = false;
        it = landmarks_.emplace(lmId, std::move(L)).first;
      }

      auto& landmark = it->second;

      if (!currentPlatformVertex_) {
        if (verbose_) {
          std::cerr << "[LM sync] currentPlatformVertex_ is null; skipping prior creation.\n";
        }
        continue;
      }

      const std::string& observerId = lme.observerId;
      if (relativeTransforms_.find(observerId) == relativeTransforms_.end()) {
        auto* rv = new VertexSE2();
        rv->setId(20000000 + nextRelativeTransformVtxId_++);
        rv->setEstimate(SE2());
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

      if (!est.estimatePriorEdge && landmark.landmark) {
        est.lmId = lmId;
        est.observerId = observerId;
        est.initialized = true;
        est.estimatePriorEdge = new EdgeSE2PointXY();

        est.estimatePriorEdge->setVertex(0, relativeTransforms_[observerId]);
        est.estimatePriorEdge->setVertex(1, landmark.landmark);
        est.estimatePriorEdge->setParameterId(0, kPoseFrameParameterId);
        est.estimatePriorEdge->setRobustKernel(newPriorToggelableGndKernel());

        if (landmark.initialized) {
          if (optimizer_->addEdge(est.estimatePriorEdge)) {
            pendingGndPriorEdges_.push_back(est.estimatePriorEdge);
          }
        }
      }

      const Eigen::Vector2d lmWorld = lme.position;

      Eigen::Matrix2d info2 = lme.information;
      if (!info2.allFinite()) {
        info2 = Eigen::Matrix2d::Identity();
      }

      if (est.estimatePriorEdge) {
        est.estimatePriorEdge->setMeasurement(lmWorld);
        est.estimatePriorEdge->setInformation(info2);
      }

      graphChanged_ = graphChanged_ || landmark.initialized;
    }
  }

  if (verbose_) {
    std::cout << "step 2 Start\n";
  }
  if (haveUninitializedObs_) {
    bool foundAny = false;
    for (size_t i = 0; i < observations_.size(); i++) {
      if (!observations_[i].initialized) {
        auto wasUpdated = std::any_of(
            message.poseEntries.begin(), message.poseEntries.end(), [&](auto const& req) {
              return req.observationId == static_cast<int>(i) && req.sourceId == robotId_ &&
                     req.subjectId == observations_[i].observedRobotId;
            });
        if (wasUpdated) {
          auto* vertex = observations_[i].observationVertex;
          (void)vertex;
          bool ok = optimizer_->addVertex(observations_[i].observationVertex);

          bool ok2 = optimizer_->addEdge(observations_[i].observationPriorEdge);
          bool ok3 = optimizer_->addEdge(observations_[i].observationEdge);
          if (ok2) {
            pendingGndPriorEdges_.push_back(observations_[i].observationPriorEdge);
          }
          if (preOptTrajectoryDumpEnabled_ && se2PriorDiagPrinted_ < kSe2PriorDiagPrintMax) {
            std::cout << "AddVertex, AddPrior, AddObservation " << ok << " " << ok2 << " " << ok3
                      << std::endl;
            std::cout << "obs.observationVertex->id(): " << observations_[i].observationVertex->id()
                      << std::endl;
            ++se2PriorDiagPrinted_;
          }
          observations_[i].initialized = true;
        } else {
          foundAny = true;
        }
      }
    }
    if (!foundAny) {
      haveUninitializedObs_ = false;
    }
  }
  if (verbose_) {
    std::cout << "handleObservationSyncResponse Complete\n";
  }
}

void UTISASlamSystem::platformEstimate(Eigen::Vector3d& x, Eigen::Matrix3d& P) {
  if (!currentPlatformVertex_) {
    x.setZero();
    P.setZero();
    return;
  }
  x = currentPlatformVertex_->estimate().toVector();
  if (currentPlatformVertex_->fixed()) {
    P.setZero();
    return;
  }
  g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;
  int idx = currentPlatformVertex_->hessianIndex();
  bool success = false;
  try {
    success = optimizer_->computeMarginals(spinv, currentPlatformVertex_);
  } catch (...) {
    P.setZero();
    return;
  }
  if (!success) {
    P.setZero();
    return;
  }
  const auto block = spinv.block(idx, idx);
  if (block) {
    P = block->topLeftCorner<3, 3>();
  } else {
    P.setZero();
  }
}

void UTISASlamSystem::platformEstimate(Eigen::Vector3d& pose) const {
  if (!currentPlatformVertex_) {
    pose.setZero();
    return;
  }
  pose = currentPlatformVertex_->estimate().toVector();
}

std::vector<std::pair<double, SE2>> UTISASlamSystem::getTrajectory() const {
  std::vector<std::pair<double, SE2>> trajectory;

  const auto& stampData = vertexStampMap_.data();

  trajectory.reserve(stampData.size());

  for (const auto& entry : stampData) {
    double timestamp = entry.t;
    int vertexId = entry.id;

    if (vertexId >= 0 && vertexId < static_cast<int>(platformVertices_.size())) {
      VertexSE2* vertex = platformVertices_[vertexId];
      if (vertex) {
        trajectory.emplace_back(timestamp, vertex->estimate());
      }
    }
  }

  return trajectory;
}

void UTISASlamSystem::getRangeBearingObservationSegments(
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& landmarkSegs,
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& robotSegs) const {
  landmarkSegs.clear();
  robotSegs.clear();
  for (const auto& e_kv : optimizer_->edges()) {
    auto* rb = dynamic_cast<EdgeRangeBearing*>(e_kv);
    if (!rb) continue;
    const auto* v0 = dynamic_cast<const VertexSE2*>(rb->vertex(0));
    const auto* v1 = dynamic_cast<const VertexPointXY*>(rb->vertex(1));
    if (!v0 || !v1) continue;
    const Eigen::Vector2d src = v0->estimate().translation();
    const Eigen::Vector2d dst = v1->estimate();
    const int id1 = v1->id();
    if (id1 >= 40000000) {
      landmarkSegs.emplace_back(src, dst);
    } else {
      robotSegs.emplace_back(src, dst);
    }
  }
}

void UTISASlamSystem::saveTrajectoryTUM(const std::string& filename) const {
  std::ofstream out(filename);
  if (!out) {
    throw std::runtime_error("Cannot open file for writing trajectory: " + filename);
  }

  out << std::fixed << std::setprecision(9);

  auto trajectory = getTrajectory();

  for (const auto& [timestamp, pose] : trajectory) {
    const Eigen::Vector2d t = pose.translation();
    const double yaw = pose.rotation().angle();
    Eigen::Quaterniond q(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));

    out << timestamp << " " << t.x() << " " << t.y() << " " << 0.0 << " " << q.x() << " " << q.y()
        << " " << q.z() << " " << q.w() << std::endl;
  }

  out.close();
  std::cout << "Saved trajectory with " << trajectory.size() << " poses to " << filename << std::endl;
}

void UTISASlamSystem::saveLandmarksXY(const std::string& filename) const {
  std::ofstream out(filename);
  if (!out) {
    throw std::runtime_error("Cannot open file for writing landmarks: " + filename);
  }
  out << std::fixed << std::setprecision(9);
  for (const auto& kv : landmarks_) {
    const int lm_id = kv.first;
    const Landmark& lm = kv.second;
    if (!lm.initialized || !lm.landmark) {
      continue;
    }
    const Eigen::Vector2d p = lm.landmark->estimate();
    out << lm_id << " " << p.x() << " " << p.y() << "\n";
  }
  out.close();
  std::cout << "Saved estimated landmarks to " << filename << std::endl;
}

}  // namespace multibotsim
}  // namespace tutorial
}  // namespace g2o
