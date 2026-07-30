#include "slam_system_new.h"

#include <fstream>
#include <stdexcept>

#include "g2o/core/sparse_block_matrix.h"

namespace g2o {
namespace tutorial {

using namespace Eigen;

SlamSystemNew::SlamSystemNew(const std::string& filename)
    : Base(filename) {
  std::ifstream f(filename);
  if (!f) {
    throw std::runtime_error("Cannot open SLAM config file: " + filename);
  }
  nlohmann::json j;
  f >> j;

  const auto offset = j.value("sensor_offset", std::vector<double>{0.0, 0.0, 0.0});
  if (offset.size() != 3) {
    throw std::runtime_error("sensor_offset must be size 3");
  }
  const SE2 sensorOffsetTransf(offset[0], offset[1], offset[2]);
  sensorOffset_ = new ParameterSE2Offset();
  sensorOffset_->setOffset(sensorOffsetTransf);
  sensorOffset_->setId(0);

  optimizer_->addParameter(sensorOffset_);
  landmarkIdMap_.clear();
}

SlamSystemNew::~SlamSystemNew() = default;

void SlamSystemNew::start() {
  if (!componentsReady_) {
    componentsReady_ = true;
  }
}

void SlamSystemNew::stop() {
  optimize(optCountStop_);
  for (const auto& vertex : optimizer_->vertices()) {
    auto* v = static_cast<OptimizableGraph::Vertex*>(vertex.second);
    if (v->id() > 0) {
      v->setFixed(false);
    }
  }
  optimize(optCountStopFix_);
}

void SlamSystemNew::setMaxObservationsPerLandmark(int maxObservationsPerLandmark) {
  maxObservationsPerLandmark_ = maxObservationsPerLandmark;
}

void SlamSystemNew::landmarkEstimates(std::vector<Vector2d>& m,
                                      std::vector<Matrix2d>& Pmm,
                                      std::vector<int>& landmarkIds) {
  m.clear();
  Pmm.clear();
  landmarkIds.clear();

  SparseBlockMatrix<MatrixX> spinv;
  optimizer_->computeMarginals(spinv, landmarkVertices_);

  const int numBlocks = static_cast<int>(spinv.rowBlockIndices().size());
  for (int i = 0; i < numBlocks; ++i) {
    const Eigen::MatrixXd* block = spinv.block(i, i);
    Vector2d est;
    landmarkVertices_[static_cast<size_t>(i)]->getEstimateData(est);
    m.emplace_back(est);
    landmarkIds.emplace_back(landmarkVertices_[static_cast<size_t>(i)]->id());
    if (block) {
      if (block->rows() < 2 || block->cols() < 2) {
        throw std::runtime_error("Block at (" + std::to_string(i) + "," +
                                 std::to_string(i) + ") is too small.");
      }
      Pmm.emplace_back(block->topLeftCorner<2, 2>());
    } else {
      Pmm.emplace_back(Eigen::Matrix2d::Zero());
    }
  }
}

void SlamSystemNew::getSceneEstimates(Eigen::Vector3d& x,
                                      std::vector<Eigen::Vector2d>& m,
                                      std::vector<int>& landmarkIds) const {
  m.clear();
  landmarkIds.clear();
  x = currentPlatformVertex_->estimate().toVector();
  for (auto* lm : landmarkVertices_) {
    Vector2d est;
    lm->getEstimateData(est);
    m.emplace_back(est);
    landmarkIds.emplace_back(lm->id());
  }
}

void SlamSystemNew::platformEstimate(Eigen::Vector3d& x) const {
  EstimateType pose;
  Base::platformEstimate(pose);
  x = pose.toVector();
}

void SlamSystemNew::getSceneEstimatesWithP(Eigen::Vector3d& x, Eigen::Matrix2d& P,
                                           std::vector<Eigen::Vector2d>& /*m*/,
                                           std::vector<Eigen::Matrix2d>& /*Pmm*/,
                                           std::vector<int>& /*landmarkIds*/) {
  optimize(20);
  EstimateType pose;
  CovarianceType fullP;
  platformEstimateMarginals(pose, fullP);
  x = pose.toVector();
  P = fullP.topLeftCorner<2, 2>();
}

void SlamSystemNew::processEvent(Event& event) {
  const double dT = event.time - currentTime_;
  if (dT < 1e-3) {
    handleNoPrediction();
  } else {
    handlePredictForwards(dT);
    currentTime_ = event.time;
    stepNumber_ += 1;
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

void SlamSystemNew::ignoreUnknownEventType() {}

void SlamSystemNew::handlePredictForwards(double dT) {
  const SE2 lastpredX = currentPlatformVertex_->estimate();
  const SE2 newX = lastpredX * (u_ * dT);

  currentPlatformVertex_ = new VertexSE2;
  currentPlatformVertex_->setEstimate(newX);
  currentPlatformVertex_->setId(++vertexId_);
  optimizer_->addVertex(currentPlatformVertex_);
  platformVertices_.emplace_back(currentPlatformVertex_);

  auto* odometry = new EdgeVelocitySE2(dT);
  odometry->setVertex(0, platformVertices_[platformVertices_.size() - 2]);
  odometry->setVertex(1, currentPlatformVertex_);
  odometry->setMeasurement(u_);
  assert(odometry->information().rows() == 3);
  odometry->setInformation(sigmaU_.inverse());
  optimizer_->addEdge(odometry);

  processModelEdges_.emplace_back(odometry);
  numProcessModelEdges_ += 1;
}

void SlamSystemNew::handleNoPrediction() {}

void SlamSystemNew::handleInitializationEvent(InitializationEvent event) {
  currentPlatformVertex_ = new VertexSE2;
  currentPlatformVertex_->setId(++vertexId_);
  currentPlatformVertex_->setEstimate(event.pose);
  optimizer_->addVertex(currentPlatformVertex_);
  platformVertices_.emplace_back(currentPlatformVertex_);
  currentPlatformVertex_->setFixed(true);

  u_ = event.velocity;
  sigmaU_ = event.sigmaU;
  initialized_ = true;
}

void SlamSystemNew::handleUpdateOdometryEvent(OdometryEvent event) {
  u_ = event.value;
  sigmaU_ = event.covariance;
}

void SlamSystemNew::handleSLAMObservationEvent(LandmarkObservationsEvent event) {
  const SE2 curvtxEst = currentPlatformVertex_->estimate();
  for (const auto& lmObs : event.landmarkObservations) {
    assert(lmObs.value.size() == 2);
    assert(lmObs.covariance.rows() == 2 && lmObs.covariance.cols() == 2);
    VertexPointXY* lmVertex = nullptr;
    const bool vtxCreated = createOrGetLandmark(lmObs.landmark_id, lmVertex);
    if (vtxCreated) {
      lmVertex->setEstimate(curvtxEst * lmObs.value);
    }

    auto* landmarkObservation = new EdgeSE2PointXY;
    landmarkObservation->setVertex(0, currentPlatformVertex_);
    landmarkObservation->setVertex(1, lmVertex);
    landmarkObservation->setMeasurement(lmObs.value);
    landmarkObservation->setInformation(lmObs.covariance.inverse());
    landmarkObservation->setParameterId(0, sensorOffset_->id());
    optimizer_->addEdge(landmarkObservation);
  }
}

void SlamSystemNew::handleRangeBearingObservationEvent(
    LMRangeBearingObservationsEvent event) {
  const Vector3d curvtxEst = currentPlatformVertex_->estimate().toVector();
  for (const auto& lmObs : event.landmarkObservations) {
    assert(lmObs.value.size() == 2);
    assert(lmObs.covariance.rows() == 2 && lmObs.covariance.cols() == 2);
    VertexPointXY* lmVertex = nullptr;
    const bool vtxCreated = createOrGetLandmark(lmObs.landmark_id, lmVertex);
    if (vtxCreated) {
      const double trueBearing = lmObs.value[1] + curvtxEst[2];
      const Vector2d disp(lmObs.value[0] * cos(trueBearing) + curvtxEst[0],
                         lmObs.value[0] * sin(trueBearing) + curvtxEst[1]);
      lmVertex->setEstimate(disp);
    }

    auto* landmarkObservation = new EdgeRangeBearing;
    landmarkObservation->setVertex(0, currentPlatformVertex_);
    landmarkObservation->setVertex(1, lmVertex);
    landmarkObservation->setMeasurement(lmObs.value);
    landmarkObservation->setInformation(lmObs.covariance.inverse());
    landmarkObservation->setParameterId(0, sensorOffset_->id());
    optimizer_->addEdge(landmarkObservation);
  }
}

bool SlamSystemNew::createOrGetLandmark(int id, VertexPointXY*& lmVertex) {
  const auto it = landmarkIdMap_.find(id);
  if (it != landmarkIdMap_.end()) {
    lmVertex = static_cast<VertexPointXY*>(landmarkVertices_[static_cast<size_t>(it->second)]);
    return false;
  }

  lmVertex = new VertexPointXY();
  lmVertex->setId(++vertexId_);
  optimizer_->addVertex(lmVertex);
  landmarkVertices_.push_back(lmVertex);
  landmarkIdMap_[id] = static_cast<int>(landmarkVertices_.size()) - 1;
  return true;
}

void SlamSystemNew::handleGPSObservationEvent(GPSObservationEvent event) {
  auto* gpsObservation = new EdgePlatformLocPriorGND;
  gpsObservation->setVertex(0, currentPlatformVertex_);
  gpsObservation->setMeasurement(event.value);
  gpsObservation->gndSetInformation(event.covariance.inverse(), 8);
  gpsObservation->setParameterId(0, sensorOffset_->id());
  optimizer_->addEdge(gpsObservation);
}

}  // namespace tutorial
}  // namespace g2o
