#pragma once

#include <map>
#include <vector>

#include <nlohmann/json.hpp>

#include "g2o_tutorial_slam2d_api.h"
#include "se2.h"
#include "events_comp.h"
#include "sensor_data.h"

#include "types_tutorial_slam2d.h"
#include "vertex_point_xy.h"
#include "vertex_se2.h"
#include "edge_se2.h"
#include "edge_se2_wt.h"
#include "edge_se2_pointxy.h"
#include "edge_range_bearing.h"
#include "edge_platform_loc_prior.h"
#include "GGDEdges/edge_platform_loc_prior_ggd.h"
#include "slam_system_base.hpp"

namespace g2o {
namespace tutorial {

/** COMP0250 SlamSystem on ::g2o::SlamSystemBase / Multibot stack. */
class G2O_TUTORIAL_SLAM2D_API SlamSystem
    : public ::g2o::SlamSystemBase<VertexSE2, EdgeVelocitySE2> {
 protected:
  using Base = ::g2o::SlamSystemBase<VertexSE2, EdgeVelocitySE2>;

  using Base::stepNumber_;
  using Base::currentTime_;
  using Base::initialized_;
  using Base::componentsReady_;
  using Base::optPeriod_;
  using Base::optCountProcess_;
  using Base::optCountStop_;
  using Base::optCountStopFix_;
  using Base::optimizer_;
  using Base::vertexId_;
  using Base::processModelEdges_;
  using Base::numProcessModelEdges_;
  using Base::unfixedTimeWindow_;
  using Base::currentPlatformVertex_;
  using Base::platformVertices_;
  using Base::optimize;
  using Base::platformEstimate;
  using Base::platformEstimateMarginals;

 public:
  using EstimateType = Base::EstimateType;
  using CovarianceType = Base::CovarianceType;

  explicit SlamSystem(const std::string& filename);
  ~SlamSystem() override;

  void start() override;
  void stop() override;

  void setMaxObservationsPerLandmark(int maxObservationsPerLandmark);

  void landmarkEstimates(std::vector<Eigen::Vector2d>& m,
                         std::vector<Eigen::Matrix2d>& Pmm,
                         std::vector<int>& landmarkIds);

  void getSceneEstimates(Eigen::Vector3d& x, std::vector<Eigen::Vector2d>& m,
                         std::vector<int>& landmarkIds) const;

  /** Pose as (x, y, theta). */
  void platformEstimate(Eigen::Vector3d& x) const;

  /** Pose + top-left 2×2 of the SE2 marginal (same API as SlamSystem / views). */
  void getSceneEstimatesWithP(Eigen::Vector3d& x, Eigen::Matrix2d& P,
                              std::vector<Eigen::Vector2d>& m,
                              std::vector<Eigen::Matrix2d>& Pmm,
                              std::vector<int>& landmarkIds);

 protected:
  void processEvent(Event& event) override;

  void ignoreUnknownEventType();
  void handlePredictForwards(double dT);
  void handleNoPrediction();
  void handleInitializationEvent(InitializationEvent event);
  void handleUpdateOdometryEvent(OdometryEvent event);
  void handleSLAMObservationEvent(LandmarkObservationsEvent event);
  void handleRangeBearingObservationEvent(LMRangeBearingObservationsEvent event);
  bool createOrGetLandmark(int id, VertexPointXY*& lmVertex);
  void handleGPSObservationEvent(GPSObservationEvent event);

 private:
  ParameterSE2Offset* sensorOffset_ = nullptr;

  OptimizableGraph::VertexContainer landmarkVertices_;
  std::map<int, int> landmarkIdMap_;
  int maxObservationsPerLandmark_ = 0;

  SE2 u_;
  Eigen::Matrix3d sigmaU_ = Eigen::Matrix3d::Identity();
};

}  // namespace tutorial
}  // namespace g2o
