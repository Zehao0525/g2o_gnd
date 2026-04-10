#pragma once

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"

#include "g2o_tutorial_slam2d_api.h"
#include "events.h"
#include "utisa_events.h"
#include "gnd_kernel.h"
#include "utisa_messages.hpp"
#include "utisa_stamp_map.hpp"

#include "types_tutorial_slam2d.h"
#include "vertex_point_xy.h"
#include "edge_platform_pose_prior.h"
#include "edge_range_bearing.h"
#include "parameter_se2_offset.h"
#include "GNDEdges/edge_platform_loc_prior_gnd.h"
#include "slam_system_base.h"

namespace g2o {
namespace tutorial {
namespace multibotsim {

using VertexContainer = g2o::OptimizableGraph::VertexContainer;

typedef BlockSolver<BlockSolverTraits<-1, -1> > SlamBlockSolver;
typedef LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

class G2O_TUTORIAL_SLAM2D_API UTISASlamSystem : public SlamSystemBase<VertexSE2, EdgeSE2> {
 protected:
  using Base = SlamSystemBase<VertexSE2, EdgeSE2>;

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

  using Base::x_;
  using Base::currentPlatformVertex_;

  using Base::platformVertices_;

 public:
  using Base::optimize;

 public:
  struct LandmarkEst {
    int lmId = 0;
    std::string observerId;
    EdgeSE2PointXY* estimatePriorEdge = nullptr;
    bool initialized = false;
  };

  struct Landmark {
    int lmId = 0;
    bool initialized = false;
    VertexPointXY* landmark = nullptr;
    std::map<std::string, LandmarkEst> landmarkEsts;
  };

  struct Observation {
    EdgeSE2PointXY* observationPriorEdge;
    EdgeRangeBearing* observationEdge;
    VertexPointXY* observationVertex;
    std::string observerRobotId;
    std::string observedRobotId;
    double observationTime;
    int observationId;
    bool initialized;

    Observation(std::string selfId, double obsTime, int obsId, std::string robotId,
                  EdgeSE2PointXY* obsPriorEdge, EdgeRangeBearing* obsEdge, VertexPointXY* obsVtx)
        : observerRobotId(std::move(selfId)),
          observationPriorEdge(obsPriorEdge),
          observationEdge(obsEdge),
          observationVertex(obsVtx),
          observedRobotId(std::move(robotId)),
          observationTime(obsTime),
          observationId(obsId),
          initialized(false) {}
  };

  UTISASlamSystem(const std::string& id, const std::string& filename);
  ~UTISASlamSystem();

  void platformEstimate(Eigen::Vector3d& x, Eigen::Matrix3d& P);

  void platformEstimate(Eigen::Vector3d& pose) const;

  std::vector<std::pair<double, SE2>> getTrajectory() const;
  void getRangeBearingObservationSegments(std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& landmarkSegs,
                                          std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& robotSegs) const;

  void saveTrajectoryTUM(const std::string& filename) const;
  void saveLandmarksXY(const std::string& filename) const;
  void setPreOptTrajectoryOutputDir(const std::string& output_dir);
  void setPreOptTrajectoryDumpEnabled(bool enabled);

  static void resetPreOptTrajectoryBatchCounter();
  static int takeNextPreOptTrajectoryBatchIndex();

  void dumpPreOptTrajectory(const std::string& run_directory);

  void start() override;

  void stop() override;

  UTSIAMessage broadcastUTSIAMessage() const;

  void handleObservationSyncResponse(const UTSIAMessage& msg);

  UTSIAMessage handleObservationSyncRequest(UTSIAMessage& msg);

  void setLmQueryEnabled(bool enabled) { lmQueryEnabled_ = enabled; }

 protected:
  void processEvent(Event& event) override;

  void ignoreUnknownEventType();

  void handleInitializationEvent(UTISAInitEvent event);

  void handleOdometryEvent(UTISAOdomEvent event);

  void handleObservationEvent(UTISAObsEvent event);

  void handleLMObservationEvent(UTISALmObsEvent event);

  const std::string& getRobotId() const { return robotId_; }

 public:
  bool gndActive_;

 protected:
  bool gndActiveConfig_ = true;

  double gndBound_ = 3.0;
  double gndPower_ = 6.0;
  double gndLnc_ = 1e-3;
  double gndTailPenaltyStd_ = 5.0;

  g2o::ToggelableGNDKernel* newPriorToggelableGndKernel();

  bool lmQueryEnabled_ = true;

  void onAfterOptimize() override;
  bool gndActiveAlwaysFalse_ = false;
  std::vector<g2o::OptimizableGraph::Edge*> pendingGndPriorEdges_;

  bool fixRelativetransform_ = false;

  /// Pose priors (`EdgePlatformPosePrior`) anchor the robot frame in world; keep identity offset here.
  static constexpr int kPoseFrameParameterId = 0;
  /// Range–bearing observations (`EdgeRangeBearing`) use sensor extrinsic; sync priors use `EdgeSE2PointXY`.
  static constexpr int kLandmarkSensorParameterId = 1;
  SE2 landmarkSensorOffset_;

  std::string robotId_;

  StampMap vertexStampMap_;

  int exVtxCount_;
  StampMap externalVertexStampMap_;

  bool haveUninitializedObs_;
  std::vector<Observation> observations_;

  std::map<int, Landmark> landmarks_;
  int nextLandmarkVertexSeq_ = 0;

  std::map<std::string, VertexSE2*> relativeTransforms_;
  std::map<std::string, bool> relativeTransformInGraph_;

  int nextRelativeTransformVtxId_ = 0;

  int obsCount_ = 0;
  bool graphChanged_;

  bool hasLastOdomTime_ = false;
  double lastOdomTime_ = 0.0;
  std::string preOptTrajectoryOutputDir_;
  bool preOptTrajectoryDumpEnabled_ = false;
  static int preOptTrajectoryBatchCounter_;

  int se2PriorDiagPrinted_ = 0;
  static constexpr int kSe2PriorDiagPrintMax = 8;
};

}  // namespace multibotsim
}  // namespace tutorial
}  // namespace g2o
