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

#pragma once

#include <map>
#include <vector>

#include <nlohmann/json.hpp>

#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
// TODO Change to Levenburg
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
// #include "simulator.h"



#include "g2o_tutorial_slam2d_api.h"
#include "se2.h"
#include "events.h"
#include "sensor_data.h"

#include "types_tutorial_slam2d.h"
#include "vertex_point_xy.h"
#include "vertex_se2.h"
#include "edge_se2.h"
#include "edge_se2_wt.h"
#include "edge_se2_pointxy.h"
#include "edge_range_bearing.h"
#include "edge_platform_loc_prior.h"
#include "GNDEdges/edge_platform_loc_prior_gnd.h"
#include "slam_system_base.h"



namespace g2o {
namespace tutorial {


using VertexContainer = g2o::OptimizableGraph::VertexContainer;


typedef BlockSolver<BlockSolverTraits<-1, -1> > SlamBlockSolver;
typedef LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;


class G2O_TUTORIAL_SLAM2D_API SlamSystem : public SlamSystemBase<VertexSE2, EdgeVelocitySE2> {


protected:
  using Base = SlamSystemBase<VertexSE2, EdgeVelocitySE2>;
  //using Base::verbose_;

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
  using Base::optimize;

 public:
  SlamSystem(const std::string& filename);
  ~SlamSystem();

  /**
   * @brief Initialie and start the SLAM system.
   */
  void start() override;


  /**
   * @brief stop the SLAM system and finallise result accumulation
   */
  void stop() override;

  /**
   * @brief set the maximum observations per landmark
   */
  void setMaxObservationsPerLandmark(int maxObservationsPerLandmark);

    /**
   * @brief return the platform estimates
   * @param m list of landmark pose estimates (outputs, empty)
   * @param Pmm list of landmark covariances (outputs, empty) 
   * @param landmarkIds list of landmark ids (outputs, empty)
   */
  void landmarkEstimates(std::vector<Eigen::Vector2d>& m, std::vector<Eigen::Matrix2d>& Pmm, std::vector<int>& landmarkIds);


  void getSceneEstimates(Eigen::Vector3d& x, std::vector<Eigen::Vector2d>& m, std::vector<int>& landmarkIds) const;


  void getSceneEstimatesWithP(Eigen::Vector3d& x, Eigen::Matrix2d& P, std::vector<Eigen::Vector2d>& m, std::vector<Eigen::Matrix2d>& Pmm, std::vector<int>& landmarkIds);
  //function [samples, chi2s] = landmarkSamples(obj, Ns, width)
  //function trajectory = getTrajectory(obj, vxtid)


protected:
  /**
   * @brief process a event
   * @param events event to process
   */
  void processEvent(Event& event) override;


  /**
   * @brief handler for event types that are not yet registered with a handle (unused)
   */
  void ignoreUnknownEventType();

  /**
   * @brief predict the platform location dT in the future
   * @param dT 
   */
  void handlePredictForwards(double dT);

  /**
   * @brief if two events are too close together we don't predict forward and generate new vertex.
   */
  void handleNoPrediction();

  /**
   * @brief event handler for initialisation events
   * @param event
   */
  void handleInitializationEvent(InitializationEvent event);

  /**
   * @brief event handler for odometry events
   * @param event
   */
  void handleUpdateOdometryEvent(OdometryEvent event);

  /**
   * @brief event handler for landmark observation events
   * @param event
   */
  void handleSLAMObservationEvent(LandmarkObservationsEvent event);

    /**
   * @brief event handler for landmark observation events
   * @param event
   */
  void handleRangeBearingObservationEvent(LMRangeBearingObservationsEvent event);

  /**
   * @brief given landmark id, retrieve landmark. Create landmark if landmark not already there
   * @param event
   */
  bool createOrGetLandmark(int id, VertexPointXY*& lmVertex);
  // handlenoUpdate()
  // handleInitializationEvent(event)
  
  // ... all other observations ,,,

  void handleGPSObservationEvent(GPSObservationEvent event);
  




private:
  ParameterSE2Offset* sensorOffset_;

  // Landmark related
  VertexContainer landmarkVertices_;
  std::map<int, int> landmarkIdMap_;
  int maxObservationsPerLandmark_;

  // process model related
  SE2 u_;
  Eigen::Matrix3d sigmaU_;
};

}  // namespace tutorial
}  // namespace g2o
