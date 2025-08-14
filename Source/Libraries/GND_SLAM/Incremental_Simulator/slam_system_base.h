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



namespace g2o {
namespace tutorial {


using VertexContainer = g2o::OptimizableGraph::VertexContainer;


typedef BlockSolver<BlockSolverTraits<-1, -1> > SlamBlockSolver;
typedef LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

template <typename VertexType, typename EdgeType>
class G2O_TUTORIAL_SLAM2D_API SlamSystemBase {

 public:
  SlamSystemBase(const std::string& filename):currentTime_(0), stepNumber_(0), lastOptStep_(-1), initialized_(false), componentsReady_(false),
                vertexId_(-1), numProcessModelEdges_(0), currentPlatformVertex_(nullptr){
    optimizer_ = std::make_unique<SparseOptimizer>();
    
    std::ifstream f(filename);
    if (!f) {
          throw std::runtime_error("Cannot open SLAM config file: " + filename);
      }
      nlohmann::json j;
      f >> j;

    verbose_ = j.value("verbose", false);
    if(verbose_){std::cout<<"- Reading all other parameters."<<std::endl;}
    optPeriod_ = j.value("optimization_period", 100);
    optimizationAlg_ = j.value("optimization_algorithm", "GaussNewton");
    if(verbose_){std::cout<<"- optimizationAlg_ = " << optimizationAlg_ <<std::endl;}
    optCountProcess_ = j["optimize_count"].value("process", 10);
    optCountStop_ = j["optimize_count"].value("stop", 10);
    optCountStopFix_ = j["optimize_count"].value("stop_fixed", 10);
    setupOptimizer();
  }
  ~SlamSystemBase()= default;


  void checkTypeRegistration(){
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

  /**
   * @brief optmize graph and store optimsation result
   */
  int optimize(int maximumNumberOfOptimizationSteps){
    if(verbose_){optimizer_->setVerbose(true);}
    optimizer_->initializeOptimization();
    int numIterations = optimizer_->optimize(maximumNumberOfOptimizationSteps);
    if(verbose_){
      std::cerr << "Final chi2: " << optimizer_->activeChi2() << std::endl;
      std::cerr << "Num Iterations: " << numIterations << std::endl;
      std::cout << "Number of vertices: " << optimizer_->vertices().size() << std::endl;
      std::cout << "Number of edges: " << optimizer_->edges().size() << std::endl;
    }
    return numIterations;
    // Add performance data?
  }

  /**
   * @brief optmize graph and store optimsation result
   */
  SparseOptimizer* optimizer() {
    return optimizer_.get();
  }


  // void setMaxObservationsPerLandmark(int maxObservationsPerLandmark);
  void setFixOlderPlatformVertices(double unfixedTimeWindow){
    unfixedTimeWindow_ = unfixedTimeWindow;
  }

  /**
   * @brief return the platform estimates
   * @param x platform pose estimate
   * @param P platform pose estimate covariance
   */
  void platformEstimate(Eigen::Vector3d& x, Eigen::Matrix2d& P){
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
    SparseBlockMatrix<MatrixX> spinv;
    if(verbose_){std::cout << " - Optimizer Computing Marginals ..." << std::endl;}


    std::cout << " - currentPlatformVertex_->fixed() " << currentPlatformVertex_->fixed() << std::endl;
    x = (currentPlatformVertex_->estimate()).toVector();
    if (currentPlatformVertex_->fixed()) {
      if (verbose_) {
        std::cout << " - Current vertex is fixed. Skipping marginal computation." << std::endl;
      }
      P.setZero();
      return;
    }


    int idx = currentPlatformVertex_->hessianIndex();
    bool success = false;

    try {
      if (verbose_) { std::cout << " - Compute Marginals..." << std::endl; }
      success = optimizer_->computeMarginals(spinv, currentPlatformVertex_);

      if (!success) {
        if (verbose_) { std::cout << " - Marginal computation reported failure." << std::endl; }
        P.setZero();
        return;
      }
    } catch (const std::exception& e) {
      std::cerr << "[computeMarginals] Exception: " << e.what() << std::endl;
      P.setZero();
      return;
    } catch (...) {
      std::cerr << "[computeMarginals] Unknown error (not a std::exception)." << std::endl;
      P.setZero();
      return;
    }



    if (!success) {
      if (verbose_) {std::cout << " - Optimization failed" << std::endl;}
      P.setZero();
      return;
    }

    if(verbose_){std::cout << " - currentPlatformVertex_->hessianIndex():" << idx << ", Compute margianl success - " << success << std::endl;}
    if(verbose_){std::cout << " - Assigning to x and P ..." << std::endl;}
    const auto block = spinv.block(idx, idx);
    if(block){
        if(verbose_){std::cout << "Covariance block:\n" << block->topLeftCorner<3,3>() << std::endl;}
        P = block->topLeftCorner<2,2>();
    } else {
        if(verbose_){std::cout << "   - WARNING: Marginal block is null (probably vertex is fixed), setting P to zero." << std::endl;}
        P.setZero();
    }
    if(verbose_){std::cout << " - SlamSystem platformEstimate end ..." << std::endl;}
  }

  void platformEstimate(Eigen::Vector3d& x){
    x = (currentPlatformVertex_->estimate()).toVector();
  }

  /**
   * @brief process an event vector
   * @param events Input the events into the slam_system
   */
  void processEvents(EventPtrVector& events){
    if(verbose_){std::cout << " - SlamSystem processEvents start ..." << std::endl;}
    for (const auto& event : events) {
      if (!event && verbose_) {
        std::cerr << "⚠️ Warning: Null event pointer encountered, skipping." << std::endl;
      }
      processEvent(*event);
    }
    // TODO improve this part
    //if(verbose_){std::cout<<"SLAM system Step: "<< stepNumber_ << ", " << lastOptStep_ << ", "<< optPeriod_  <<std::endl;        }
    if(stepNumber_ == 0 || lastOptStep_ + optPeriod_ <= stepNumber_){
      if(verbose_){std::cout << " - SlamSystem processEvents optimize ..." << std::endl;}
      optimize(optCountProcess_);
      lastOptStep_ = stepNumber_;
      if(verbose_){std::cout << " - SlamSystem processEvents optimize complete ..." << std::endl;}
    }
  }


  /**
   * @brief set verbose
   * @param verbose verbose
   */
  void setVerbose(bool verbose){verbose_ = verbose;}


  /**
   * @brief calls save on the optimizer
   * @param fileName verbose
   */
  void saveOptimizerResults(std::string fileName){
    optimizer_->save(fileName.c_str());
  }


  /**
   * @brief Initialie and start the SLAM system.
   */
  virtual void start() = 0;


  /**
   * @brief stop the SLAM system and finallise result accumulation
   */
  virtual void stop() = 0;


  //void landmarkEstimates(std::vector<Eigen::Vector2d>& m, std::vector<Eigen::Matrix2d>& Pmm, std::vector<int>& landmarkIds);
  //void getSceneEstimates(Eigen::Vector3d& x, std::vector<Eigen::Vector2d>& m, std::vector<int>& landmarkIds) const;
  //void getSceneEstimatesWithP(Eigen::Vector3d& x, Eigen::Matrix2d& P, std::vector<Eigen::Vector2d>& m, std::vector<Eigen::Matrix2d>& Pmm, std::vector<int>& landmarkIds);



protected:

  /**
   * @brief As of now this actually also resets the optmizer.
   */
  void setupOptimizer(){
    if(verbose_){std::cout<<"- Setup Optimizer ..." <<std::endl;}

    auto linearSolver = std::make_unique<LinearSolverEigen<BlockSolver<BlockSolverTraits<-1, -1>>::PoseMatrixType>>();
    if (optimizationAlg_ == "GaussNewton") {
        optimizer_->setAlgorithm(new OptimizationAlgorithmGaussNewton(
            std::make_unique<BlockSolver<BlockSolverTraits<-1, -1>>>(std::move(linearSolver))
        ));
    }
    else if (optimizationAlg_ == "LevenbergMarquardt") {
        optimizer_->setAlgorithm(new OptimizationAlgorithmLevenberg(
            std::make_unique<BlockSolver<BlockSolverTraits<-1, -1>>>(std::move(linearSolver))
        ));
    }
    else {
        throw std::runtime_error("Unknown optimization_algorithm: " + optimizationAlg_);
    }
    platformVertices_.clear();
    processModelEdges_.clear();
  }

  /**
   * @brief process a event
   * @param events event to process
   */
  virtual void processEvent(Event& event) = 0;


  // void ignoreUnknownEventType();
  // void handlePredictForwards(double dT);
  // void handleNoPrediction();
  // void handleInitializationEvent(InitializationEvent event);
  // void handleUpdateOdometryEvent(OdometryEvent event);
  // void handleSLAMObservationEvent(LandmarkObservationsEvent event);
  // void handleRangeBearingObservationEvent(LMRangeBearingObservationsEvent event);
  // bool createOrGetLandmark(int id, VertexPointXY*& lmVertex);
  // void handleGPSObservationEvent(GPSObservationEvent event);
  




protected:
  bool verbose_;

  int stepNumber_;
  int lastOptStep_;
  double currentTime_;
  bool initialized_;
  bool componentsReady_ = false;

  int optPeriod_;
  std::string optimizationAlg_;
  int optCountProcess_;
  int optCountStop_;
  int optCountStopFix_;


  //static thread_local std::unique_ptr<SparseOptimizer> optimizer_;
  std::unique_ptr<SparseOptimizer> optimizer_;

  std::vector<VertexType*> platformVertices_;

  int vertexId_;

  std::vector<EdgeType*> processModelEdges_;
  int numProcessModelEdges_;
  int unfixedTimeWindow_;

  VertexType* currentPlatformVertex_;


  SE2 x_;
};

//template <typename VertexType, typename EdgeType>
//std::unique_ptr<SparseOptimizer> SlamSystemBase<VertexType, EdgeType>::optimizer_ = nullptr;

}  // namespace tutorial
}  // namespace g2o
