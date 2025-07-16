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

#include "incremental_simulator.h"

#include <cmath>
#include <iostream>
#include <map>
#include <fstream>
#include <iomanip>

#include "g2o/stuff/sampler.h"
using namespace std;

namespace g2o {
namespace tutorial {

using namespace Eigen;

thread_local std::unique_ptr<GaussianSampler<Vector3d, Matrix3d>> IncrementalSimulator::odomSampler_;

IncrementalSimulator::IncrementalSimulator() = default;

IncrementalSimulator::~IncrementalSimulator() = default;


SE2 IncrementalSimulator::xTrue() const{
  return x_;
}

void IncrementalSimulator::history(std::vector<double> &timeHistory, std::vector<SE2> & xTrueHistory) const{
  timeHistory = timeStore_;
  xTrueHistory = xTrueStore_;
}

void IncrementalSimulator::start(){
  //start@ebe.core.EventBasedSimulator(obj);
  if(verbose_){std::cout << " - Starting IncrementalSimulator ... " << std::endl;}
  if(verbose_){std::cout << " - Creating parameters ... " << std::endl;}
  //obj.platformController.start();
  currentTime_ = 0;
  stepNumber_ = 0;

  carryOnRunning_ = true;
  initialized_ = false;

  //% Create the queue for generating the events
  //obj.eventGeneratorQueue = ebe.core.detail.EventGeneratorQueue();

  // % Set the noise scale
  noiseScale_ = 1;

  // We will set maximum step number to 4000
  //obj.timeStore.resize(4000);
  //obj.xTrueStore.resize(4000);

  // HARDCODE TODO
  maximumStepNumber_ = 4000;
  if(verbose_){std::cout << "   - Creating sigmaUSqrtm_ ... " << std::endl;}
  sigmaUSqrtm_.fill(0.);
  sigmaUSqrtm_(0, 0) = 0.2;
  sigmaUSqrtm_(1, 1) = 0.1;
  sigmaUSqrtm_(2, 2) = deg2rad(1);
  if(verbose_){std::cout << "   - Creating sigmaU_ ... " << std::endl;}
  sigmaU_ = sigmaUSqrtm_ * sigmaUSqrtm_;
  if(verbose_){std::cout << "   - Setting odomSampler_ ... " << std::endl;}
  if (!odomSampler_) {
    if(verbose_){std::cout << "   - odomSampler_ not initialised, initialising ... " << std::endl;}
    odomSampler_ = std::make_unique<GaussianSampler<Vector3d, Matrix3d>>();
  }
  odomSampler_->setDistribution(sigmaU_);


  //obj.eventGeneratorQueue.insert(0, @obj.initialize);

  //Get the landmarks
  // slamLandmarks = obj.scenario.landmarks.slam;

  // % Just handle random case for now
  // if (strcmp(slamLandmarks.configuration, 'random') == true)
  //     lms = [slamLandmarks.x_min;slamLandmarks.y_min] + ...
  //         [slamLandmarks.x_max-slamLandmarks.x_min;slamLandmarks.y_max-slamLandmarks.y_min] .* ...
  //         rand(2, slamLandmarks.numLandmarks);
  // else
  //     lms = slamLandmarks.landmarks;
  // end

  //[[15,15],[35,35],[15,35],[35,15]]

  if(verbose_){std::cout << " - Creating landmarks ... " << std::endl;}
  // Generate LMs like this as of now
  // HARDCODE
  std::vector<Eigen::Vector2d> lm_locs = {
    {15.0, 15.0},
    {35.0, 35.0},
    {15.0, 35.0},
    {35.0, 15.0}
  };

  int id_counter = 0;
  for (const auto& loc : lm_locs) {
    auto* lm = new Landmark();       // allocate Landmark on heap
    lm->id = id_counter++;
    lm->truePose = loc;
    lm->simulatedPose = loc;         // if you want to match the truePose initially
    landmarks_.push_back(lm);
  }


  // HARDCODE TODO
  Matrix2d lmrbObsSigma;
  lmrbObsSigma.fill(0);
  lmrbObsSigma(0,0) = 1;
  lmrbObsSigma(1,1) = 0.1;


  if(verbose_){std::cout << " - Creating event queue ... " << std::endl;}

  eventQueue_ = OrderedEventQueue();

  if(verbose_){std::cout << " - Creating system model ... " << std::endl;}
  systemModel_ = std::make_unique<SystemModel>(true, lmrbObsSigma, 25, Matrix2d());


  if(verbose_){std::cout << " - Creating platform comtroller ... " << std::endl;}
  // HARDCODE
  platformController_ = PlatformController();
  platformController_.setControllerParams(1,10,0.5,20,20,0.2,0.25,false);
  std::vector<Eigen::Vector2d> waypoints;
  waypoints.emplace_back(50.0, 0.0);
  waypoints.emplace_back(50.0, 50.0);
  waypoints.emplace_back(0.0, 50.0);
  waypoints.emplace_back(0.0, 0.0);
  waypoints.emplace_back(25.0, 0.0);
  platformController_.setWaypoints(waypoints);

  // NOTE for syncronouse Simulator only
  odomPeriod_ = 1;
  slamObsPeriod_ = 5;

  if(verbose_){std::cout << " - Trigger initialization event ... " << std::endl;}
  initialize();


  if(verbose_){std::cout << " - IncrementalSimulator start() complete ... " << std::endl;}
}


std::vector<EventPtr> IncrementalSimulator::aquireEvents(){
    std::vector<EventPtr> events = eventQueue_.orderedEvents();
    eventQueue_.clear();
    return events;
  }



//Scenario IncrementalSimulator::getScenario() const;

bool IncrementalSimulator::keepRunning() const{
  return ((carryOnRunning_) && (stepNumber_ <= maximumStepNumber_));
}

void IncrementalSimulator::step(){
  // increment step
  if(verbose_){std::cout << " - SlamSystem step() ..."<< std::endl;}
  stepNumber_ += 1;
  currentTime_ += dT_;

  if(verbose_){std::cout << " - Predicting forward ..."<< std::endl;}
  handlePredictForwards(dT_);

  if(stepNumber_ % odomPeriod_ == 0){
    if(verbose_){std::cout << " - Updating Odom ..."<< std::endl;}
    updateOdometry();
  }
  if(stepNumber_ % odomPeriod_ == 0){
    if(verbose_){std::cout << " - Predicting SLAM Obs ..."<< std::endl;}
    predictSLAMObservations();
  }

  // We need to have an outgoing events queue, where the estimator can aquire events from there.
  std::cout << "- Storing Step Result ..."<< std::endl;
  storeStepResults();
  std::cout << "- SSLAMSystem step() complete" << std::endl;
  
}

void IncrementalSimulator::handlePredictForwards(double dT){
  x_ = systemModel_->predictState(x_, u_, dT);
}

void IncrementalSimulator::initialize(){
  // Initialize the ground truth state
  // Here we manually set x0 and P0 to 0s
  if(verbose_){std::cout << "   - initialize() start ... " << std::endl;}
  Matrix3d P0;
  P0.fill(0);

  SE2 x0 = SE2(0,0,0);
  x_ = x0;
  initialized_ = true;
  if(verbose_){std::cout << "   - Creating initialization event ... " << std::endl;}
  InitializationEvent initEvent = InitializationEvent(currentTime_, x0, SE2(0,0,0), P0, Matrix3d::Zero());
  if(verbose_){std::cout << "   - Push event to queue ... " << std::endl;}
  eventQueue_.push(std::make_shared<InitializationEvent>(initEvent));
  if(verbose_){std::cout << "   - initialize() complete " << std::endl;}
}
// We consult the system model and we give an event.




void IncrementalSimulator::updateOdometry(){
  // Initialize the ground truth state
  // Here we manually set x0 and P0 to 0s

  // Implement Controller

  if (platformController_.off()){
      carryOnRunning_ = false;
      return;
  }

  u_ = platformController_.computeControlInputs(x_);

  SE2 u = u_ + (noiseScale_ * (odomSampler_->generateSample()));      

  OdometryEvent odomEvent = OdometryEvent(currentTime_, u_, sigmaU_);
  eventQueue_.push(std::make_shared<OdometryEvent>(odomEvent));
}




//void predictGPSObservation();
//void predictCompassObservation();
//void predictBearingObservations();
void IncrementalSimulator::predictSLAMObservations() {

  // Create and push event
  if(verbose_){std::cout << "   - predictSLAMObservations() start ..."<< std::endl;}
  if(verbose_){std::cout << "   - querying system model for lm observation vector ..."<< std::endl;}
  LandmarkObservationVector lmObsVec = systemModel_->predictSLAMObservations(x_, landmarks_);
  if(verbose_){std::cout << "   - creating lmObsEvent ..."<< std::endl;}
  auto lmObsEvent = std::make_shared<LandmarkObservationsEvent>(currentTime_, lmObsVec);
  eventQueue_.push(lmObsEvent);
  if(verbose_){std::cout << "   - predictSLAMObservations() complete"<< std::endl;}
}

//void generateHeartbeat();
void IncrementalSimulator::storeStepResults(){
  timeStore_.emplace_back(currentTime_);
  xTrueStore_.emplace_back(x_);
}


}  // namespace tutorial
}  // namespace g2o
