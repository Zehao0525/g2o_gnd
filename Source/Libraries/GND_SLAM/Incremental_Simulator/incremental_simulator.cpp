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

void forceLinkTypesTutorialSlam2d();

using namespace Eigen;

thread_local std::unique_ptr<GaussianSampler<Vector3d, Matrix3d>> IncrementalSimulator::odomSampler_;

IncrementalSimulator::IncrementalSimulator(const std::string& filename):
      currentTime_(0), stepNumber_(0), carryOnRunning_(true), initialized_(false){
  std::ifstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open Simulator config file: " + filename);
    }
    nlohmann::json j;
    f >> j;

    verbose_ = j.value("verbose", false);
    if(verbose_){std::cout<<"- SlamSystem Created, verbose_ = true."<<std::endl;}
    if(verbose_){std::cout<<"- Reading all other parameters."<<std::endl;}
    maximumStepNumber_ = j.value("max_steps", 4000);
    if(verbose_){std::cout<<"- maximumStepNumber_ = " << maximumStepNumber_ <<std::endl;}

    auto noise = j["platform"]["noise"].value("sigmaU", std::vector<double>{0.2, 0.1, 1.0});
    sigmaUSqrtm_.setZero();
    sigmaUSqrtm_(0, 0) = noise[0];
    sigmaUSqrtm_(1, 1) = noise[1];
    sigmaUSqrtm_(2, 2) = (j["platform"]["noise"].value("deg_to_rad", true)) ? deg2rad(noise[2]) : noise[2];
    sigmaU_ = sigmaUSqrtm_ * sigmaUSqrtm_.transpose();
    if(verbose_){std::cout<<"- sigmaU set" <<std::endl;}

    // TODO change 
    odomPeriod_ = j.value("odom_period", 1);

    bool have_slam_obs = j["sensors"]["landmark_relative_location"].value("on", false);
    slamObsPeriod_ =  have_slam_obs ? j["sensors"]["landmark_relative_location"].value("measurment_period", 5) : maximumStepNumber_+2;


    bool have_rb_obs = j["sensors"]["landmark_range_bearing"].value("on", false);
    rangBearingObsPeriod_ = have_rb_obs ? j["sensors"]["landmark_range_bearing"].value("measurment_period", 5) : maximumStepNumber_+2;

    if(verbose_){std::cout<<"- odomPeriod_ = " << odomPeriod_ << ", slamObsPeriod_ = " << slamObsPeriod_ <<std::endl;}

    if(verbose_){std::cout<<"- Adding landmarks ..."<<std::endl;}
    landmarks_.clear();
    int id_counter = 0;
    for (const auto& lm : j["landmarks"]) {
        auto* l = new Landmark();
        l->id = id_counter++;
        l->truePose = Eigen::Vector2d(lm[0], lm[1]);
        l->simulatedPose = l->truePose;
        landmarks_.push_back(l);
    }




    bool have_gps_obs = j["sensors"]["gps"].value("on", false);
    gpsObsPeriod_ = have_gps_obs ? j["sensors"]["gps"].value("measurment_period", 5) : maximumStepNumber_+2;





    bool addNoise = j.value("perturb_with_noise", true);
    if(addNoise){
      noiseScale_  =1;
    }
    else{
      noiseScale_ = 0;
    }
    if(verbose_){std::cout<<"- perturb_with_noise = " << addNoise <<std::endl;}




    if(verbose_){std::cout<<"- creating system model ... " << addNoise <<std::endl;}
    systemModel_ = std::make_unique<SystemModel>(j);


    if(verbose_){std::cout<<"- creating platform controller ... " << addNoise <<std::endl;}
    platformController_ = std::make_unique<PlatformController>();
    platformController_->setControllerParams(j["platform"].value("min_speed",1.0),
                                            j["platform"].value("max_speed",10.0),
                                            j["platform"].value("max_accel",0.5),
                                            j["platform"].value("max_delta", 20),
                                            j["platform"].value("max_delta_rate", 20),
                                            j["platform"].value("odom_update_period",0.2),
                                            j["platform"].value("B",0.25),
                                            j["platform"].value("repeat", false));
    std::vector<Eigen::Vector2d> waypoints;
    for (const auto& wp : j["waypoints"]) {
        waypoints.emplace_back(wp[0], wp[1]);
    }
    platformController_->setWaypoints(waypoints);



    if(verbose_){std::cout<<"- creating odom sampler" <<std::endl;}
    if (!odomSampler_) odomSampler_ = std::make_unique<GaussianSampler<Vector3d, Matrix3d>>();
    odomSampler_->setDistribution(sigmaU_);
    


    eventQueue_ = OrderedEventQueue();
}

IncrementalSimulator::~IncrementalSimulator() = default;


SE2 IncrementalSimulator::xTrue() const{
  return x_;
}

std::vector<Eigen::Vector2d> IncrementalSimulator::landmarkPosesTrue() const{
  std::vector<Eigen::Vector2d> poses;
  for (const auto* lm : landmarks_) {
      poses.push_back(lm->truePose);
  }
  return poses;
}

std::vector<Eigen::Vector2d> IncrementalSimulator::waypointsTrue() const{
  return platformController_->getWaypoints();
}

void IncrementalSimulator::history(std::vector<double> &timeHistory, std::vector<SE2> & xTrueHistory) const{
  timeHistory = timeStore_;
  xTrueHistory = xTrueStore_;
}

void IncrementalSimulator::start(){
  //start@ebe.core.EventBasedSimulator(obj);
  forceLinkTypesTutorialSlam2d();

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

  if(verbose_){std::cout << " - Current Pose: " << x_.toVector() << ", Current Vel: " << u_.toVector() << std::endl;}

  if(verbose_){std::cout << " - Predicting forward ..."<< std::endl;}
  handlePredictForwards(dT_);

  if(stepNumber_ % odomPeriod_ == 0){
    if(verbose_){std::cout << " - Updating Odom ..."<< std::endl;}
    updateOdometry();
  }
  if(stepNumber_ % slamObsPeriod_ == 0){
    if(verbose_){std::cout << " - Predicting SLAM Obs ..."<< std::endl;}
    predictSLAMObservations();
  }
  if(stepNumber_ % rangBearingObsPeriod_ == 0){
    if(verbose_){std::cout << " - Predicting range bearing observation ..."<< std::endl;}
    predictRangeBearingObservations();
  }
  if(stepNumber_ % gpsObsPeriod_ == 0){
    if(verbose_){std::cout << " - Predicting range bearing observation ..."<< std::endl;}
    predictGPSObservation();
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
  InitializationEvent initEvent = InitializationEvent(currentTime_, x0, SE2(0,0,0), P0, sigmaU_);
  if(verbose_){std::cout << "   - Push event to queue ... " << std::endl;}
  eventQueue_.push(std::make_shared<InitializationEvent>(initEvent));
  if(verbose_){std::cout << "   - initialize() complete " << std::endl;}
}
// We consult the system model and we give an event.




void IncrementalSimulator::updateOdometry(){
  // Initialize the ground truth state
  // Here we manually set x0 and P0 to 0s

  // Implement Controller

  if (platformController_->off()){
      carryOnRunning_ = false;
      std::cout<<"Platform Controller reached the end, Simulator ending as well."<<std::endl;
      return;
  }

  u_ = platformController_->computeControlInputs(x_);

  std::cout << "u_:" << u_.toVector() << std::endl;
  Vector3d noise = Vector3d(Sampler::gaussRand(0.0, sigmaUSqrtm_(0,0)), Sampler::gaussRand(0.0, sigmaUSqrtm_(1,1)), Sampler::gaussRand(0.0, sigmaUSqrtm_(2,2)));
  //Vector3d noise = Vector3d(0.0,0.0,0.0);
  SE2 u;
  u.fromVector(u_.toVector() + noiseScale_ * noise); //(noiseScale_ * (odomSampler_->generateSample()));      
  std::cout << "noise:" << noise << std::endl;
  std::cout << "u_:" << u.toVector() << std::endl;

  OdometryEvent odomEvent = OdometryEvent(currentTime_, u, sigmaU_);
  eventQueue_.push(std::make_shared<OdometryEvent>(odomEvent));
}




//void predictGPSObservation();
//void predictCompassObservation();
//void predictBearingObservations();
void IncrementalSimulator::predictSLAMObservations() {

  // Create and push event
  if(verbose_){std::cout << "   - predictSLAMObservations() start ..."<< std::endl;}
  LandmarkObservationVector lmObsVec = systemModel_->predictSLAMObservations(x_, landmarks_);
  auto lmObsEvent = std::make_shared<LandmarkObservationsEvent>(currentTime_, lmObsVec);
  eventQueue_.push(lmObsEvent);
  if(verbose_){std::cout << "   - predictSLAMObservations() complete"<< std::endl;}
}


void IncrementalSimulator::predictRangeBearingObservations() {

  // Create and push event
  if(verbose_){std::cout << "   - predictRangeBearingObservations() start ..."<< std::endl;}
  LMRangeBearingObservationVector lmObsVec = systemModel_->predictRangeBearingObservations(x_, landmarks_);
  auto lmObsEvent = std::make_shared<LMRangeBearingObservationsEvent>(currentTime_, lmObsVec);
  eventQueue_.push(lmObsEvent);
  if(verbose_){std::cout << "   - predictRangeBearingObservations() complete"<< std::endl;}
}


void IncrementalSimulator::predictGPSObservation(){
  if(verbose_){std::cout << "   - predictGPSObservations() start ..."<< std::endl;}
  Eigen::Vector2d value;
  Eigen::Matrix2d R;
  systemModel_->predictGPSObservation(x_, value, R);
  eventQueue_.push(std::make_shared<GPSObservationEvent>(currentTime_, value, R));
  if(verbose_){std::cout << "   - predictGPSObservations() complete"<< std::endl;}
}


//void generateHeartbeat();
void IncrementalSimulator::storeStepResults(){
  timeStore_.emplace_back(currentTime_);
  xTrueStore_.emplace_back(x_);
}

void IncrementalSimulator::saveGroundTruth(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return;
    }

    // Optional: write an offset param line like in tutorial_before.g2o
    out << "TUTORIAL_PARAMS_SE2_OFFSET 0 0 0 0\n";

    for (size_t i = 0; i < xTrueStore_.size(); ++i) {
        const auto& pose = xTrueStore_[i];
        out << std::fixed << std::setprecision(6);
        out << "TUTORIAL_VERTEX_SE2 " << i << " "
            << pose[0] << " "
            << pose[1] << " "
            << pose[2] << "\n";
    }

    // Optional: fix the first pose if needed
    if (!xTrueStore_.empty()) {
        out << "FIX 0\n";
    }

    out.close();
    std::cout << "Ground truth written to " << filename << std::endl;
}


}  // namespace tutorial
}  // namespace g2o
