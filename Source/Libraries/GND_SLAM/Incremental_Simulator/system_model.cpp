#include "system_model.h"

#include <cmath>
#include <iostream>
#include <map>
#include <fstream>
#include <iomanip>

#include "g2o/stuff/sampler.h"


namespace g2o {
namespace tutorial {

    thread_local  std::unique_ptr<GaussianSampler<Eigen::Vector2d, Eigen::Matrix2d>> SystemModel::slamSampler_;

    SystemModel::SystemModel(bool perturbWithNoise,
                 const Eigen::Matrix2d& SLAMsigmaRSqrt,
                 double SLAM_Range,
                 const Eigen::Matrix2d& GPSsigmaRSqrt):
                 RSLAMSqrt_(SLAMsigmaRSqrt),
                 SlamRange_(SLAM_Range),
                 RGPSSqrtm_(GPSsigmaRSqrt){
        
        if(verbose_){std::cout<<" - Initialising SystemModel"<<std::endl;}
        RSLAM_ = SLAMsigmaRSqrt * SLAMsigmaRSqrt.transpose();
        if(verbose_){std::cout<<" - Setting slamSampler_ distribution"<<std::endl;}
        if(!slamSampler_){
            if(verbose_){std::cout<<"   - No slamSampler_, creating slamSampler_"<<std::endl;}
            slamSampler_ = std::make_unique<GaussianSampler<Eigen::Vector2d, Eigen::Matrix2d>>();
            if(verbose_){std::cout<<"   - slamSampler_ created"<<std::endl;}
        }
        slamSampler_->setDistribution(RSLAM_);
        RGPS_ = GPSsigmaRSqrt * GPSsigmaRSqrt.transpose();
        perturbWithNoise_ = perturbWithNoise;
        if(perturbWithNoise_){noiseScale_ = 1;}else{noiseScale_ = 0;}
        if(verbose_){std::cout<<" - SystemModel Initialisation complete"<<std::endl;}

    }

    // Given x and 
    SE2 SystemModel::predictState(const SE2& x, const SE2& u, double dT) const{
        return x * (u*dT);
    }

    // For effefciency: This avoids unessesary inverses.
    void SystemModel::setPlatformPose(const SE2& xTrue){
        trueInv_ = xTrue.inverse();
        truePose_ = xTrue;
    }

    LandmarkObservationVector SystemModel::predictSLAMObservations(const SE2& xTrue, const LandmarkPtrVector& ls){
        if(verbose_){std::cout<<" - SystemModel::predictSLAMObservations() start ..."<<std::endl;}
        SE2 trueInv = xTrue.inverse();

        LandmarkObservationVector lmObsVec;
        if(verbose_){std::cout<<" - Iteratively creating observations ..."<<std::endl;}
        for (const auto& l : ls) {
            // Wrong. You
            Eigen::Vector2d delta = trueInv * l->truePose;  // from robot to landmark
            double range = delta.norm();

            if(range > SlamRange_){continue;}

            double bearing = normalize_theta(std::atan2(delta.y(), delta.x()) - xTrue.rotation().angle());

            Eigen::Vector2d observation(range, bearing);

            observation += noiseScale_ * slamSampler_->generateSample();

            lmObsVec.emplace_back(l->id, observation, RSLAM_);
        }
        return lmObsVec;
    }

    //TODO Deal with
    // GPSObservation SystemModel::predictGPSObservation(){
    //     GPSObservation result;
    //     result.trueObservation = truePose_;
    //     result.noisyObservation = result.trueObservation;

    //     if (perturbWithNoise_) {
    //         result.noisyObservation += sampleMultivariateNormal(RGPS_);
    //     }

    //     result.R = RGPS_;
    //     return result;

    // }
};
}
