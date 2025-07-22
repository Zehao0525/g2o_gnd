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
                 RGPSSqrt_(GPSsigmaRSqrt){
        
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

    SystemModel::SystemModel(nlohmann::json j){
        verbose_ = j.value("verbose", false);
        if(verbose_){std::cout<<" - Setting System model"<<std::endl;}
        RSLAMSqrt_ = Eigen::Matrix2d::Zero();
        std::vector<double> rslamsqrt = j["sensors"]["landmark_relative_location"].value("sigma", std::vector<double>(1,1));
        RSLAMSqrt_(0,0) = rslamsqrt[0];
        RSLAMSqrt_(1,1) = rslamsqrt[1];
        RSLAM_ = RSLAMSqrt_ * RSLAMSqrt_.transpose();
        SlamRange_ = j["sensors"]["landmark_relative_location"].value("range",25);

        RRBSqrt_ = Eigen::Matrix2d::Zero();
        std::vector<double> rrbsqrt = j["sensors"]["landmark_range_bearing"].value("sigma", std::vector<double>(1,0.1));
        RRBSqrt_(0,0) = rrbsqrt[0];
        RRBSqrt_(1,1) = rrbsqrt[1];
        RRB_ = RRBSqrt_ * RRBSqrt_.transpose();
        RBRange_ = j["sensors"]["landmark_range_bearing"].value("range",25);

        RGPSSqrt_ = Eigen::Matrix2d::Zero();
        std::vector<double> rgpssqrt = j["sensors"]["gps"].value("sigma", std::vector<double>(1,1));
        RGPSSqrt_(0,0) = rgpssqrt[0];
        RGPSSqrt_(1,1) = rgpssqrt[1];
        RGPS_ = RGPSSqrt_ * RGPSSqrt_.transpose();

        perturbWithNoise_ = j.value("perturb_with_noise", false);
        if(perturbWithNoise_){noiseScale_ = 1;}else{noiseScale_ = 0;}
    }

    // Given x and 
    SE2 SystemModel::predictState(const SE2& x, const SE2& u, double dT) const{
        return x * (u * dT);
    }

    // For effefciency: This avoids unessesary inverses.
    void SystemModel::setPlatformPose(const SE2& xTrue){
        trueInv_ = xTrue.inverse();
        truePose_ = xTrue;
    }

    void SystemModel::predictGPSObservation(const SE2& xTrue, Eigen::Vector2d& value, Eigen::Matrix2d& R) const {
        Eigen::Vector2d noise = Eigen::Vector2d(Sampler::gaussRand(0.0, RGPSSqrt_(0,0)), Sampler::gaussRand(0.0, RGPSSqrt_(1,1)));
        value = xTrue.translation() + noise;
        R = RGPS_;
        if(verbose_){std::cout<<" - GPS value" << value <<std::endl;}
        if(verbose_){std::cout<<" - GPS R" << R <<std::endl;}
    }

    LandmarkObservationVector SystemModel::predictSLAMObservations(const SE2& xTrue, const LandmarkPtrVector& ls) const{
        if(verbose_){std::cout<<" - SystemModel::predictSLAMObservations() start ..."<<std::endl;}
        SE2 trueInv = xTrue.inverse();

        LandmarkObservationVector lmObsVec;
        if(verbose_){std::cout<<" - Iteratively creating observations ..."<<std::endl;}
        for (const auto& l : ls) {
            Eigen::Vector2d delta = trueInv * l->truePose;  // from robot to landmark
            double range = delta.norm();

            if(range > SlamRange_){continue;}

            // Overrides range bering observation and does a linear GT observation
            //lmObsVec.emplace_back(l->id, observation, RSLAM_);
            Eigen::Vector2d noise = Eigen::Vector2d(Sampler::gaussRand(0.0, RSLAMSqrt_(0,0)), Sampler::gaussRand(0.0, RSLAMSqrt_(1,1)));
            delta += noiseScale_ * noise; // slamSampler_->generateSample();
            lmObsVec.emplace_back(l->id, delta, RSLAM_);
        }
        return lmObsVec;
    }



    LMRangeBearingObservationVector SystemModel::predictRangeBearingObservations(const SE2& xTrue, const LandmarkPtrVector& ls) const{
        if(verbose_){std::cout<<" - SystemModel::predictSLAMObservations() start ..."<<std::endl;}
        SE2 trueInv = xTrue.inverse();

        LMRangeBearingObservationVector lmObsVec;
        if(verbose_){std::cout<<" - Iteratively creating observations ..."<<std::endl;}
        for (const auto& l : ls) {
            Eigen::Vector2d delta = trueInv * l->truePose;  // from robot to landmark
            if(verbose_){std::cout<<" - delta: " << delta <<std::endl;}
            double range = delta.norm();

            if(range > RBRange_){continue;}

            double bearing = normalize_theta(std::atan2(delta[1], delta[0]));

            if(verbose_){std::cout<<" - range: " << range << ", bearing: " << bearing <<std::endl;}
            if(verbose_){std::cout<<" - range noise: " << RRBSqrt_(0,0) << ", bearing noise: " << RRBSqrt_(1,1) <<std::endl;}
            Eigen::Vector2d observation(range, bearing);

            Eigen::Vector2d noise = Eigen::Vector2d(Sampler::gaussRand(0.0, RRBSqrt_(0,0)), Sampler::gaussRand(0.0, RRBSqrt_(1,1)));
            observation = observation + noiseScale_ * noise;

            lmObsVec.emplace_back(l->id, observation, RRB_);
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
