#include "system_model.h"

#include <cmath>
#include <iostream>
#include <map>
#include <fstream>
#include <iomanip>

#include "g2o/stuff/sampler.h"


namespace g2o {
namespace tutorial {

    SystemModel::SystemModel(bool perturbWithNoise,
                 std::optional<Eigen::Matrix2d> SLAMsigmaRSqrt,
                 std::optional<double> SLAM_Range,
                 std::optional<Eigen::Matrix2d> GPSsigmaRSqrt){

        RSLAM_ = SLAMsigmaRSqrt * SLAMsigmaRSqrt.transpose();
        SLAMSamper_.setDistribution(RSLAM_);
        RSLAMSqrtm_ = SLAMsigmaRSqrt;
        SlamRange_ = SLAM_Range;
        RGPS_ = GPSsigmaRSqrt * GPSsigmaRSqrt.transpose();
        RGPSSqrtm_ = GPSsigmaRSqrt;
        perturbWithNoise_ = perturbWithNoise;
        if(perturbWithNoise_){noiseScale_ = 1;}else{noiseScale_ = 0;}

    }

    // Given x and 
    SE2 SystemModel::predictState(const SE2& x, const SE2& u, double dT){
        return x + u.scalerMul(dT);
    }

    // For effefciency: This avoids unessesary inverses.
    void SystemModel::setPlatformPose(const SE2& xTrue){
        trueInv_ = xTrue.inverse();
        truePose_ = xTrue;
    }

    LandmarkObservationVector SystemModel::predictSLAMObservations(const SE2& xTrue, const LandmarkPtrVector& ls) {
        SE2 trueInv = xTrue.inverse();

        LandmarkObservationVector lmObsVec;
        for (const auto& l : ls) {
            // Wrong. You
            Eigen::Vector2d delta = trueInv * l->truePose;  // from robot to landmark
            double range = delta.norm();

            if(range > SlamRange_){continue;}

            double bearing = normalize_theta(std::atan2(delta.y(), delta.x()) - xTrue.rotation().angle());

            Eigen::Vector2d observation(range, bearing);

            observation += noiseScale_ * SLAMSampler_.generateSample();

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
