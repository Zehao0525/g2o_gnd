#ifndef G2O_INCSIM2D_SYSTEM_MODEL_H
#define G2O_INCSIM2D_SYSTEM_MODEL_H

#include <map>
#include <vector>
#include <optional>

#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include "g2o_tutorial_slam2d_api.h"
#include "se2.h"
#include "events.h"

#include "g2o/stuff/sampler.h"


namespace g2o {
namespace tutorial {


class G2O_TUTORIAL_SLAM2D_API SystemModel {
public:

    static constexpr int NP = 3; // Platform dimension
    static constexpr int NL = 2; // Landmark dimension

    SystemModel(bool perturbWithNoise,
                const Eigen::Matrix2d& SLAMsigmaRSqrt,
                double SLAM_Range,
                const Eigen::Matrix2d& GPSsigmaRSqrt);

    
    SystemModel(nlohmann::json j);

                 
    void setPlatformPose(const SE2& x);

    // Given x and 
    SE2 predictState(const SE2& x,
                    const SE2& u,
                    double dT) const;

    GPSObservationEvent SystemModel::predictGPSObservation(const SE2& xTrue) const;

    LandmarkObservationVector predictSLAMObservations(const SE2& xTrue, const LandmarkPtrVector& ls);

    LMRangeBearingObservationVector predictRangeBearingObservations(const SE2& xTrue, const LandmarkPtrVector& ls);

    //Eigen::VectorXd predictGPSObservation();

protected:
    void setupModels();

    bool perturbWithNoise_;
    double noiseScale_;

    // What should This be?
    Eigen::Matrix2d RGPS_;
    Eigen::Matrix2d RGPSSqrt_;

    Eigen::Matrix2d RSLAM_;
    static thread_local std::unique_ptr<g2o::GaussianSampler<Eigen::Vector2d, Eigen::Matrix2d>> slamSampler_;
    Eigen::Matrix2d RSLAMSqrt_;
    double SlamRange_;

    Eigen::Matrix2d RRB_;
    Eigen::Matrix2d RRBSqrt_;
    double RBRange_;

    SE2 trueInv_;
    SE2 truePose_;

    std::default_random_engine rng_;


    bool verbose_ = true;
};
}
}

#endif
