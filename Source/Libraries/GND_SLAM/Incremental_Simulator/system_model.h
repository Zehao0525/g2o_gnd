#ifndef G2O_INCSIM2D_SYSTEM_MODEL_H
#define G2O_INCSIM2D_SYSTEM_MODEL_H

#include <map>
#include <vector>

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

    SystemModel(bool perturbWithNoise = false,
                 SE2& SLAM_sigmaR = std::nullopt,
                 SE2& GPS_sigmaR = std::nullopt);

                 
    void setPlatformPose(const SE2& x);

    // Given x and 
    SE2 predictState(const SE2& x,
                    const SE2& u,
                    double dT) const;

    LandmarkObservation predictSLAMObservation(const Landmark& l);

    //Eigen::VectorXd predictGPSObservation();

protected:
    void setupModels();

    Config config_;
    bool perturbWithNoise_;
    double noiseScale_;

    // What should This be?
    Matrix2d RGPS_;
    Matrix2d RGPSSqrtm_;

    Matrix2d RSLAM_;
    g2o::GaussianSampler<Eigen::Vector2d, Eigen::Matrix2d> SLAMSamper_;
    Matrix2d RSLAMSqrt_;
    double SlamRange_;

    SE2 trueInv_;
    SE2 truePose_;

    std::default_random_engine rng_;
};
}
}
