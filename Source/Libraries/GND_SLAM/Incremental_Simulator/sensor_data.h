#ifndef G2O_INCSIN2D_SNESOR_DATA_H
#define G2O_INCSIN2D_SNESOR_DATA_H

#include "g2o_tutorial_slam2d_api.h"
#include "se2.h"

namespace g2o {
namespace tutorial {

  /**
   * \brief simulated landmark
   */
  struct G2O_TUTORIAL_SLAM2D_API Landmark {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int id;
    Eigen::Vector2d truePose;
    Eigen::Vector2d simulatedPose;
    std::vector<int> seenBy;
    Landmark() : id(-1) {}
  };
  using LandmarkVector = std::vector<Landmark>;
  using LandmarkPtrVector = std::vector<Landmark*>;

  /**
   * simulated pose of the robot
   */
  struct G2O_TUTORIAL_SLAM2D_API GridPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int id;
    SE2 truePose;
    SE2 simulatorPose;
    LandmarkPtrVector landmarks;  ///< the landmarks observed by this node
  };
  using PosesVector = std::vector<GridPose>;

  /**
   * \brief odometry constraint
   */
  struct G2O_TUTORIAL_SLAM2D_API GridEdge {
    int from;
    int to;
    SE2 trueTransf;
    SE2 simulatorTransf;
    Eigen::Matrix3d information;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };
  using GridEdgeVector = std::vector<GridEdge>;

  struct G2O_TUTORIAL_SLAM2D_API LandmarkEdge {
    int from;
    int to;
    Eigen::Vector2d trueMeas;
    Eigen::Vector2d simulatorMeas;
    Eigen::Matrix2d information;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };
  using LandmarkEdgeVector = std::vector<LandmarkEdge>;
    
    // value = Range,Bearing
    struct LandmarkObservation {
        int landmark_id;
        Eigen::Vector2d value;
        Eigen::Matrix2d covariance;

        LandmarkObservation(int id, const Eigen::Vector2d& val, const Eigen::Matrix2d& cov)
            : landmark_id(id), value(val), covariance(cov) {}
    };

    using LandmarkObservationVector = std::vector<LandmarkObservation>;

}
}

#endif