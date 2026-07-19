#pragma once

#include "g2o/core/base_binary_edge.h"
#include "g2o_tutorial_slam2d_api.h"
#include "parameter_se2_offset.h"
#include "vertex_se2.h"
#include "vertex_point_xy.h"
#include "edge_none_gaussian_binary.h"

namespace g2o {

namespace tutorial {

class ParameterSE2Offset;
class CacheSE2Offset;

class G2O_TUTORIAL_SLAM2D_API EdgeCauchySE2PointXY
    : public EdgeNoneGaussianBinary<2, VertexSE2, VertexPointXY>{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EdgeCauchySE2PointXY();
  ~EdgeCauchySE2PointXY() = default;

  bool read(std::istream& is) override;
  bool write(std::ostream& os) const override;
  void computeError();
  void linearizeOplusy();
};

}  // namespace tutorial
}  // namespace g2o
