#include "cauchy_edge_se2_pointxy.h"


namespace g2o {

namespace tutorial {


EdgeCauchySE2PointXY::EdgeCauchySE2PointXY()
    : EdgeNoneGaussianBinary<2, VertexSE2, VertexPointXY>() {
        _lnc = 1e-16;
    }

bool EdgeCauchySE2PointXY::read(std::istream& is) {
    return EdgeNoneGaussianBinary<2, VertexSE2, VertexPointXY>::read(is);
}

bool EdgeCauchySE2PointXY::write(std::ostream& os) const {
    return EdgeNoneGaussianBinary<2, VertexSE2, VertexPointXY>::write(os);
}

void EdgeCauchySE2PointXY::computeError() {
  const VertexPointXY* l2 = static_cast<const VertexPointXY*>(_vertices[1]);
  Eigen::Vector2d error = (_sensorCache->w2n() * l2->estimate()) - _measurement;

  // L_{MSE}(\mu) = \frac{d+1}{2} * \log( 1 + (x_i - \mu)^T \Sigma^{-1} (x_i - \mu))
  // d = 2
  _error[0] = sqrt(_lnc + (3/2) * log(1 + (error.transpose() * _realInformation * error)));
}

void EdgeCauchySE2PointXY::linearizeOplusy() {
  const VertexPointXY* l2 = static_cast<const VertexPointXY*>(_vertices[1]);
  Eigen::Vector2d error = (_sensorCache->w2n() * l2->estimate()) - _measurement;

  // Jacobian w.r.t. robot pose (x, y, theta)
  // f is the equivalent distribution of all functions that actually matters
  double g = (error.transpose() * _realInformation * error);
  double f = pow(1 + g, -3/2);
  double lnf = -(3/2) * log(1 + g);
  double r = sqrt(_lnc - lnf);
  double drdf = -1/(2 * f * sqrt(_lnc - lnf));
  double dfdg = -3/2 * pow(1 + g, -5/2);
  double dgdx = 2*(_realInformation(0,0) * error[0] + _realInformation(0,1) * error[1]);
  double dgdy = 2*(_realInformation(0,1) * error[0] + _realInformation(1,1) * error[1]);

  // Jacobian w.r.t. robot pose (x, y, theta)
  _jacobianOplusXi.setZero();
  _jacobianOplusXi(0,0) = -drdf * dfdg * dgdx;
  _jacobianOplusXi(0,1) = -drdf * dfdg * dgdy;

  // Jacobian w.r.t. landmark
  _jacobianOplusXj.setZero();
  _jacobianOplusXj(0,0) = drdf * dfdg * dgdx;
  _jacobianOplusXj(0,1) = drdf * dfdg * dgdy;
}

}}