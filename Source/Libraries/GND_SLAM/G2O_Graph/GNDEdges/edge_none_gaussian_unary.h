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

#ifndef G2O_TUTORIAL_EDGE_NONE_GAUSSIAN_UNARY_H
#define G2O_TUTORIAL_EDGE_NONE_GAUSSIAN_UNARY_H

#include "g2o/core/base_unary_edge.h"
#include "g2o_tutorial_slam2d_api.h"
#include "parameter_se2_offset.h"
#include "vertex_se2.h"

namespace g2o {

namespace tutorial {

class ParameterSE2Offset;
class CacheSE2Offset;

template <int Dim, typename VertexType>
class EdgeNoneGaussianUnary : public g2o::BaseUnaryEdge<1, Eigen::Matrix<double, Dim, 1>, VertexType> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    using InformationType = Eigen::Matrix<double, Dim, Dim>;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeNoneGaussianUnary()
      : g2o::BaseUnaryEdge<1, Eigen::Matrix<double, Dim, 1>, VertexType>(),
        _sensorOffset(0),
        _sensorCache(0),
        _power(2),
        _lnc(1e-3) {
    this->resizeParameters(1);
    this->installParameter(_sensorOffset, 0);
  }

  bool read(std::istream& is) {
    for (int i = 0; i < Dim; ++i)
      for (int j = i; j < Dim; ++j) {
        is >> this->information()(i, j);
        if (i != j) this->information()(j, i) = this->information()(i, j);
      }
    return true;
  }

  bool write(std::ostream& os) const {
    for (int i = 0; i < Dim; ++i) {
      os << this->_measurement[i] << " ";
    }
    for (int i = 0; i < Dim; ++i)
      for (int j = i; j < Dim; ++j) os << " " << this->information()(i, j);
    return os.good();
  }

  void gndSetInformation(const InformationType& information) {
    _realInformation = information;
    this->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
  }

  void gndSetInformation(const InformationType& information, double power) {
    _power = power;
    _realInformation = information / 4;
    this->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
  }

  void gndSetInformation(const InformationType& information, double power, double lnc) {
    _lnc = lnc;
    gndSetInformation(information, power);
  }

protected:
  bool resolveCaches() {
    ParameterVector pv(1);
    pv[0] = _sensorOffset;
    BaseUnaryEdge<1, Eigen::Matrix<double, Dim, 1>, VertexType>::resolveCache(_sensorCache,
                static_cast<OptimizableGraph::Vertex*>(BaseUnaryEdge<1, Eigen::Matrix<double, Dim, 1>, VertexType>::_vertices[0]),
                "TUTORIAL_CACHE_SE2_OFFSET", pv);
    return _sensorCache != 0;
  }

 protected:
  ParameterSE2Offset* _sensorOffset;
  CacheSE2Offset* _sensorCache;

  InformationType _realInformation;  // Original info matrix before powering
  double _power;
  double _lnc; // log normalization constant
};

}  // namespace tutorial
}  // namespace g2o

#endif

