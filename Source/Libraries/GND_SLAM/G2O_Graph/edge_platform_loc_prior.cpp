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

#include "edge_platform_loc_prior.h"

using namespace Eigen;

namespace g2o {
namespace tutorial {

EdgePlatformLocPrior::EdgePlatformLocPrior()
    : BaseUnaryEdge<2, Vector2d, VertexSE2>(),
      _sensorOffset(0),
      _sensorCache(0) {
  resizeParameters(1);
  installParameter(_sensorOffset, 0);
}

bool EdgePlatformLocPrior::read(std::istream& is) {
  int paramId;
  is >> paramId;
  if (!setParameterId(0, paramId)) return false;
  is >> _measurement[0] >> _measurement[1];
  is >> information()(0, 0) >> information()(0, 1) >> information()(1, 1);
  information()(1, 0) = information()(0, 1);
  return true;
}

bool EdgePlatformLocPrior::write(std::ostream& os) const {
  os << _sensorOffset->id() << " ";
  os << measurement()[0] << " " << measurement()[1] << " ";
  os << information()(0, 0) << " " << information()(0, 1) << " "
     << information()(1, 1);
  return os.good();
}

void EdgePlatformLocPrior::computeError() {
  Eigen::Vector3d pose = (_sensorCache->n2w()).toVector();


  _error[0] = pose[0] - _measurement[0];
  _error[1] = pose[1] - _measurement[1];  // Normalize angle to [-pi, pi]
}

bool EdgePlatformLocPrior::resolveCaches() {
  ParameterVector pv(1);
  pv[0] = _sensorOffset;
  resolveCache(_sensorCache,
               static_cast<OptimizableGraph::Vertex*>(_vertices[0]),
               "TUTORIAL_CACHE_SE2_OFFSET", pv);
  return _sensorCache != 0;
}

}  // namespace tutorial
}  // namespace g2o
