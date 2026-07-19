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

#include "edge_platform_loc_prior_gnd.h"

using namespace Eigen;

namespace g2o {
namespace tutorial {

EdgePlatformLocPriorGND::EdgePlatformLocPriorGND()
    : EdgeNoneGaussianUnary<3, VertexSE2>() {}


bool EdgePlatformLocPriorGND::read(std::istream& is) {
    return EdgeNoneGaussianUnary<3, VertexSE2>::read(is);
}

bool EdgePlatformLocPriorGND::write(std::ostream& os) const {
    return EdgeNoneGaussianUnary<3, VertexSE2>::write(os);
}

void EdgePlatformLocPriorGND::computeError() {
  Eigen::Vector3d pose = (_sensorCache->n2w()).toVector();


  Eigen::Vector3d error;
  error[0] = pose[0] - _measurement[0];
  error[1] = pose[1] - _measurement[1];
  error[2] = pose[2] - _measurement[2];   // Normalize angle to [-pi, pi]

  _error[0] = sqrt(_lnc + pow((error.transpose() * _realInformation * error),  _power));

}

}  // namespace tutorial
}  // namespace g2o
