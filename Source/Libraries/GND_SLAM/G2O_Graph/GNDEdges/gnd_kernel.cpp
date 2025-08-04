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

#include "gnd_kernel.h"

#include <cmath>

#include "robust_kernel_factory.h"

namespace g2o {

// For an unscaled information matrix, the flat reigion will roughly end at 1 std. 
// The errScale marks roughly how many stds (covariance sqrts) you want the bound to be at.
// _delta set to 100 by default: you should never be 10 stds off. But if you are, well, enjoy the gaussianess I guess.
GNDKernel::GNDKernel():GNDKernel(2,8){}
GNDKernel::GNDKernel(double bound, double power):GNDKernel(bound, power, 1e-3){}
GNDKernel::GNDKernel(double bound, double power, double lnc):GNDKernel(bound, power, lnc, 9){}
GNDKernel::GNDKernel(double bound, double power, double lnc, double tailPenaltyStd):RobustKernel(tailPenaltyStd * tailPenaltyStd), bound_(bound), power_(power), lnc_(lnc){
  if(power == 2){
    // This will effectivly invalidate the kernel
    _delta = -1;
  }
  else{
    assert(power/2 > 2.0 && "GND kernel requires beta > 4 to ensure convexity and differentiability");
  }
}

void GNDKernel::setParameters(double bound, double power, double lnc, double tailPenaltyStd){
  if(power == 2){
    // This will effectivly invalidate the kernel
    _delta = -1;
  }
  else{
    assert(power/2 > 2.0 && "GND kernel requires beta > 4 to ensure convexity and differentiability");
    bound_ = bound;
    power_ = power;
    lnc_ =lnc;
    _delta = tailPenaltyStd * tailPenaltyStd;
  }
}

void GNDKernel::robustify(double e2, Vector3& rho) const {
  // double lnf = pow(e2, power_/2);
  // double r = lnc_ + lnf;
  double scaledE2 = e2/(bound_*bound_);
  double scaledDelta = _delta/(bound_*bound_);
  double dr0 = (power_ / 2) * std::pow(scaledDelta, power_ / 2 - 1);
  rho[0] = lnc_ + pow(scaledE2, power_/2);
  rho[1] = power_/2 * pow(scaledE2, power_/2 - 1) * (1 / (bound_ * bound_));
  rho[2] = (power_/2) * (power_/2 - 1) * pow(scaledE2, power_/2 - 2) *  (1 / (bound_ * bound_)) *  (1 / (bound_ * bound_));

}



ToggelableGNDKernel::ToggelableGNDKernel():ToggelableGNDKernel(2,8,nullptr){}

ToggelableGNDKernel::ToggelableGNDKernel(double bound, double power, const bool* gndActive)
  : GNDKernel(bound, power), gndActive_(gndActive) {}

ToggelableGNDKernel::ToggelableGNDKernel(double bound, double power, double lnc, const bool* gndActive)
  : GNDKernel(bound, power, lnc), gndActive_(gndActive) {}

ToggelableGNDKernel::ToggelableGNDKernel(double bound, double power, double lnc, double tailPenaltyStd, const bool* gndActive)
  : GNDKernel(bound, power, lnc, tailPenaltyStd), gndActive_(gndActive) {}

void ToggelableGNDKernel::robustify(double e2, Vector3& rho) const {
  if (gndActive_ && *gndActive_) {
    GNDKernel::robustify(e2, rho);
  } else {
    rho[0] = e2;
    rho[1] = 1.0;
    rho[2] = 0.0;
  }
}

void ToggelableGNDKernel::setBoolPointer(const bool* gndActive){
  gndActive_ = gndActive;
}

G2O_REGISTER_ROBUST_KERNEL(GND, GNDKernel)
G2O_REGISTER_ROBUST_KERNEL(ToggelableGND, ToggelableGNDKernel)
}  // end namespace g2o
