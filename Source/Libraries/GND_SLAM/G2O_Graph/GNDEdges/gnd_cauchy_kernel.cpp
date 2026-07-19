#include "gnd_cauchy_kernel.h"

#include <cassert>
#include <cmath>

#include "robust_kernel_factory.h"

namespace g2o {

namespace {

void validateGndPower(double power) {
  if (power == 2.0) {
    return;
  }
  assert(power / 2.0 > 2.0 &&
         "GND Cauchy kernel requires beta > 4 to ensure convexity and differentiability");
}

}  // namespace

GNDCauchyKernel::GNDCauchyKernel() : GNDCauchyKernel(2, 8) {}

GNDCauchyKernel::GNDCauchyKernel(double bound, double power) : GNDCauchyKernel(bound, power, 1e-3) {}

GNDCauchyKernel::GNDCauchyKernel(double bound, double power, double lnc)
    : GNDCauchyKernel(bound, power, lnc, 0.0) {}

GNDCauchyKernel::GNDCauchyKernel(double bound, double power, double lnc, double cauchy_delta)
    : GNDKernel(bound, power, lnc) {
  setParameters(bound, power, lnc, cauchy_delta);
}

void GNDCauchyKernel::setParameters(double bound, double power, double lnc, double cauchy_delta) {
  if (power == 2.0) {
    _delta = -1;
  } else {
    validateGndPower(power);
    bound_ = bound;
    power_ = power;
    lnc_ = lnc;
  }
  cauchy_delta_ = (cauchy_delta > 0.0) ? cauchy_delta : bound_;
  updateCauchyShift();
}

void GNDCauchyKernel::updateCauchyShift() {
  const double e0 = bound_ * bound_;
  rho_inner_at_e0_ = lnc_ + 1.0;

  const double dsqr = cauchy_delta_ * cauchy_delta_;
  cauchy_at_e0_ = dsqr * std::log(1.0 + e0 / dsqr);
}

void GNDCauchyKernel::robustify(double e2, Vector3& rho) const {
  const double e0 = bound_ * bound_;
  if (e2 <= e0) {
    GNDKernel::robustify(e2, rho);
    return;
  }

  const double dsqr = cauchy_delta_ * cauchy_delta_;
  const double dsqrReci = 1.0 / dsqr;
  const double aux = dsqrReci * e2 + 1.0;
  const double cauchy_rho = dsqr * std::log(aux);

  rho[0] = rho_inner_at_e0_ + (cauchy_rho - cauchy_at_e0_);
  rho[1] = 1.0 / aux;
  rho[2] = -dsqrReci * rho[1] * rho[1];
}

ToggelableGNDCauchyKernel::ToggelableGNDCauchyKernel() : ToggelableGNDCauchyKernel(2, 8, nullptr) {}

ToggelableGNDCauchyKernel::ToggelableGNDCauchyKernel(double bound, double power,
                                                       const bool* gndActive)
    : ToggelableGNDCauchyKernel(bound, power, 1e-3, gndActive) {}

ToggelableGNDCauchyKernel::ToggelableGNDCauchyKernel(double bound, double power, double lnc,
                                                       const bool* gndActive)
    : ToggelableGNDCauchyKernel(bound, power, lnc, 0.0, gndActive) {}

ToggelableGNDCauchyKernel::ToggelableGNDCauchyKernel(double bound, double power, double lnc,
                                                       double cauchy_delta, const bool* gndActive)
    : GNDCauchyKernel(bound, power, lnc, cauchy_delta), gndActive_(gndActive) {}

void ToggelableGNDCauchyKernel::robustify(double e2, Vector3& rho) const {
  if (gndActive_ && *gndActive_) {
    GNDCauchyKernel::robustify(e2, rho);
  } else {
    rho[0] = e2;
    rho[1] = 1.0;
    rho[2] = 0.0;
  }
}

void ToggelableGNDCauchyKernel::setBoolPointer(const bool* gndActive) { gndActive_ = gndActive; }

G2O_REGISTER_ROBUST_KERNEL(GNDCauchy, GNDCauchyKernel)
G2O_REGISTER_ROBUST_KERNEL(ToggelableGNDCauchy, ToggelableGNDCauchyKernel)

}  // namespace g2o
