#include "ggd_cauchy_kernel.h"

#include <cassert>
#include <cmath>

#include "robust_kernel_factory.h"

namespace g2o {

namespace {

void validateGgdPower(double power) {
  if (power == 2.0) {
    return;
  }
  assert(power / 2.0 > 2.0 &&
         "GGD Cauchy kernel requires beta > 4 to ensure convexity and differentiability");
}

}  // namespace

GGDCauchyKernel::GGDCauchyKernel() : GGDCauchyKernel(2, 8) {}

GGDCauchyKernel::GGDCauchyKernel(double bound, double power) : GGDCauchyKernel(bound, power, 1e-3) {}

GGDCauchyKernel::GGDCauchyKernel(double bound, double power, double lnc)
    : GGDCauchyKernel(bound, power, lnc, 0.0) {}

GGDCauchyKernel::GGDCauchyKernel(double bound, double power, double lnc, double cauchy_delta)
    : GGDKernel(bound, power, lnc) {
  setParameters(bound, power, lnc, cauchy_delta);
}

void GGDCauchyKernel::setParameters(double bound, double power, double lnc, double cauchy_delta) {
  if (power == 2.0) {
    _delta = -1;
  } else {
    validateGgdPower(power);
    bound_ = bound;
    power_ = power;
    lnc_ = lnc;
  }
  cauchy_delta_ = (cauchy_delta > 0.0) ? cauchy_delta : bound_;
  updateCauchyShift();
}

void GGDCauchyKernel::updateCauchyShift() {
  const double e0 = bound_ * bound_;
  rho_inner_at_e0_ = lnc_ + 1.0;

  const double dsqr = cauchy_delta_ * cauchy_delta_;
  cauchy_at_e0_ = dsqr * std::log(1.0 + e0 / dsqr);
}

void GGDCauchyKernel::robustify(double e2, Vector3& rho) const {
  const double e0 = bound_ * bound_;
  if (e2 <= e0) {
    GGDKernel::robustify(e2, rho);
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

ToggelableGGDCauchyKernel::ToggelableGGDCauchyKernel() : ToggelableGGDCauchyKernel(2, 8, nullptr) {}

ToggelableGGDCauchyKernel::ToggelableGGDCauchyKernel(double bound, double power,
                                                       const bool* ggdActive)
    : ToggelableGGDCauchyKernel(bound, power, 1e-3, ggdActive) {}

ToggelableGGDCauchyKernel::ToggelableGGDCauchyKernel(double bound, double power, double lnc,
                                                       const bool* ggdActive)
    : ToggelableGGDCauchyKernel(bound, power, lnc, 0.0, ggdActive) {}

ToggelableGGDCauchyKernel::ToggelableGGDCauchyKernel(double bound, double power, double lnc,
                                                       double cauchy_delta, const bool* ggdActive)
    : GGDCauchyKernel(bound, power, lnc, cauchy_delta), ggdActive_(ggdActive) {}

void ToggelableGGDCauchyKernel::robustify(double e2, Vector3& rho) const {
  if (ggdActive_ && *ggdActive_) {
    GGDCauchyKernel::robustify(e2, rho);
  } else {
    rho[0] = e2;
    rho[1] = 1.0;
    rho[2] = 0.0;
  }
}

void ToggelableGGDCauchyKernel::setBoolPointer(const bool* ggdActive) { ggdActive_ = ggdActive; }

G2O_REGISTER_ROBUST_KERNEL(GGDCauchy, GGDCauchyKernel)
G2O_REGISTER_ROBUST_KERNEL(ToggelableGGDCauchy, ToggelableGGDCauchyKernel)

}  // namespace g2o
