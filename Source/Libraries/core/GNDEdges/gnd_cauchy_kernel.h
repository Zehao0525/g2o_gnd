#pragma once

#include "gnd_kernel.h"

namespace g2o {

class G2O_CORE_API GNDCauchyKernel : public GNDKernel {
 public:
  GNDCauchyKernel();
  GNDCauchyKernel(double bound, double power);
  GNDCauchyKernel(double bound, double power, double lnc);
  GNDCauchyKernel(double bound, double power, double lnc, double cauchy_delta);

  void robustify(double e2, Vector3& rho) const override;

  void setParameters(double bound, double power, double lnc, double cauchy_delta = 0.0);

  double cauchyDelta() const { return cauchy_delta_; }

 protected:
  void updateCauchyShift();

  double cauchy_delta_ = 0.0;
  double rho_inner_at_e0_ = 0.0;
  double cauchy_at_e0_ = 0.0;
};

class G2O_CORE_API ToggelableGNDCauchyKernel : public GNDCauchyKernel {
 public:
  ToggelableGNDCauchyKernel();
  ToggelableGNDCauchyKernel(double bound, double power, const bool* gndActive);
  ToggelableGNDCauchyKernel(double bound, double power, double lnc, const bool* gndActive);
  ToggelableGNDCauchyKernel(double bound, double power, double lnc, double cauchy_delta,
                            const bool* gndActive);

  void robustify(double e2, Vector3& rho) const override;
  void setBoolPointer(const bool* gndActive);

 protected:
  const bool* gndActive_;
};

}  // namespace g2o
