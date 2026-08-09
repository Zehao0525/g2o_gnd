#pragma once

#include "ggd_kernel.h"

namespace g2o {

class G2O_CORE_API GGDCauchyKernel : public GGDKernel {
 public:
  GGDCauchyKernel();
  GGDCauchyKernel(double bound, double power);
  GGDCauchyKernel(double bound, double power, double lnc);
  GGDCauchyKernel(double bound, double power, double lnc, double cauchy_delta);

  void robustify(double e2, Vector3& rho) const override;

  void setParameters(double bound, double power, double lnc, double cauchy_delta = 0.0);

  double cauchyDelta() const { return cauchy_delta_; }

 protected:
  void updateCauchyShift();

  double cauchy_delta_ = 0.0;
  double rho_inner_at_e0_ = 0.0;
  double cauchy_at_e0_ = 0.0;
};

class G2O_CORE_API ToggelableGGDCauchyKernel : public GGDCauchyKernel {
 public:
  ToggelableGGDCauchyKernel();
  ToggelableGGDCauchyKernel(double bound, double power, const bool* ggdActive);
  ToggelableGGDCauchyKernel(double bound, double power, double lnc, const bool* ggdActive);
  ToggelableGGDCauchyKernel(double bound, double power, double lnc, double cauchy_delta,
                            const bool* ggdActive);

  void robustify(double e2, Vector3& rho) const override;
  void setBoolPointer(const bool* ggdActive);

 protected:
  const bool* ggdActive_;
};

}  // namespace g2o
