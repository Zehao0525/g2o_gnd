#pragma once
#include "view.h"
#include "slam_system.h"


namespace g2o {
namespace tutorial {
namespace viz {

class SLAMSystemView : public View {
public:
    SLAMSystemView(SlamSystem* system, const Eigen::Vector3f& color);

    void update() override;

private:
    SlamSystem* slamSystem_;
};

}}}