#pragma once
#include "view.h"
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include "slam_system.h"


namespace g2o {
namespace tutorial {
namespace viz {

class SLAMSystemView : public View {
public:
    SLAMSystemView(SlamSystem* system, const Eigen::Vector3f& color);

    SLAMSystemView(SlamSystem* system, const std::string& filename);

    void setView(const std::string& filename);

    void update() override;

    void pause() override;

    void computeMarginals();

private:
    SlamSystem* slamSystem_;

    Eigen::Vector3f lmColor_;
    double lmSize_;
    int marginalizePeriod_;
    int marginalizeCounter_;

};

}}}