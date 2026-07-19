#pragma once
#include "view.h"
#include <Eigen/Core>
#include <nlohmann/json.hpp>

// Forward declaration
namespace g2o {
namespace tutorial {
namespace multibotsim {
class MultiDroneSLAMSystem;
}}}

namespace g2o {
namespace tutorial {
namespace viz {

class MultiDroneSLAMSystemView : public View {
public:
    MultiDroneSLAMSystemView(multibotsim::MultiDroneSLAMSystem* system, const Eigen::Vector3f& color, bool visualise_path = true);
    MultiDroneSLAMSystemView(multibotsim::MultiDroneSLAMSystem* system, const std::string& filename, bool visualise_path = true);

    void setView(const std::string& filename) override;
    void update() override;
    void pause() override;
    void renderScene() const override;

private:
    multibotsim::MultiDroneSLAMSystem* slamSystem_;
    Eigen::Isometry3d currentPose3d_;
    std::vector<Eigen::Isometry3d> path3d_;
    bool visualise_path_;
};

}}}
