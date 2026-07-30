#pragma once
#include "view.h"
#include <Eigen/Core>
#include <nlohmann/json.hpp>

// Forward declarations
namespace g2o {
namespace tutorial {
namespace multibotsim {
class MultiDroneSLAMSystem;
class MultiDroneSLAMSystemNew;
}}}

namespace g2o {
namespace tutorial {
namespace viz {

class MultiDroneSLAMSystemView : public View {
public:
    MultiDroneSLAMSystemView(multibotsim::MultiDroneSLAMSystem* system, const Eigen::Vector3f& color, bool visualise_path = true);
    MultiDroneSLAMSystemView(multibotsim::MultiDroneSLAMSystem* system, const std::string& filename, bool visualise_path = true);

    MultiDroneSLAMSystemView(multibotsim::MultiDroneSLAMSystemNew* system, const Eigen::Vector3f& color, bool visualise_path = true);
    MultiDroneSLAMSystemView(multibotsim::MultiDroneSLAMSystemNew* system, const std::string& filename, bool visualise_path = true);

    void setView(const std::string& filename) override;
    void update() override;
    void pause() override;
    void renderScene() const override;

private:
    multibotsim::MultiDroneSLAMSystem* slamSystem_ = nullptr;
    multibotsim::MultiDroneSLAMSystemNew* slamSystemNew_ = nullptr;
    Eigen::Isometry3d currentPose3d_;
    std::vector<Eigen::Isometry3d> path3d_;
    bool visualise_path_;
};

}}}
