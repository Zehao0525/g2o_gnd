#pragma once
#include "view.h"
#include <Eigen/Core>
#include <nlohmann/json.hpp>

// Forward declaration
namespace g2o {
namespace tutorial {
namespace multibotsim {
class DataBasedSimulation;
}}}

namespace g2o {
namespace tutorial {
namespace viz {

/// View that displays ground-truth pose and path from a DataBasedSimulation.
class DataBasedSimulationView : public View {
public:
    DataBasedSimulationView(multibotsim::DataBasedSimulation* sim, const Eigen::Vector3f& color, bool visualise_path = true);
    DataBasedSimulationView(multibotsim::DataBasedSimulation* sim, const std::string& filename, bool visualise_path = true);

    void setView(const std::string& filename) override;
    void update() override;
    void pause() override;
    void renderScene() const override;

private:
    multibotsim::DataBasedSimulation* simulation_;
    Eigen::Isometry3d currentPose3d_;
    std::vector<Eigen::Isometry3d> path3d_;
    bool visualise_path_;
};

}}}
