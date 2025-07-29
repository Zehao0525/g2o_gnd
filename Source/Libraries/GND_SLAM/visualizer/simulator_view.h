#pragma once
#include "view.h"
#include "incremental_simulator.h"

#include <fstream> 
#include <Eigen/Core>
#include <nlohmann/json.hpp>


namespace g2o {
namespace tutorial {
namespace viz {

class SimulatorView : public View {
public:
    SimulatorView(IncrementalSimulator* simulator, const Eigen::Vector3f& color, const Eigen::Vector3f& lmColor, const Eigen::Vector3f& wpColor);
    SimulatorView(IncrementalSimulator* simulator, const std::string& filename);

    void processEvents(EventPtrVector& events);

    void pause() override;

    void update() override;

    void setView(const std::string& filename) override;

    void renderScene() const override;

private:
    IncrementalSimulator* simulator_;
    Eigen::Vector3f lmColor_;
    double lmSize_;
    Eigen::Vector3f wpColor_;
    double wpSize_;
    std::vector<Eigen::Vector2d> landmarkPoses_;
    std::vector<Eigen::Vector2d> wayPoints_;

    Eigen::Vector3f gpsMeasColor_;
    Eigen::Vector3f lmrbMeasColor_;
    Eigen::Vector3f lmobMeasColor_;
};

}}}
