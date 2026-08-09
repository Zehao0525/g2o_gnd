#pragma once
#include "view.h"
#include "file_simulator.h"

#include <fstream> 
#include <Eigen/Core>
#include <nlohmann/json.hpp>


namespace g2o {
namespace tutorial {
namespace viz {

class FileSimulatorView : public View {
public:
    FileSimulatorView(FileSimulator* simulator, const Eigen::Vector3f& color, const Eigen::Vector3f& lmColor, const Eigen::Vector3f& wpColor);
    FileSimulatorView(FileSimulator* simulator, const std::string& filename);

    void processEvents(EventPtrVector& events);

    void pause() override;

    void update() override;

    void setView(const std::string& filename) override;

    void renderScene() const override;

private:
    FileSimulator* simulator_;
    Eigen::Vector3f lmColor_;
    double lmSize_;
    Eigen::Vector3f wpColor_;
    double wpSize_;
    std::vector<Eigen::Vector2d> landmarkPoses_;
    std::vector<Eigen::Vector2d> wayPoints_;

    Eigen::Vector3f obsMeasColor_;
};

}}}
