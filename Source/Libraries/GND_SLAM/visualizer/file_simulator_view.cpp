#include "file_simulator_view.h"
#include <GL/gl.h>


namespace g2o {
namespace tutorial {
namespace viz {


FileSimulatorView::FileSimulatorView(FileSimulator* sim, const Eigen::Vector3f& botColor, const Eigen::Vector3f& lmColor, const Eigen::Vector3f& wpColor)
    : View(botColor), simulator_(sim), lmColor_(lmColor), wpColor_(wpColor),
    lmSize_(0.5), wpSize_(0.5) {
    }

FileSimulatorView::FileSimulatorView(FileSimulator* sim, const std::string& filename):
    simulator_(sim), View(Eigen::Vector3f(0.0,1.0,0.0)), lmColor_(Eigen::Vector3f(0.0,1.0,0.5)), wpColor_(Eigen::Vector3f(1.0,5.0,0.0)),
    lmSize_(0.5), wpSize_(0.5) {
        setView(filename);
    }


void FileSimulatorView::pause(){
    View::pause();
}

void FileSimulatorView::update() {
    updateRobotPose(simulator_->xTrue2d().toVector());
}

void FileSimulatorView::setView(const std::string& filename){
    std::ifstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open Simulator config file: " + filename);
    }
    nlohmann::json j;
    f >> j;

    std::cout << "View_Manager_setup" <<std::endl;

    auto color = j["simulator"]["platform"].value("color", std::vector<float>{0.0f,1.0f,0.0f});
    color_ = Eigen::Vector3f(color[0], color[1], color[2]);

    auto lmobMeasColor = j["slam_system"]["measurment"].value("lm_observation_color", std::vector<float>{0.0f,1.0f,0.5f});
    obsMeasColor_ = Eigen::Vector3f(lmobMeasColor[0], lmobMeasColor[1], lmobMeasColor[2]);

    platformSize_ = j["simulator"]["platform"].value("size", 0.2);
}


void FileSimulatorView::processEvents(EventPtrVector& events){
    updateRobotPose(simulator_->xTrue2d().toVector());

}



void FileSimulatorView::renderScene() const{

}

}}}