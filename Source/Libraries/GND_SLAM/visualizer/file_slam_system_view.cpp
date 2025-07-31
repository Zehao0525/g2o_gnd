#include "file_slam_system_view.h"
#include <GL/gl.h>


namespace g2o {
namespace tutorial {
namespace viz {


FileSlamSystemView::FileSlamSystemView(FileSlamSystem* sim, const Eigen::Vector3f& botColor, const Eigen::Vector3f& lmColor, const Eigen::Vector3f& wpColor)
    : View(botColor), slamSystem_(sim), lmColor_(lmColor), wpColor_(wpColor),
    lmSize_(0.5), wpSize_(0.5) {
    }

FileSlamSystemView::FileSlamSystemView(FileSlamSystem* sim, const std::string& filename):
    slamSystem_(sim), View(Eigen::Vector3f(0.0,1.0,0.0)), lmColor_(Eigen::Vector3f(0.0,1.0,0.5)), wpColor_(Eigen::Vector3f(1.0,5.0,0.0)),
    lmSize_(0.5), wpSize_(0.5) {
        setView(filename);
    }


void FileSlamSystemView::pause(){
    View::pause();
}

void FileSlamSystemView::update() {
    Eigen::Vector3d x;
    slamSystem_->platformEstimate2d(x);
    updateRobotPose(x);
    std::cout << "\n\n x:" << x <<std::endl;
}

void FileSlamSystemView::setView(const std::string& filename){
    std::ifstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open Simulator config file: " + filename);
    }
    nlohmann::json j;
    f >> j;

    std::cout << "View_Manager_setup" <<std::endl;

    auto color = j["slam_system"]["platform"].value("color", std::vector<float>{0.0f,0.0f,1.0f});
    color_ = Eigen::Vector3f(color[0], color[1], color[2]);

    auto lmobMeasColor = j["slam_system"]["measurment"].value("lm_observation_color", std::vector<float>{0.0f,1.0f,0.5f});
    obsMeasColor_ = Eigen::Vector3f(lmobMeasColor[0], lmobMeasColor[1], lmobMeasColor[2]);

    platformSize_ = j["slam_system"]["platform"].value("size", 0.2);
}


void FileSlamSystemView::processEvents(EventPtrVector& events){
    update();
}



void FileSlamSystemView::renderScene() const{

}

}}}