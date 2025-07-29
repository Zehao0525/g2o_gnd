// SLAMSystemView.cpp
#include "slam_system_view.h"
#include <GL/gl.h>

namespace g2o {
namespace tutorial {
namespace viz {


SLAMSystemView::SLAMSystemView(SlamSystem* system, const Eigen::Vector3f& color)
    : View(color), slamSystem_(system), marginalizeCounter_(0)  {}


SLAMSystemView::SLAMSystemView(SlamSystem* system, const std::string& filename)
    : View(Eigen::Vector3f(0.0,0.0,0.0)), slamSystem_(system), marginalizeCounter_(0) {
        setView(filename);
    }


void SLAMSystemView::pause(){
    View::pause();
    computeMarginals();
}


void SLAMSystemView::setView(const std::string& filename){
    std::ifstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open Simulator config file: " + filename);
    }
    nlohmann::json j;
    f >> j;

    auto color = j["slam_system"]["platform"].value("color", std::vector<float>{0.0f,1.0f,0.0f});
    color_ = Eigen::Vector3f(color[0], color[1], color[2]);
    platformSize_ = j["slam_system"]["platform"].value("size", 1.5);

    auto lmColor = j["slam_system"]["landmark"].value("color", std::vector<float>{0.0f,1.0f,0.5f});
    lmColor_ = Eigen::Vector3f(lmColor[0], lmColor[1], lmColor[2]);
    lmSize_ = j["slam_system"]["landmark"].value("size", 0.5);

    auto measColor = j["slam_system"]["measurment"].value("color", std::vector<float>{0.0f,1.0f,0.5f});
    measColor_ = Eigen::Vector3f(measColor[0], measColor[1], measColor[2]);

    marginalizePeriod_ = j["slam_system"].value("marginal_update_period", 25);
}

void SLAMSystemView::update() {
    //std::lock_guard<std::mutex> lock(slamSystem_->mutex());
    std::cout << " SLAMSystemView update " << std::endl;
    marginalizeCounter_++;
    Eigen::Vector3d x;
    slamSystem_->platformEstimate(x);
    updateRobotPose(x);
    if(marginalizeCounter_ > marginalizePeriod_){
        computeMarginals();
        marginalizeCounter_ = 0;
    }
}

void SLAMSystemView::computeMarginals(){
    Eigen::Vector3d x;
    Eigen::Matrix2d P;
    std::vector<Eigen::Vector2d> m;
    std::vector<Eigen::Matrix2d> Pmm;
    std::vector<int> lmIDs;
    slamSystem_->getSceneEstimatesWithP(x,P,m,Pmm,lmIDs);
    Eigen::LLT<Eigen::Matrix2d> llt(P);
    Eigen::Matrix2d PSqrt_ = llt.matrixL();;

    std::cout << " PSqrt_ " << PSqrt_ << std::endl;
    std::cout << " PSqrt_ " << PSqrt_ << std::endl;
    std::cout << " PSqrt_ " << PSqrt_ << std::endl;
    std::cout << " PSqrt_ " << PSqrt_ << std::endl;
    std::shared_ptr<CovarianceViz> covViz = std::make_shared<CovarianceViz>(Eigen::Vector2d(0.0,0.0), PSqrt_, marginalizePeriod_, marginalizePeriod_/4,  MeasurmentViz::AttachmentType::Loc, color_);
    
    std::cout << " Adding Covariance " << std::endl;
    measVizVertex_.push_back(covViz);
    std::cout << " Complete Adding Covariance " << std::endl;
}

}}}
