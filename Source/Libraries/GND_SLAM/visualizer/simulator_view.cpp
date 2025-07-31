#include "simulator_view.h"
#include <GL/gl.h>


namespace g2o {
namespace tutorial {
namespace viz {


SimulatorView::SimulatorView(IncrementalSimulator* sim, const Eigen::Vector3f& botColor, const Eigen::Vector3f& lmColor, const Eigen::Vector3f& wpColor)
    : View(botColor), simulator_(sim), lmColor_(lmColor), wpColor_(wpColor),
    lmSize_(0.5), wpSize_(0.5) {
        landmarkPoses_ = simulator_->landmarkPosesTrue();
        wayPoints_ = simulator_->waypointsTrue();
    }

SimulatorView::SimulatorView(IncrementalSimulator* sim, const std::string& filename):
    simulator_(sim), View(Eigen::Vector3f(0.0,1.0,0.0)), lmColor_(Eigen::Vector3f(0.0,1.0,0.5)), wpColor_(Eigen::Vector3f(1.0,5.0,0.0)),
    lmSize_(0.5), wpSize_(0.5) {
        landmarkPoses_ = simulator_->landmarkPosesTrue();
        wayPoints_ = simulator_->waypointsTrue();
        setView(filename);
    }


void SimulatorView::pause(){
    View::pause();
}

void SimulatorView::update() {
    updateRobotPose(simulator_->xTrue().toVector());
}

void SimulatorView::setView(const std::string& filename){
    std::ifstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open Simulator config file: " + filename);
    }
    nlohmann::json j;
    f >> j;

    auto color = j["simulator"]["platform"].value("color", std::vector<float>{0.0f,1.0f,0.0f});
    color_ = Eigen::Vector3f(color[0], color[1], color[2]);
    platformSize_ = j["simulator"]["platform"].value("size", 1.5);

    auto lmColor = j["simulator"]["landmark"].value("color", std::vector<float>{0.0f,1.0f,0.5f});
    lmColor_ = Eigen::Vector3f(lmColor[0], lmColor[1], lmColor[2]);
    lmSize_ = j["simulator"]["landmark"].value("size", 0.5);

    auto wpColor = j["simulator"]["waypoint"].value("color", std::vector<float>{1.0f,0.5f,0.0f});
    wpColor_ = Eigen::Vector3f(wpColor[0], wpColor[1], wpColor[2]);
    wpSize_ = j["simulator"]["waypoint"].value("size", 0.5);

    auto measColor = j["slam_system"]["measurment"].value("color", std::vector<float>{0.0f,1.0f,0.5f});
    measColor_ = Eigen::Vector3f(measColor[0], measColor[1], measColor[2]);

    // measurment colors
    auto gpsColor = j["slam_system"]["measurment"].value("gps_color", std::vector<float>{0.0f,1.0f,0.5f});
    gpsMeasColor_ = Eigen::Vector3f(gpsColor[0], gpsColor[1], gpsColor[2]);

    auto lmrbMeasColor = j["slam_system"]["measurment"].value("lm_range_bearing_color", std::vector<float>{0.0f,1.0f,0.5f});
    lmrbMeasColor_ = Eigen::Vector3f(lmrbMeasColor[0], lmrbMeasColor[1], lmrbMeasColor[2]);

    auto lmobMeasColor = j["slam_system"]["measurment"].value("lm_observation_color", std::vector<float>{0.0f,1.0f,0.5f});
    lmobMeasColor_ = Eigen::Vector3f(lmobMeasColor[0], lmobMeasColor[1], lmobMeasColor[2]);
}


void SimulatorView::processEvents(EventPtrVector& events){
    updateRobotPose(simulator_->xTrue().toVector());
    for (const auto& event : events) {
      switch(event->type()){
        case Event::EventType::LandmarkObservations:{
            LandmarkObservationsEvent& lmObsEvent = static_cast<LandmarkObservationsEvent&>(*event);
            for(const auto& lmObs : lmObsEvent.landmarkObservations){
                double x = lmObs.value[0]*cos(pose_[2]) - lmObs.value[1]*sin(pose_[2]) + pose_[0];
                double y = lmObs.value[0]*sin(pose_[2]) + lmObs.value[1]*cos(pose_[2]) + pose_[1];
                std::shared_ptr<LineViz> lmObsViz = std::make_shared<LineViz>(Eigen::Vector2d(x,y), Eigen::Vector2d(pose_[0],pose_[1]), 25, 0,  MeasurmentViz::AttachmentType::World, lmobMeasColor_);
                measVizVertex_.push_back(lmObsViz);
            }
            break;
        }
        case Event::EventType::LMRangeBearingObservations:{
            LMRangeBearingObservationsEvent& lmObsEvent = static_cast<LMRangeBearingObservationsEvent&>(*event);
            for(const auto& lmObs : lmObsEvent.landmarkObservations){
                double x = lmObs.value[0] * cos(lmObs.value[1] + pose_[2]) + pose_[0];
                double y = lmObs.value[0] * sin(lmObs.value[1] + pose_[2]) + pose_[1];
                std::shared_ptr<LineViz> lmObsViz = std::make_shared<LineViz>(Eigen::Vector2d(x,y), pose_.head<2>(), 25, 0,  MeasurmentViz::AttachmentType::World, lmrbMeasColor_);
                measVizVertex_.push_back(lmObsViz);
            }
            break;
        }
        case Event::EventType::GPSObservation:{
            GPSObservationEvent& gpsEvent = static_cast<GPSObservationEvent&>(*event);
            std::shared_ptr<CircleViz> gpsViz = std::make_shared<CircleViz>(gpsEvent.value, sqrt(gpsEvent.covariance(0,0))*2.0, 25, 0,  MeasurmentViz::AttachmentType::World, gpsMeasColor_);
            measVizVertex_.push_back(gpsViz);
            break;
        }
        default:
            break;
      }
    }
}



void SimulatorView::renderScene() const{
    // Draw landmarks as crosses
    glColor3f(lmColor_[0], lmColor_[1], lmColor_[2]);
    const double crossSize = lmSize_;

    for (const auto& lm : landmarkPoses_) {
        double x = lm[0];
        double y = lm[1];
        glBegin(GL_LINES);
        glVertex2d(x - crossSize, y - crossSize);
        glVertex2d(x + crossSize, y + crossSize);
        glVertex2d(x - crossSize, y + crossSize);
        glVertex2d(x + crossSize, y - crossSize);
        glEnd();
    }

    // Draw waypoints as circles
    glColor3f(wpColor_[0], wpColor_[1], wpColor_[2]);
    const double radius = wpSize_;
    const int numSegments = 20;

    for (const auto& wp : wayPoints_) {
        double x = wp[0];
        double y = wp[1];
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < numSegments; ++i) {
            double theta = 2.0 * M_PI * double(i) / double(numSegments);
            double dx = radius * cos(theta);
            double dy = radius * sin(theta);
            glVertex2d(x + dx, y + dy);
        }
        glEnd();
    }
}


}}}