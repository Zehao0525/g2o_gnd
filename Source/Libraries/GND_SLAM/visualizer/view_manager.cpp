#include "view_manager.h"
#include <pangolin/pangolin.h>


namespace g2o {
namespace tutorial {
namespace viz {

ViewManager::ViewManager(const std::string& filename) : running_(false), vizType_(VizType::PoseOnly) {
    std::ifstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open Simulator config file: " + filename);
    }
    nlohmann::json j;
    f >> j;

    std::cout << "ViewManager_setup" <<std::endl;
    cameraLookat_ = j["camera_setting"].value("look_at", std::vector<double>{25.0,25.0,60.0,25.0,25.0,0.0});
    projectionMatrix_ = j["camera_setting"].value("projection_matrix", std::vector<double>{25.0,25.0,60.0,25.0,25.0,0.0});

}

void ViewManager::addView(std::shared_ptr<View> view) {
    views_.push_back(view);
}

void ViewManager::start() {
    running_ = true;
    renderThread_ = std::thread(&ViewManager::renderLoop, this);
}

void ViewManager::pause(){
    for (auto& view : views_) {
        view->pause();
    }
}

void ViewManager::stop() {
    running_ = false;
    if (renderThread_.joinable()) {
        renderThread_.join();
    }
}

void ViewManager::setVizType(std::string typeKey){
    std::transform(typeKey.begin(), typeKey.end(), typeKey.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (typeKey == "poseonly") {
        vizType_ = VizType::PoseOnly;
    } else if (typeKey == "path") {
        vizType_ = VizType::Path;
    } else {
        std::cerr << "[Visualizer] Unknown visualization type: " << typeKey 
                  << ". Using default (PoseOnly)." << std::endl;
        vizType_ = VizType::PoseOnly;
    }
}

void ViewManager::renderLoop() {
    pangolin::CreateWindowAndBind("SLAM Visualization", 800, 600);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        //pangolin::ProjectionMatrix(800, 600, 500, 500, 400, 300, 0.1, 100),
        pangolin::ProjectionMatrix(projectionMatrix_[0], projectionMatrix_[1], projectionMatrix_[2], projectionMatrix_[3], projectionMatrix_[4], projectionMatrix_[5], projectionMatrix_[6], projectionMatrix_[7]),
        //pangolin::ModelViewLookAt(25, 25, 65, 25, 25, 0, pangolin::AxisY)
        pangolin::ModelViewLookAt(cameraLookat_[0], cameraLookat_[1], cameraLookat_[2], cameraLookat_[3], cameraLookat_[4], cameraLookat_[5], pangolin::AxisY)
    );

    pangolin::View& d_cam = pangolin::CreateDisplay();
    d_cam.SetBounds(0.0, 1.0, 0.0, 1.0, -800.0f / 600.0f);
    d_cam.SetHandler(new pangolin::Handler3D(s_cam));

    while (running_ && !pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        for (auto& view : views_) {
            //view->update();
            switch(vizType_){
                case VizType::Path:
                    view->renderRobotPath();
                    break;
                default:
                    break;
            }
            view->renderRobotPose();
            view->renderScene();
            view->renderMeasurmentViz();
            
        }

        pangolin::FinishFrame();
    }
}

}}}