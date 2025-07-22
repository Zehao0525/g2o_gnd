#include "view_manager.h"
#include <pangolin/pangolin.h>


namespace g2o {
namespace tutorial {
namespace viz {

ViewManager::ViewManager() : running_(false), vizType_(VizType::PoseOnly) {}

void ViewManager::addView(std::shared_ptr<View> view) {
    views_.push_back(view);
}

void ViewManager::start() {
    running_ = true;
    renderThread_ = std::thread(&ViewManager::renderLoop, this);
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
        pangolin::ProjectionMatrix(800, 600, 500, 500, 400, 300, 0.1, 100),
        pangolin::ModelViewLookAt(25, 25, 65, 25, 25, 0, pangolin::AxisY)
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
            
        }

        pangolin::FinishFrame();
    }
}

}}}