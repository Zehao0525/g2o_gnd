#include "slam_visualizer.h"

namespace g2o {
namespace tutorial {
namespace viz {

Visualizer::Visualizer() : running_(false), vizType_(Viztype::PoseOnly) {}

Visualizer::~Visualizer() {
    stop();
}

void Visualizer::start() {
    if(running_){
        std::cout << "visualizer already running" << std::endl;
        return;
    }
    running_ = true;
    renderThread_ = std::thread(&Visualizer::renderLoop, this);
}

void Visualizer::setVizType(std::string typeKey){
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


void Visualizer::start() {
    running_ = true;
    renderThread_ = std::thread(&Visualizer::renderLoop, this);
}

void Visualizer::stop() {
    running_ = false;
    if (renderThread_.joinable())
        renderThread_.join();
}

void Visualizer::updateRobotPath(const std::vector<Eigen::Vector3d>& path) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    robotPath_ = path;
}

void Visualizer::appendToRobotPath(const Eigen::Vector3d& pose) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    robotPath_.emplace_back(path);
}

void Visualizer::updateLandmarks(const std::vector<Eigen::Vector2d>& lms) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    landmarks_ = lms;
}

void Visualizer::renderLoop() {
    pangolin::CreateWindowAndBind("SLAM Viewer", 800, 600);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(800, 600, 500, 500, 400, 300, 0.1, 100),
        pangolin::ModelViewLookAt(0, 0, 10, 0, 0, 0, pangolin::AxisY)
    );

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -800.0f / 600.0f)
        .SetHandler(new pangolin::Handler2D(s_cam));

    while (!pangolin::ShouldQuit() && running_) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        {
            std::lock_guard<std::mutex> lock(dataMutex_);

            // Draw landmarks
            glPointSize(5.0f);
            glBegin(GL_POINTS);
            glColor3f(0.0, 1.0, 0.0);
            for (const auto& lm : landmarks_) {
                glVertex2d(lm[0], lm[1]);
            }
            glEnd();

            // Draw robot path

            switch(viztype_){
                case Viztype::PoseOnly:
                    renderRobotPose(robotPath_.back());
                    break;
                case VizType::Path:
                    renderRobotPath(robotPath_);
                    renderRobotPath(robotPath_.back());
                    break;
                default:
                    break;
            }
        }

        pangolin::FinishFrame();
    }
}


void Visualizer::renderRobotPose(const Eigen::Vector3d& currentPose) {
    const double arrowSize = 0.3;   // Length from tip to base
    const double arrowWidth = 0.15; // Width of the base

    double x = currentPose[0];
    double y = currentPose[1];
    double theta = currentPose[2];

    // Triangle vertices
    Eigen::Vector2d tip(x, y);
    Eigen::Vector2d baseCenter = tip - arrowSize * Eigen::Vector2d(std::cos(theta), std::sin(theta));

    Eigen::Vector2d baseLeft = baseCenter + (arrowWidth / 2.0) * Eigen::Vector2d(-std::sin(theta), std::cos(theta));
    Eigen::Vector2d baseRight = baseCenter + (arrowWidth / 2.0) * Eigen::Vector2d(std::sin(theta), -std::cos(theta));

    // Draw triangle
    glColor3f(1.0f, 0.0f, 0.0f); // Red arrow
    glBegin(GL_TRIANGLES);
    glVertex2d(tip[0], tip[1]);
    glVertex2d(baseLeft[0], baseLeft[1]);
    glVertex2d(baseRight[0], baseRight[1]);
    glEnd();
}


void Visualizer::renderRobotPath(const std::vector<Eigen::Vector3d>& robotPath) {
    if (robotPath.size() < 2) return;

    glColor3f(0.0f, 0.0f, 1.0f); // Blue path
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    for (const auto& pose : robotPath) {
        glVertex2d(pose[0], pose[1]);
    }
    glEnd();
}


}  // namespace viz
}  // namespace tutorial
}  // namespace g2o
