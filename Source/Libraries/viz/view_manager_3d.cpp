#include "view_manager_3d.h"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <thread>

#include <pangolin/pangolin.h>

namespace g2o {
namespace tutorial {
namespace viz {

namespace {

pangolin::AxisDirection inferEnforceUp(const std::vector<double>& u) {
    if (u.size() < 3) return pangolin::AxisNone;
    double nx = u[0], ny = u[1], nz = u[2];
    const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (norm < 1e-9) return pangolin::AxisNone;
    nx /= norm;
    ny /= norm;
    nz /= norm;
    if (std::abs(nz) >= 0.9) return pangolin::AxisZ;
    if (std::abs(ny) >= 0.9) return pangolin::AxisY;
    return pangolin::AxisNone;
}

}  // namespace

ViewManager3D::ViewManager3D(const std::string& filename) : ViewManager(filename) {
    initialize3DConfig(filename);
}

ViewManager3D::~ViewManager3D() {
    // Ensure we never trigger std::terminate due to a still-joinable thread.
    running_ = false;
    if (renderThread_.joinable()) {
        // Avoid self-join (shouldn't happen, but keep it safe).
        if (renderThread_.get_id() != std::this_thread::get_id()) {
            renderThread_.join();
        }
    }
}

void ViewManager3D::initialize3DConfig(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open Simulator config file: " + filename);
    }
    nlohmann::json j;
    f >> j;

    const auto& cs = j["camera_setting"];

    // Optional camera override (so you can customize without touching `look_at`).
    // Format:
    //   camera_eye:    [eye_x, eye_y, eye_z]
    //   camera_target: [target_x, target_y, target_z]
    if (cs.contains("camera_eye") && cs.contains("camera_target")) {
        const auto eye = cs["camera_eye"].get<std::vector<double>>();
        const auto target = cs["camera_target"].get<std::vector<double>>();
        if (eye.size() >= 3 && target.size() >= 3) {
            cameraLookat_ = {eye[0], eye[1], eye[2], target[0], target[1], target[2]};
        }
    }

    cameraUp_ = cs.value("camera_up", std::vector<double>{0.0, 0.0, 1.0});

    finalRenderPauseSec_ = j.value("final_render_pause_sec", 0.0);
    finalPauseRequested_ = false;
}

void ViewManager3D::start() {
    running_ = true;
    renderThread_ = std::thread(&ViewManager3D::renderLoop, this);
}

void ViewManager3D::stop() {
    finalPauseRequested_ = true;
    running_ = false;
    if (renderThread_.joinable()) {
        renderThread_.join();
    }
}

void ViewManager3D::renderLoop() {
    pangolin::CreateWindowAndBind("SLAM Visualization", 800, 600);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    pangolin::OpenGlMatrix modelView;
    if (cameraLookat_.size() >= 6 && cameraUp_.size() >= 3) {
        const double dx = cameraLookat_[3] - cameraLookat_[0];
        const double dy = cameraLookat_[4] - cameraLookat_[1];
        const double dz = cameraLookat_[5] - cameraLookat_[2];
        const double lx = std::sqrt(dx * dx + dy * dy + dz * dz);

        double ux = cameraUp_[0];
        double uy = cameraUp_[1];
        double uz = cameraUp_[2];
        const double ul = std::sqrt(ux * ux + uy * uy + uz * uz);

        if (lx > 1e-9 && ul > 1e-9) {
            const double cosang = std::abs((dx * ux + dy * uy + dz * uz) / (lx * ul));
            if (cosang > 0.999) {
                // Avoid ModelViewLookAt invalid_argument when look and up are parallel.
                ux = 0.0;
                uy = 1.0;
                uz = 0.0;
            }
        } else {
            ux = 0.0;
            uy = 1.0;
            uz = 0.0;
        }

        modelView = pangolin::ModelViewLookAt(
            cameraLookat_[0], cameraLookat_[1], cameraLookat_[2],
            cameraLookat_[3], cameraLookat_[4], cameraLookat_[5],
            ux, uy, uz);
    } else {
        modelView = pangolin::ModelViewLookAt(
            cameraLookat_[0], cameraLookat_[1], cameraLookat_[2],
            cameraLookat_[3], cameraLookat_[4], cameraLookat_[5],
            pangolin::AxisY);
    }

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(
            projectionMatrix_[0], projectionMatrix_[1], projectionMatrix_[2], projectionMatrix_[3],
            projectionMatrix_[4], projectionMatrix_[5], projectionMatrix_[6], projectionMatrix_[7]),
        modelView);

    pangolin::View& d_cam = pangolin::CreateDisplay();
    d_cam.SetBounds(0.0, 1.0, 0.0, 1.0, -800.0f / 600.0f);
    d_cam.SetHandler(new pangolin::Handler3D(s_cam, inferEnforceUp(cameraUp_)));

    while (running_ && !pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        for (auto& view : views_) {
            switch (vizType_) {
                case VizType::Path:
                    view->renderRobotPath();
                    break;
                default:
                    break;
            }
            view->renderScene();
            view->renderMeasurmentViz();
        }

        pangolin::FinishFrame();
    }

    // One last render after an explicit `stop()` so the last state is visible for a moment.
    if (finalPauseRequested_ && finalRenderPauseSec_ > 1e-12 && !pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        for (auto& view : views_) {
            switch (vizType_) {
                case VizType::Path:
                    view->renderRobotPath();
                    break;
                default:
                    break;
            }
            view->renderScene();
            view->renderMeasurmentViz();
        }
        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::duration<double>(finalRenderPauseSec_));
    }
}

}}}

