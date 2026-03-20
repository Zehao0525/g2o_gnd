#pragma once

#include "view_manager.h"

namespace g2o {
namespace tutorial {
namespace viz {

class ViewManager3D : public ViewManager {
public:
    explicit ViewManager3D(const std::string& filename);
    ~ViewManager3D() = default;

    void start() override;
    void stop() override;

protected:
    void renderLoop() override;
    void initialize3DConfig(const std::string& filename);

    std::vector<double> cameraUp_;
};

}}}

