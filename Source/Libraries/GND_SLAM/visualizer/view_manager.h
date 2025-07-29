#pragma once
#include <vector>
#include <thread>
#include <memory>
#include "view.h"
#include <pangolin/pangolin.h>
#include <pangolin/handler/handler.h>

namespace g2o {
namespace tutorial {
namespace viz {

enum class VizType {
    PoseOnly,
    Path
};

class ViewManager {
public:
    ViewManager();
    ~ViewManager() = default;


    void addView(std::shared_ptr<View> view);
    void setVizType(std::string typeKey);
    void start();
    void stop();
    void pause();

private:
    void renderLoop();

    std::vector<std::shared_ptr<View>> views_;
    std::thread renderThread_;
    bool running_;

    VizType vizType_;
};

}}}