// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "types_tutorial_slam2d.h"
#include "edge_se2_pointxy.h"

#include <iostream>

#include "g2o/core/factory.h"
#include "g2o/stuff/macros.h"


namespace g2o {
namespace tutorial {

// -> Statically compiled might be the trick (dynamic building might be the cause. To check)
// Object construction
void forceLinkTypesTutorialSlam2d() {
    std::cerr << "✅ Registering CacheSE2Offset...\n";
}

//G2O_REGISTER_TYPE_GROUP(tutorial_slam2d);

G2O_REGISTER_TYPE_NAME("TUTORIAL_VERTEX_SE2", VertexSE2);
G2O_REGISTER_TYPE_NAME("TUTORIAL_VERTEX_POINT_XY", VertexPointXY);

G2O_REGISTER_TYPE_NAME("TUTORIAL_PARAMS_SE2_OFFSET", ParameterSE2Offset);

G2O_REGISTER_TYPE_NAME("TUTORIAL_CACHE_SE2_OFFSET", CacheSE2Offset);

G2O_REGISTER_TYPE_NAME("TUTORIAL_EDGE_SE2", EdgeSE2);
G2O_REGISTER_TYPE_NAME("TUTORIAL_EDGE_SE2_POINT_XY", EdgeSE2PointXY);
}  // namespace tutorial
}  // namespace g2o
