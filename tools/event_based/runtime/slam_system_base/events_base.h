#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "g2o_tutorial_slam2d_api.h"

namespace g2o {
namespace tutorial {

/**
 * Slim polymorphic event base. Domain-specific payloads live in separate headers
 * (e.g. events_comp.h, events_glenn.h, utisa_events.h). Dispatch with dynamic_cast.
 */
struct G2O_TUTORIAL_SLAM2D_API Event {
  double time;
  /// Monotonic sequence for stable ordering when time and priority tie (e.g. file order).
  std::uint64_t tieOrder{0};

  explicit Event(double t) : time(t) {}
  virtual ~Event() = default;

  /// Lower runs first when timestamps are equal (see EventCompare).
  virtual int sortPriority() const { return 100; }
};

using EventPtr = std::shared_ptr<Event>;
using EventPtrVector = std::vector<EventPtr>;

// Comparator: time first; then sortPriority; then tieOrder; pointer last.
struct EventCompare {
  bool operator()(const EventPtr& a, const EventPtr& b) const {
    if (a->time != b->time) {
      return a->time < b->time;
    }
    const int pa = a->sortPriority();
    const int pb = b->sortPriority();
    if (pa != pb) {
      return pa < pb;
    }
    if (a->tieOrder != b->tieOrder) {
      return a->tieOrder < b->tieOrder;
    }
    return a.get() < b.get();
  }
};

using EventSet = std::multiset<EventPtr, EventCompare>;

}  // namespace tutorial
}  // namespace g2o
