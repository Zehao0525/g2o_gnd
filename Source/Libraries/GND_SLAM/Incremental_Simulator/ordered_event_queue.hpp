#ifndef G2O_INCSIN2D_ORDEREDEVENTQUEUE_H
#define G2O_INCSIN2D_ORDEREDEVENTQUEUE_H

#include <memory>
#include <set>
#include "events.h"  // for Event base class

namespace g2o {
namespace tutorial {

class OrderedEventQueue {
public:

    // Push a new event
    void push(const EventPtr& event) {
        events_.insert(event);
    }

    // Pop the earliest event
    EventPtr pop() {
        if (events_.empty()) return nullptr;
        auto it = events_.begin();
        EventPtr earliest = *it;
        events_.erase(it);
        return earliest;
    }

    // Return all events in order
    std::vector<EventPtr> orderedEvents() const {
        return std::vector<EventPtr>(events_.begin(), events_.end());
    }

    // Clear all events
    void clear() {
        events_.clear();
    }

    // Check if empty
    bool empty() const {
        return events_.empty();
    }

    // Get size
    size_t size() const {
        return events_.size();
    }

private:
    EventSet events_;
};

} // namespace tutorial
} // namespace g2o

#endif
