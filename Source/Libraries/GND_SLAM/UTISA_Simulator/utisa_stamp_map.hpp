#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace g2o {
namespace tutorial {
namespace multibotsim {

class StampMap {
 public:
  using Time = double;
  using VertexId = int;

 private:
  struct Entry {
    Time t;
    VertexId id;
  };

  std::vector<Entry> data_;
  Time t0_ = 0.0;
  double dt_avg_ = 0.0;
  size_t count_ = 0;

  static constexpr int WINDOW = 4;

  /// O(log n) nearest timestamp; used when the ±WINDOW heuristic may have missed.
  VertexId queryExactNearest(Time query_t) const {
    auto it = std::lower_bound(data_.begin(), data_.end(), query_t,
                               [](const Entry& e, Time qt) { return e.t < qt; });
    if (it == data_.begin()) return it->id;
    if (it == data_.end()) return data_.back().id;
    const auto prev = std::prev(it);
    const double d_after = it->t - query_t;
    const double d_before = query_t - prev->t;
    return (d_before <= d_after) ? prev->id : it->id;
  }

 public:
  StampMap() = default;

  void add(Time t, VertexId id) {
    if (data_.empty()) {
      t0_ = t;
    } else {
      Time last_t = data_.back().t;
      if (t < last_t) {
        throw std::runtime_error("StampMap::add(): timestamps must be non-decreasing.");
      }
      double dt = t - last_t;
      if (dt > 0) {
        dt_avg_ = (dt_avg_ * count_ + dt) / (count_ + 1);
        count_++;
      }
    }
    data_.push_back({t, id});
  }

  // Fast path: hint from mean Δt + local ±WINDOW scan. Fail-safe: if the best index
  // lies on the window edge, the true nearest can lie outside → exact search.
  VertexId query(Time query_t) const {
    if (data_.empty()) {
      throw std::runtime_error("StampMap::query(): no entries.");
    }
    if (data_.size() == 1) {
      return data_.front().id;
    }
    if (dt_avg_ <= 0.0) {
      return queryExactNearest(query_t);
    }

    double k_real = (query_t - t0_) / dt_avg_;
    long k = static_cast<long>(std::llround(k_real));
    k = std::max<long>(0, std::min<long>(k, static_cast<long>(data_.size() - 1)));

    long best_k = k;
    double best_dist = std::abs(data_[static_cast<size_t>(k)].t - query_t);

    long start = std::max<long>(0, k - WINDOW);
    long end = std::min<long>(static_cast<long>(data_.size() - 1), k + WINDOW);

    for (long i = start; i <= end; ++i) {
      double d = std::abs(data_[static_cast<size_t>(i)].t - query_t);
      if (d < best_dist) {
        best_dist = d;
        best_k = i;
      }
    }

    // If we are on either left or right edge, we suspect a fail and we query the exact nearest.
    // Unless start is the first element or end is the last element.
    const bool on_left_edge = (best_k == start && start > 0);
    const bool on_right_edge =
        (best_k == end && end < static_cast<long>(data_.size() - 1));
    if (on_left_edge || on_right_edge) {
      return queryExactNearest(query_t);
    }
    return data_[static_cast<size_t>(best_k)].id;
  }

  size_t size() const noexcept { return data_.size(); }

  const std::vector<Entry>& data() const noexcept { return data_; }
};

}  // namespace multibotsim
}  // namespace tutorial
}  // namespace g2o
