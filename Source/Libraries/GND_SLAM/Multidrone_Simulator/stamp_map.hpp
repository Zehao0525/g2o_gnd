#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>



namespace g2o {
namespace tutorial {
namespace multibotsim{
class StampMap {
public:
    using Time     = double;       // adjust if needed
    using VertexId = int;//std::size_t;  // your ID type

private:
    struct Entry {
        Time     t;
        VertexId id;
    };

    std::vector<Entry> data_;   // sorted by timestamp, appended in order
    Time   t0_      = 0.0;      // timestamp of first entry
    double dt_avg_  = 0.0;      // running average Δt
    size_t count_   = 0;        // number of intervals contributing to dt_avg_

    // How far around the estimated index we check (local search)
    static constexpr int WINDOW = 4;

public:
    StampMap() = default;

    // Insert a new (timestamp, id) pair. Timestamps must be monotonic.
    void add(Time t, VertexId id)
    {
        if (data_.empty()) {
            t0_ = t;
        } else {
            Time last_t = data_.back().t;
            if (t < last_t) {
                throw std::runtime_error("StampMap::add(): timestamps must be non-decreasing.");
            }
            double dt = t - last_t;
            if (dt > 0) {  // update running average only when dt > 0
                dt_avg_ = (dt_avg_ * count_ + dt) / (count_ + 1);
                count_++;
            }
        }
        data_.push_back({t, id});
    }

    // Query the id whose timestamp is closest to query_t.
    VertexId query(Time query_t) const
    {
        if (data_.empty()) {
            throw std::runtime_error("StampMap::query(): no entries.");
        }
        if (data_.size() == 1 || dt_avg_ <= 0.0) {
            // Degenerate case: just return the only entry.
            return data_.front().id;
        }

        // Estimate index using linear interpolation
        double k_real = (query_t - t0_) / dt_avg_;
        long k = static_cast<long>(std::llround(k_real));

        // Clamp to valid range
        k = std::max<long>(0, std::min<long>(k, data_.size() - 1));

        // Local window search to correct jitter
        long best_k = k;
        double best_dist = std::abs(data_[k].t - query_t);

        long start = std::max<long>(0, k - WINDOW);
        long end   = std::min<long>(data_.size() - 1, k + WINDOW);

        for (long i = start; i <= end; ++i) {
            double d = std::abs(data_[i].t - query_t);
            if (d < best_dist) {
                best_dist = d;
                best_k = i;
            }
        }

        return data_[best_k].id;
    }

    // Convenience: return size
    size_t size() const noexcept { return data_.size(); }

    // Convenience: access raw data if needed
    const std::vector<Entry>& data() const noexcept { return data_; }
};

}}}
