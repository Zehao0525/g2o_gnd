#ifndef G2O_INCSIN2D_EVENTS_H
#define G2O_INCSIN2D_EVENTS_H

#include <map>
#include <vector>
#include <memory>
#include <set>

#include "g2o_tutorial_slam2d_api.h"
#include "se2.h"
#include "sensor_data.h"

namespace g2o {
namespace tutorial {


    struct G2O_TUTORIAL_SLAM2D_API Event {
        double time;
        enum class EventType {
            HeartBeat,
            LMRangeBearingObservations,
            LandmarkObservations,
            GPSObservation,
            Odometry,
            Initialization
        };

        Event(double t) : time(t) {};
        virtual ~Event() = default;
        virtual EventType type() const = 0;
        
    };

    struct G2O_TUTORIAL_SLAM2D_API HeartBeat : public Event{
        HeartBeat(const double timestamp): Event(timestamp) { };
        EventType type() const override{return EventType::HeartBeat;};
    };


    struct G2O_TUTORIAL_SLAM2D_API LandmarkObservationsEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        LandmarkObservationVector landmarkObservations;

        LandmarkObservationsEvent(double timestamp, LandmarkObservationVector obsertations)
            : Event(timestamp), landmarkObservations(obsertations) {}

        EventType type() const override {
            return EventType::LandmarkObservations;
        }
    };


    struct G2O_TUTORIAL_SLAM2D_API LMRangeBearingObservationsEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        LMRangeBearingObservationVector landmarkObservations;

        LMRangeBearingObservationsEvent(double timestamp, LMRangeBearingObservationVector obsertations)
            : Event(timestamp), landmarkObservations(obsertations) {}

        EventType type() const override {
            return EventType::LMRangeBearingObservations;
        }
    };
    

    struct G2O_TUTORIAL_SLAM2D_API OdometryEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        SE2 value;
        Eigen::Matrix3d covariance;
        OdometryEvent(  const double timestamp,
                        const SE2& velosity,
                        const Eigen::Matrix3d& cov): Event(timestamp), value(velosity), covariance(cov) { };
        EventType type() const override{return EventType::Odometry;};
    };


    struct G2O_TUTORIAL_SLAM2D_API InitializationEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        SE2 pose;
        SE2 velocity;
        Eigen::Matrix3d covariance;
        Eigen::Matrix3d sigmaU;
        InitializationEvent(const double timestamp,
                        const SE2& pos,
                        const SE2& vel,
                        const Eigen::Matrix3d& posCov,
                        const Eigen::Matrix3d& sigmau): Event(timestamp), pose(pos), velocity(vel), covariance(posCov), sigmaU(sigmau) {  };
        EventType type() const override{return EventType::Initialization;};
    };


    struct G2O_TUTORIAL_SLAM2D_API GPSObservationEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Eigen::Vector2d value;
        Eigen::Matrix3d covariance;
        InitializationEvent(const double timestamp,
                        const Eigen::Vector2d& pos,
                        const Eigen::Matrix3d& cov): Event(timestamp), value(pos), covariance(cov) {  };
        EventType type() const override{return EventType::GPSObservation;};
    };


    using EventPtr = std::shared_ptr<Event>;
    using EventPtrVector = std::vector<EventPtr>;

    // Comparator: earlier time comes first
    struct EventCompare {
        bool operator()(const EventPtr& a, const EventPtr& b) const {
            return a->time < b->time;
        }
    };

    // Internal container: automatically sorted by time
    using EventSet = std::multiset<EventPtr, EventCompare>;
}
}

#endif
