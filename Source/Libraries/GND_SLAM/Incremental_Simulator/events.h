#ifndef G2O_INCSIN2D_EVENTS_H
#define G2O_INCSIN2D_EVENTS_H

#include <map>
#include <vector>
#include <memory>

#include "g2o_tutorial_slam2d_api.h"
#include "se2.h"
#include "sensor_data.h"

namespace g2o {
namespace tutorial {


    struct G2O_TUTORIAL_SLAM2D_API Event {
        double time;
        enum class EventType {
            HeartBeat,
            LandmarkObservation,
            LandmarkObservations,
            Odometry,
            Initialization
        };

        virtual ~Event() = default;
        virtual EventType type() const = 0;
        
    };

    struct G2O_TUTORIAL_SLAM2D_API HeartBeat : public Event{
        HeartBeat(const double timestamp)time(timestamp) { };
        EventType type() const override{return EventType::HeartBeat;};
    };


    struct G2O_TUTORIAL_SLAM2D_API LandmarkObservationEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        int landmark_id;
        Eigen::Vector2d value;
        Eigen::Matrix2d covariance;
        LandmarkObservationEvent(const double timestamp,
                                const Eigen::Vector2d& obs,
                                const Eigen::Matrix2d& cov,
                                int id): value(obs), covariance(cov), time(timestamp), landmark_id(id) { };

        EventType type() const override{return EventType::LandmarkObservation;};
    };s

    struct G2O_TUTORIAL_SLAM2D_API LandmarkObservationsEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        LandmarkObservationVector landmarkObservations;

        LandmarkObservationsEvent(double timestamp, LandmarkObservationVector obsertations)
            : landmarkObservations(obsertations), time(timestamp) {}

        EventType type() const override {
            return EventType::LandmarkObservations;
        }
    };
    

    struct G2O_TUTORIAL_SLAM2D_API OdometryEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        SE2 value;
        Eigen::Matrix3d covariance;
        OdometryEvent(  const double timestamp,
                        const SE2& velosity,
                        const Eigen::Matrix3d& cov): time(timestamp), value(velosity), covariance(cov) { };
        EventType type() const override{return EventType::Odometry;};
    };


    struct G2O_TUTORIAL_SLAM2D_API InitializationEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        SE2 value;
        Eigen::Matrix3d covariance;
        InitializationEvent(const double timestamp,
                        const SE2& pose,
                        const Eigen::Matrix3d& cov): time(timestamp), value(pose), covariance(cov) {  };
        EventType type() const override{return EventType::Initialization;};
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
