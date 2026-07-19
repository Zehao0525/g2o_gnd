#ifndef G2O_INCSIN2D_EVENTS_H
#define G2O_INCSIN2D_EVENTS_H

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "g2o_tutorial_slam2d_api.h"
#include "se2.h"
#include "sensor_data.h"

#include "g2o/types/slam3d/se3quat.h"

namespace g2o {
namespace tutorial {


    struct G2O_TUTORIAL_SLAM2D_API Event {
        double time;
        /// Monotonic sequence for stable ordering when time and priority tie (e.g. file order).
        std::uint64_t tieOrder{0};
        enum class EventType {
            HeartBeat,
            LMRangeBearingObservations,
            LandmarkObservations,
            GPSObservation,
            Odometry,
            Initialization,
            FileInitialization,
            FileObservation,
            FileIntraObservation,
            FileOdometry,
            Other
        };

        Event(double t) : time(t) {}
        virtual ~Event() = default;
        virtual EventType type() const = 0;
        /// Lower runs first when timestamps are equal (see `EventCompare`).
        virtual int sortPriority() const { return 100; }
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
        Eigen::Matrix2d covariance;
        GPSObservationEvent(const double timestamp,
                        const Eigen::Vector2d& pos,
                        const Eigen::Matrix2d& cov): Event(timestamp), value(pos), covariance(cov) {  };
        EventType type() const override{return EventType::GPSObservation;};
    };


    struct G2O_TUTORIAL_SLAM2D_API FileInitEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        int vtxId;
        Isometry3 value;
        Eigen::Matrix<double,6,6> information;
        FileInitEvent(  const double eventTime,
                        const int vtxId,
                        const Isometry3& pos,
                        const Eigen::Matrix<double,6,6>& info): Event(eventTime), vtxId(vtxId), value(pos), information(info) {  };
        EventType type() const override{return EventType::FileInitialization;};
    };

    struct G2O_TUTORIAL_SLAM2D_API FileOdomEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        int vtxId;
        Isometry3 value;
        Eigen::Matrix<double,6,6> information;
        FileOdomEvent(  const double eventTime,
                        const int vtxId,
                        const Isometry3& pos,
                        const Eigen::Matrix<double,6,6>& info): Event(eventTime), vtxId(vtxId), value(pos), information(info) {  };
        EventType type() const override{return EventType::FileOdometry;};
    };


    struct G2O_TUTORIAL_SLAM2D_API FileObsEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        int robotIdFrom;
        int robotIdTo;
        int vtxIdFrom;
        int vtxIdTo;
        Isometry3 value;
        Eigen::Matrix<double,6,6> information;
        FileObsEvent(   const double eventTime,
                        const int robotId,
                        const int targetRobotId,
                        const int vtxId0,
                        const int vtxId1,
                        const Isometry3& pos,
                        const Eigen::Matrix<double,6,6>& info): Event(eventTime), 
                        robotIdFrom(robotId), robotIdTo(targetRobotId),vtxIdFrom(vtxId0), vtxIdTo(vtxId1), value(pos), information(info) {  };
        EventType type() const override{return EventType::FileObservation;};
    };

    struct G2O_TUTORIAL_SLAM2D_API FileIntraObsEvent : public Event {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        int vtxIdFrom;
        int vtxIdTo;
        Isometry3 value;
        Eigen::Matrix<double,6,6> information;
        FileIntraObsEvent(   const double eventTime,
                        const int vtxId0,
                        const int vtxId1,
                        const Isometry3& pos,
                        const Eigen::Matrix<double,6,6>& info): Event(eventTime), vtxIdFrom(vtxId0), vtxIdTo(vtxId1), value(pos), information(info) {  };
        EventType type() const override{return EventType::FileIntraObservation;};
    };
    using EventPtr = std::shared_ptr<Event>;
    using EventPtrVector = std::vector<EventPtr>;


    // Comparator: time first; then sortPriority (e.g. DataEventBase: init<odom<obs<lmobs);
    // then tieOrder; pointer last (non-deterministic — avoid relying on it).
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

    // Internal container: automatically sorted by time
    using EventSet = std::multiset<EventPtr, EventCompare>;
}
}

#endif
