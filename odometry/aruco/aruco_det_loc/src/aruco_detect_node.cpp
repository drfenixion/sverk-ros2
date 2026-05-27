#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "geometry_msgs/msg/point32.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "aruco_det_loc/msg/marker.hpp"
#include "aruco_det_loc/msg/marker_array.hpp"

#include "std_srvs/srv/set_bool.hpp"

#include "cv_bridge/cv_bridge.hpp"
#include "image_transport/image_transport.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>

namespace aruco_det_loc
{

using MarkerMsg      = aruco_det_loc::msg::Marker;
using MarkerArrayMsg = aruco_det_loc::msg::MarkerArray;
using SetBool        = std_srvs::srv::SetBool;

class ArucoDetectNode : public rclcpp::Node
{
public:
  ArucoDetectNode() : rclcpp::Node("aruco_detect_node")
  {
    declare_parameter<bool>("enabled", true);
    declare_parameter<double>("marker_size", 0.5); // meters
    declare_parameter<bool>("send_tf", true);
    declare_parameter<std::string>("frame_id_prefix", "aruco_");
    declare_parameter<bool>("use_map_markers", true);
    declare_parameter<std::string>("parent_frame_id", "main_camera_optical");

    declare_parameter<bool>("detect_individual_markers", true);

    // true: per-marker solvePnP, pose/TF/axes; false: pose zeros, no TF
    declare_parameter<bool>("estimate_marker_pose", true);

    // Global solvePnP rate limit per frame
    declare_parameter<bool>("pnp_fps_cap_enabled", false);
    declare_parameter<int>("pnp_fps_cap", 10); // Hz when cap > 0

    // Marker debug image
    declare_parameter<bool>("publish_debug_image", true);
    declare_parameter<std::string>("debug_image_topic", "/aruco_det/debug_image");

    declare_parameter<std::string>("debug_service_name", "toggle_debug");

    // Debug: world axes on image
    declare_parameter<bool>("publish_world_debug_image", false);
    declare_parameter<std::string>("world_debug_image_topic", "/aruco_det/world_debug_image");
    declare_parameter<std::string>("pose_cov_topic", "/aruco_map/pose_cov");
    declare_parameter<double>("world_axes_length", 0.5);
    declare_parameter<int>("world_axes_thickness", 3);

    declare_parameter<bool>("pose_cov_is_world_to_camera", false);

    // Skip stale pose_cov vs new image; <=0 disables
    declare_parameter<double>("world_pose_max_age_sec", 0.2);

    declare_parameter<bool>("publish_visualization", true);

    declare_parameter<std::string>(
      "image_topic",
      "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image");
    declare_parameter<std::string>(
      "camera_info_topic",
      "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/camera_info");
    declare_parameter<std::string>("map_markers_topic", "/map_markers");

    declare_parameter<int>("dictionary_id", 3); // cv::aruco::PREDEFINED_DICTIONARY_NAME

    this->declare_parameter<std::vector<int64_t>>(
      "size_override.ids", std::vector<int64_t>{});
    this->declare_parameter<std::vector<double>>(
      "size_override.sizes", std::vector<double>{});

    enabled_                   = get_parameter("enabled").as_bool();
    default_marker_size_       = get_parameter("marker_size").as_double();
    send_tf_                   = get_parameter("send_tf").as_bool();
    frame_id_prefix_           = get_parameter("frame_id_prefix").as_string();
    use_map_markers_           = get_parameter("use_map_markers").as_bool();
    detect_individual_markers_ = get_parameter("detect_individual_markers").as_bool();
    parent_frame_id_           =  get_parameter("parent_frame_id").as_string();
    if (parent_frame_id_.empty()) parent_frame_id_ = "main_camera_optical";

    estimate_marker_pose_      = get_parameter("estimate_marker_pose").as_bool();

    pnp_fps_cap_enabled_       = get_parameter("pnp_fps_cap_enabled").as_bool();
    pnp_fps_cap_               = get_parameter("pnp_fps_cap").as_int();
    if (pnp_fps_cap_ <= 0) pnp_fps_cap_ = 1;

    publish_debug_image_         = get_parameter("publish_debug_image").as_bool();
    debug_image_topic_           = get_parameter("debug_image_topic").as_string();

    debug_service_name_          = get_parameter("debug_service_name").as_string();

    publish_world_debug_image_   = get_parameter("publish_world_debug_image").as_bool();
    world_debug_image_topic_     = get_parameter("world_debug_image_topic").as_string();
    pose_cov_topic_              = get_parameter("pose_cov_topic").as_string();
    world_axes_length_           = get_parameter("world_axes_length").as_double();
    world_axes_thickness_        = get_parameter("world_axes_thickness").as_int();
    pose_cov_is_world_to_camera_ = get_parameter("pose_cov_is_world_to_camera").as_bool();
    world_pose_max_age_sec_      = get_parameter("world_pose_max_age_sec").as_double();

    publish_visualization_     = get_parameter("publish_visualization").as_bool();

    image_topic_       = get_parameter("image_topic").as_string();
    camera_info_topic_ = get_parameter("camera_info_topic").as_string();
    map_markers_topic_ = get_parameter("map_markers_topic").as_string();
    dictionary_id_     = get_parameter("dictionary_id").as_int();

    if (default_marker_size_ <= 0.0) default_marker_size_ = 0.16;
    if (!(world_axes_length_ > 0.0) || !std::isfinite(world_axes_length_)) world_axes_length_ = 0.5;
    if (world_axes_thickness_ <= 0) world_axes_thickness_ = 3;

    try {
      dictionary_ = cv::aruco::getPredefinedDictionary(
        static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(dictionary_id_));
    } catch (...) {
      dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    }
    detector_params_ = cv::aruco::DetectorParameters::create();

    camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
    dist_coeffs_   = cv::Mat();

    updateOverridesFromParameters();

    param_cb_handle_ = add_on_set_parameters_callback(
      std::bind(&ArucoDetectNode::onParams, this, std::placeholders::_1));

    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ArucoDetectNode::cameraInfoCb, this, std::placeholders::_1));

    updateMapMarkersSubscription();

    updatePoseCovSubscription();

    debug_srv_ = create_service<SetBool>(
      debug_service_name_,
      std::bind(&ArucoDetectNode::setDebugSrvCb, this,
                std::placeholders::_1, std::placeholders::_2));

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    markers_pub_ = create_publisher<MarkerArrayMsg>("markers", qos);

    if (publish_visualization_) {
      vis_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("visualization", qos);
    }
  }

  void initialize()
  {
    if (initialized_) return;
    initialized_ = true;

    auto self = this->shared_from_this();
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(self);

    image_transport::ImageTransport it(self);
    image_sub_ = it.subscribe(
      image_topic_, 1,
      std::bind(&ArucoDetectNode::imageCb, this, std::placeholders::_1));

    if (publish_debug_image_) {
      debug_image_pub_ = it.advertise(debug_image_topic_, 1);
    }

    if (publish_world_debug_image_) {
      world_debug_image_pub_ = it.advertise(world_debug_image_topic_, 1);
    }
  }

private:
  void updateOverridesFromParameters()
  {
    size_override_param_.clear();
    auto ids   = get_parameter("size_override.ids").as_integer_array();
    auto sizes = get_parameter("size_override.sizes").as_double_array();
    if (ids.size() != sizes.size()) return;

    for (size_t i = 0; i < ids.size(); ++i) {
      const double s = sizes[i];
      if (std::isfinite(s) && s > 0.0) {
        size_override_param_[static_cast<uint32_t>(ids[i])] = s;
      }
    }
  }

  void updateMapMarkersSubscription()
  {
    if (use_map_markers_) {
      if (!map_markers_sub_) {
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
        map_markers_sub_ = create_subscription<MarkerArrayMsg>(
          map_markers_topic_, qos,
          std::bind(&ArucoDetectNode::mapMarkersCb, this, std::placeholders::_1));
      }
    } else {
      map_markers_sub_.reset();
      map_ids_.clear();
      size_override_map_.clear();
    }
  }

  void updatePoseCovSubscription()
  {
    if (publish_world_debug_image_) {
      if (!pose_cov_sub_) {
        // Pose is computed, not raw sensor — reliable QoS OK
        auto qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
        pose_cov_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          pose_cov_topic_, qos,
          std::bind(&ArucoDetectNode::poseCovCb, this, std::placeholders::_1));
      }
    } else {
      pose_cov_sub_.reset();
      std::lock_guard<std::mutex> lk(world_pose_mutex_);
      have_world_pose_ = false;
    }
  }

  rcl_interfaces::msg::SetParametersResult onParams(const std::vector<rclcpp::Parameter>& params)
  {
    rcl_interfaces::msg::SetParametersResult r;
    r.successful = true;

    for (const auto& p : params) {
      const auto& name = p.get_name();

      if (name == "enabled") {
        enabled_ = p.as_bool();
      } else if (name == "marker_size") {
        const double v = p.as_double();
        if (v <= 0.0) { r.successful = false; r.reason = "marker_size must be > 0"; return r; }
        default_marker_size_ = v;
      } else if (name == "send_tf") {
        send_tf_ = p.as_bool();
      } else if (name == "frame_id_prefix") {
        frame_id_prefix_ = p.as_string();
      } else if (name == "map_markers_topic") {
        map_markers_topic_ = p.as_string();
        map_markers_sub_.reset();
        updateMapMarkersSubscription();
      } else if (name == "use_map_markers") {
        use_map_markers_ = p.as_bool();
        updateMapMarkersSubscription();
      } else if (name == "detect_individual_markers") {
        detect_individual_markers_ = p.as_bool();
      } else if (name == "estimate_marker_pose") {
        estimate_marker_pose_ = p.as_bool();
      } else if (name == "pnp_fps_cap_enabled") {
        pnp_fps_cap_enabled_ = p.as_bool();
        if (pnp_fps_cap_enabled_ && pnp_fps_cap_ <= 0) pnp_fps_cap_ = 1;
      } else if (name == "pnp_fps_cap") {
        const int v = p.as_int();
        if (v <= 0) { r.successful = false; r.reason = "pnp_fps_cap must be > 0"; return r; }
        pnp_fps_cap_ = v;
      }
      else if (name == "publish_debug_image") {
        publish_debug_image_ = p.as_bool();
        if (publish_debug_image_ && !debug_image_pub_) {
          auto self = this->shared_from_this();
          image_transport::ImageTransport it(self);
          debug_image_pub_ = it.advertise(debug_image_topic_, 1);
        }
      } else if (name == "debug_image_topic") {
        debug_image_topic_ = p.as_string();
        if (publish_debug_image_) {
          auto self = this->shared_from_this();
          image_transport::ImageTransport it(self);
          debug_image_pub_ = it.advertise(debug_image_topic_, 1);
        }
      }
      else if (name == "debug_service_name") {
        debug_service_name_ = p.as_string();
      }

      else if (name == "publish_world_debug_image") {
        publish_world_debug_image_ = p.as_bool();

        // recreate subscription
        pose_cov_sub_.reset();
        updatePoseCovSubscription();

        // recreate publisher
        if (publish_world_debug_image_) {
          if (!world_debug_image_pub_) {
            auto self = this->shared_from_this();
            image_transport::ImageTransport it(self);
            world_debug_image_pub_ = it.advertise(world_debug_image_topic_, 1);
          }
        } else {
          std::lock_guard<std::mutex> lk(world_pose_mutex_);
          have_world_pose_ = false;
        }
      } else if (name == "world_debug_image_topic") {
        world_debug_image_topic_ = p.as_string();
        if (publish_world_debug_image_) {
          auto self = this->shared_from_this();
          image_transport::ImageTransport it(self);
          world_debug_image_pub_ = it.advertise(world_debug_image_topic_, 1);
        }
      } else if (name == "pose_cov_topic") {
        pose_cov_topic_ = p.as_string();
        pose_cov_sub_.reset();
        updatePoseCovSubscription();
      } else if (name == "world_axes_length") {
        world_axes_length_ = p.as_double();
        if (!(world_axes_length_ > 0.0) || !std::isfinite(world_axes_length_)) {
          world_axes_length_ = 0.5;
        }
      } else if (name == "world_axes_thickness") {
        world_axes_thickness_ = p.as_int();
        if (world_axes_thickness_ <= 0) world_axes_thickness_ = 3;
      } else if (name == "pose_cov_is_world_to_camera") {
        pose_cov_is_world_to_camera_ = p.as_bool();
      } else if (name == "world_pose_max_age_sec") {
        world_pose_max_age_sec_ = p.as_double();
      }

      else if (name == "size_override.ids" || name == "size_override.sizes") {
        updateOverridesFromParameters();
      } else if (name == "dictionary_id") {
        dictionary_id_ = p.as_int();
        dictionary_ = cv::aruco::getPredefinedDictionary(
          static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(dictionary_id_));
      }

      // RViz visualization
      else if (name == "publish_visualization") {
        publish_visualization_ = p.as_bool();

        if (publish_visualization_) {
          if (!vis_pub_) {
            auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
            vis_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("visualization", qos);
          }
        } else {
          if (vis_pub_) {
            visualization_msgs::msg::MarkerArray arr;
            visualization_msgs::msg::Marker clear;
            clear.action = visualization_msgs::msg::Marker::DELETEALL;
            arr.markers.push_back(clear);
            vis_pub_->publish(arr);
          }
          vis_pub_.reset();
        }
      }
      else if (name == "parent_frame_id") {
        parent_frame_id_ = p.as_string();
        if (parent_frame_id_.empty()) {
          r.successful = false;
          r.reason = "parent_frame_id must be non-empty";
          return r;
        }
      }
    }

    return r;
  }

  void setDebugSrvCb(const std::shared_ptr<SetBool::Request> req,
                     std::shared_ptr<SetBool::Response> res)
  {
    const bool enable = req->data;

    std::vector<rclcpp::Parameter> ps;
    ps.emplace_back("publish_debug_image", enable);
    ps.emplace_back("publish_world_debug_image", enable);
    ps.emplace_back("publish_visualization", enable);

    const auto results = this->set_parameters(ps);

    bool ok = true;
    std::string reason;
    for (const auto& rr : results) {
      if (!rr.successful) {
        ok = false;
        if (!reason.empty()) reason += "; ";
        reason += rr.reason;
      }
    }

    res->success = ok;
    res->message = ok ? (enable ? "debug enabled" : "debug disabled") : reason;
  }

  void cameraInfoCb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg)
  {
    last_camera_info_ = msg;
    if (!camera_ready_) {
      parseCameraInfo(*msg);
      camera_ready_ = true;
    }
  }

  void mapMarkersCb(const MarkerArrayMsg::ConstSharedPtr& msg)
  {
    map_ids_.clear();
    size_override_map_.clear();
    for (const auto& m : msg->markers) {
      map_ids_.insert(m.id);
      if (m.size > 0.0f) {
        size_override_map_[m.id] = static_cast<double>(m.size);
      }
    }
  }

  void poseCovCb(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr& msg)
  {
    tf2::Transform T;
    tf2::fromMsg(msg->pose.pose, T);

    std::lock_guard<std::mutex> lk(world_pose_mutex_);
    last_pose_stamp_ = msg->header.stamp;
    last_pose_tf_ = T;
    have_world_pose_ = true;
  }

  void imageCb(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
  {
    if (!enabled_) return;

    MarkerArrayMsg out;
    out.header = msg->header;
    out.header.frame_id = parent_frame_id_;

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    } catch (...) {
      return;
    }
    const cv::Mat& image = cv_ptr->image;
    if (image.empty()) {
      markers_pub_->publish(out);
      if (publish_visualization_) {
        publishVis(out);
      }
      return;
    }

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    cv::aruco::detectMarkers(image, dictionary_, corners, ids, detector_params_, rejected);

    const bool want_marker_dbg =
      publish_debug_image_ &&
      debug_image_pub_ &&
      (debug_image_pub_.getNumSubscribers() > 0);

    // Output markers: 2D corners always; pose later or zeros
    std::vector<int> out_index(ids.size(), -1);
    std::vector<double> det_sizes(ids.size(), default_marker_size_);

    out.markers.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      const uint32_t id = static_cast<uint32_t>(ids[i]);

      if (!shouldProcessMarker(id)) {
        continue;
      }

      MarkerMsg m;
      m.id   = id;
      m.size = static_cast<float>(markerSize(id));
      det_sizes[i] = static_cast<double>(m.size);

      for (int k = 0; k < 4; ++k) {
        m.corners[k].x = corners[i][k].x;
        m.corners[k].y = corners[i][k].y;
        m.corners[k].z = 0.0f;
      }

      // pose filled below if estimate_marker_pose_
      out.markers.push_back(m);
      out_index[i] = static_cast<int>(out.markers.size() - 1);
    }

    std::vector<cv::Vec3d> rvecs(ids.size());
    std::vector<cv::Vec3d> tvecs(ids.size());
    std::vector<uint8_t> pnp_have(ids.size(), 0);
    std::vector<uint8_t> pnp_fresh(ids.size(), 0);

    const bool need_pose_outputs = camera_ready_ && estimate_marker_pose_ && !ids.empty();
    const bool need_axes         = camera_ready_ && want_marker_dbg && !ids.empty();

    // Whether to run solvePnP this frame (FPS cap)
    bool do_fresh_pnp = (need_pose_outputs || need_axes);
    if (do_fresh_pnp && pnp_fps_cap_enabled_ && pnp_fps_cap_ > 0) {
      const rclcpp::Time img_stamp = msg->header.stamp;
      if (last_pnp_run_stamp_.nanoseconds() != 0) {
        const double dt = (img_stamp - last_pnp_run_stamp_).seconds();
        const double min_period = 1.0 / static_cast<double>(pnp_fps_cap_);
        if (std::isfinite(dt) && dt >= 0.0 && dt < min_period) {
          do_fresh_pnp = false;
        }
      }
    }

    if ((need_pose_outputs || need_axes) && !ids.empty()) {
      for (size_t i = 0; i < ids.size(); ++i) {
        if (out_index[i] < 0) continue;

        MarkerMsg& m = out.markers[static_cast<size_t>(out_index[i])];
        const uint32_t id = m.id;

        bool used_cache = false;
        auto cache_it = pnp_cache_.find(id);

        if (do_fresh_pnp) {
          const double sz = det_sizes[i];
          const double h = sz * 0.5;

          std::vector<cv::Point3f> obj = {
            {-static_cast<float>(h),  static_cast<float>(h), 0.0f},
            { static_cast<float>(h),  static_cast<float>(h), 0.0f},
            { static_cast<float>(h), -static_cast<float>(h), 0.0f},
            {-static_cast<float>(h), -static_cast<float>(h), 0.0f},
          };

          const std::vector<cv::Point2f>& img = corners[i];

          cv::Vec3d rvec, tvec;
          const bool ok = cv::solvePnP(
            obj, img, camera_matrix_, dist_coeffs_,
            rvec, tvec,
            false, cv::SOLVEPNP_ITERATIVE);

          if (ok) {
            rvecs[i] = rvec;
            tvecs[i] = tvec;
            pnp_have[i]  = 1;
            pnp_fresh[i] = 1;

            CachedPnP cp;
            cp.rvec = rvec;
            cp.tvec = tvec;
            cp.stamp = msg->header.stamp;
            fillPose(rvec, tvec, cp.pose);
            cp.valid = true;
            pnp_cache_[id] = cp;
            cache_it = pnp_cache_.find(id);
          }
        }

        if (!pnp_have[i]) {
          if (cache_it != pnp_cache_.end() && cache_it->second.valid) {
            rvecs[i] = cache_it->second.rvec;
            tvecs[i] = cache_it->second.tvec;
            pnp_have[i] = 1;
            used_cache = true;
          }
        }

        if (need_pose_outputs && pnp_have[i]) {
          if (used_cache) {
            m.pose = cache_it->second.pose;
          } else {
            if (cache_it != pnp_cache_.end() && cache_it->second.valid) {
              m.pose = cache_it->second.pose;
            } else {
              fillPose(rvecs[i], tvecs[i], m.pose);
            }
          }

          if (send_tf_ && tf_broadcaster_ && map_ids_.find(m.id) == map_ids_.end()) {
            geometry_msgs::msg::TransformStamped tfm;
            tfm.header = msg->header;
            tfm.child_frame_id = frame_id_prefix_ + std::to_string(m.id);
            tfm.transform.translation.x = tvecs[i][0];
            tfm.transform.translation.y = tvecs[i][1];
            tfm.transform.translation.z = tvecs[i][2];
            tfm.transform.rotation = m.pose.orientation;
            tf_broadcaster_->sendTransform(tfm);
          }
        }
      }

      if (do_fresh_pnp) {
        last_pnp_run_stamp_ = msg->header.stamp;
      }
    }

    // Debug frame: drawDetectedMarkers; axes from fresh or cached PnP
    cv::Mat marker_debug_image;
    if (want_marker_dbg) {
      image.copyTo(marker_debug_image);

      if (!ids.empty()) {
        cv::aruco::drawDetectedMarkers(marker_debug_image, corners, ids);

        if (camera_ready_) {
          for (size_t i = 0; i < ids.size(); ++i) {
            if (out_index[i] < 0) continue;

            if (pnp_have[i]) {
              const double sz = det_sizes[i];
              cv::drawFrameAxes(marker_debug_image, camera_matrix_, dist_coeffs_,
                                rvecs[i], tvecs[i], sz * 0.5);

              cv::Point2f center(0, 0);
              for (const auto& c : corners[i]) center += c;
              center *= 0.25f;

              const uint32_t id = static_cast<uint32_t>(ids[i]);
              const std::string id_text = "ID: " + std::to_string(id);
              cv::putText(marker_debug_image, id_text,
                          cv::Point(static_cast<int>(center.x + 20), static_cast<int>(center.y - 10)),
                          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
          }
        }
      }

      if (!rejected.empty()) {
        cv::aruco::drawDetectedMarkers(marker_debug_image, rejected, cv::noArray(), cv::Scalar(0, 0, 255));
      }
    }

    // Debug: world axes overlay
    cv::Mat world_debug_image;
    const bool want_world_dbg =
      publish_world_debug_image_ &&
      world_debug_image_pub_ &&
      (world_debug_image_pub_.getNumSubscribers() > 0);

    if (want_world_dbg) {
      image.copyTo(world_debug_image);

      if (!camera_ready_) {
        cv::putText(world_debug_image, "world axes: camera not ready", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
      } else {
        // Pose snapshot under mutex
        tf2::Transform pose_tf;
        rclcpp::Time pose_stamp(0, 0, RCL_ROS_TIME);
        bool have_pose = false;
        {
          std::lock_guard<std::mutex> lk(world_pose_mutex_);
          have_pose = have_world_pose_;
          pose_tf = last_pose_tf_;
          pose_stamp = last_pose_stamp_;
        }

        if (!have_pose) {
          cv::putText(world_debug_image, "world axes: no pose yet", cv::Point(10, 30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        } else {
          // Pose age gating vs image stamp
          bool age_ok = true;
          if (world_pose_max_age_sec_ > 0.0) {
            const rclcpp::Time img_stamp = msg->header.stamp;
            const double age = std::fabs((img_stamp - pose_stamp).seconds());
            if (std::isfinite(age) && age > world_pose_max_age_sec_) {
              age_ok = false;
              char buf[128];
              std::snprintf(buf, sizeof(buf), "world axes: pose too old (%.0f ms)", age * 1000.0);
              cv::putText(world_debug_image, buf, cv::Point(10, 30),
                          cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            }
          }

          if (age_ok) {
            tf2::Transform T_camera_world;
            if (pose_cov_is_world_to_camera_) {
              T_camera_world = pose_tf;
            } else {
              T_camera_world = pose_tf.inverse();
            }

            // Skip if world origin is behind camera
            const tf2::Vector3 t = T_camera_world.getOrigin();
            if (t.z() <= 0.0) {
              cv::putText(world_debug_image, "world axes: behind camera", cv::Point(10, 30),
                          cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            } else {
              // tf2 -> rvec/tvec for OpenCV
              cv::Vec3d tvec(t.x(), t.y(), t.z());

              tf2::Matrix3x3 m(T_camera_world.getRotation());
              cv::Mat R(3, 3, CV_64F);
              R.at<double>(0,0) = m[0][0]; R.at<double>(0,1) = m[0][1]; R.at<double>(0,2) = m[0][2];
              R.at<double>(1,0) = m[1][0]; R.at<double>(1,1) = m[1][1]; R.at<double>(1,2) = m[1][2];
              R.at<double>(2,0) = m[2][0]; R.at<double>(2,1) = m[2][1]; R.at<double>(2,2) = m[2][2];

              cv::Vec3d rvec;
              cv::Rodrigues(R, rvec);

              cv::drawFrameAxes(
                world_debug_image, camera_matrix_, dist_coeffs_,
                rvec, tvec,
                world_axes_length_, world_axes_thickness_);

              cv::putText(world_debug_image, "world axes", cv::Point(10, 30),
                          cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
          }
        }
      }
    }

    markers_pub_->publish(out);

    if (publish_visualization_) {
      publishVis(out);
    }

    // Publish marker debug image
    if (want_marker_dbg && !marker_debug_image.empty()) {
      auto debug_msg = cv_bridge::CvImage(msg->header, "bgr8", marker_debug_image).toImageMsg();
      debug_image_pub_.publish(debug_msg);
    }

    // Publish world-axes debug image
    if (want_world_dbg && !world_debug_image.empty()) {
      auto world_msg = cv_bridge::CvImage(msg->header, "bgr8", world_debug_image).toImageMsg();
      world_debug_image_pub_.publish(world_msg);
    }
  }

  bool shouldProcessMarker(uint32_t id) const
  {
    if (use_map_markers_ && !detect_individual_markers_) {
      return map_ids_.find(id) != map_ids_.end();
    }
    return true;
  }

  double markerSize(uint32_t id) const
  {
    auto itp = size_override_param_.find(id);
    if (itp != size_override_param_.end()) return itp->second;

    if (use_map_markers_) {
      auto itm = size_override_map_.find(id);
      if (itm != size_override_map_.end()) return itm->second;
    }
    return default_marker_size_;
  }

  void parseCameraInfo(const sensor_msgs::msg::CameraInfo& msg)
  {
    for (int i = 0; i < 9; ++i) {
      camera_matrix_.at<double>(i / 3, i % 3) = msg.k[i];
    }
    dist_coeffs_ = cv::Mat::zeros(static_cast<int>(msg.d.size()), 1, CV_64F);
    for (size_t i = 0; i < msg.d.size(); ++i) {
      dist_coeffs_.at<double>(static_cast<int>(i), 0) = msg.d[i];
    }
  }

  static void fillPose(const cv::Vec3d& rvec, const cv::Vec3d& tvec, geometry_msgs::msg::Pose& pose)
  {
    pose.position.x = tvec[0];
    pose.position.y = tvec[1];
    pose.position.z = tvec[2];

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    tf2::Matrix3x3 m(
      R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
      R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
      R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2));
    tf2::Quaternion q;
    m.getRotation(q);
    pose.orientation = tf2::toMsg(q);
  }

  void publishVis(const MarkerArrayMsg& dets)
  {
    if (!publish_visualization_) return;
    if (!vis_pub_ || vis_pub_->get_subscription_count() == 0) return;

    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);

    if (dets.markers.empty()) {
      vis_pub_->publish(arr);
      return;
    }

    int idgen = 0;
    for (const auto& m : dets.markers) {
      const double sz = (m.size > 0.0f) ? static_cast<double>(m.size) : default_marker_size_;

      if (m.pose.orientation.w == 0.0 && m.pose.orientation.x == 0.0 &&
          m.pose.orientation.y == 0.0 && m.pose.orientation.z == 0.0) {
        continue;
      }

      visualization_msgs::msg::Marker cube;
      cube.header = dets.header;
      cube.ns = "aruco";
      cube.id = idgen++;
      cube.type = visualization_msgs::msg::Marker::CUBE;
      cube.action = visualization_msgs::msg::Marker::ADD;
      cube.pose = m.pose;
      cube.scale.x = sz;
      cube.scale.y = sz;
      cube.scale.z = 0.001;
      cube.color.r = 1.0f; cube.color.g = 1.0f; cube.color.b = 1.0f; cube.color.a = 0.9f;
      arr.markers.push_back(cube);

      visualization_msgs::msg::Marker text;
      text.header = dets.header;
      text.ns = "aruco_id";
      text.id = idgen++;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose = m.pose;
      text.pose.position.z += sz * 0.25;
      text.scale.z = sz * 0.6;
      text.color.g = 1.0f; text.color.a = 1.0f;
      text.text = std::to_string(m.id);
      arr.markers.push_back(text);
    }

    vis_pub_->publish(arr);
  }

private:
  bool initialized_{false};
  bool enabled_{true};
  double default_marker_size_{0.16};
  bool send_tf_{true};
  std::string frame_id_prefix_{"aruco_"};
  bool use_map_markers_{true};
  bool detect_individual_markers_{true};

  bool estimate_marker_pose_{true};

  // solvePnP FPS cap and PnP cache
  bool pnp_fps_cap_enabled_{false};
  int pnp_fps_cap_{10};
  rclcpp::Time last_pnp_run_stamp_{0, 0, RCL_ROS_TIME};

  struct CachedPnP
  {
    cv::Vec3d rvec{0.0, 0.0, 0.0};
    cv::Vec3d tvec{0.0, 0.0, 0.0};
    geometry_msgs::msg::Pose pose{};
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    bool valid{false};
  };
  std::unordered_map<uint32_t, CachedPnP> pnp_cache_;

  // Marker debug
  bool publish_debug_image_{true};
  std::string debug_image_topic_{"debug_image"};

  // Runtime debug toggle service
  std::string debug_service_name_{"toggle_debug"};
  rclcpp::Service<SetBool>::SharedPtr debug_srv_;

  // Message header frame_id
  std::string parent_frame_id_{"main_camera_optical"};

  // World axes debug topic
  bool publish_world_debug_image_{false};
  std::string world_debug_image_topic_{"/aruco_det/world_debug_image"};
  std::string pose_cov_topic_{"/aruco_map/pose_cov"};
  double world_axes_length_{0.5};
  int world_axes_thickness_{3};
  bool pose_cov_is_world_to_camera_{false};
  double world_pose_max_age_sec_{0.2};

  // RViz visualization
  bool publish_visualization_{true};

  std::string image_topic_;
  std::string camera_info_topic_;
  std::string map_markers_topic_;
  int dictionary_id_{2};

  cv::Ptr<cv::aruco::Dictionary> dictionary_;
  cv::Ptr<cv::aruco::DetectorParameters> detector_params_;

  sensor_msgs::msg::CameraInfo::ConstSharedPtr last_camera_info_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  bool camera_ready_{false};

  std::unordered_map<uint32_t, double> size_override_param_;
  std::unordered_map<uint32_t, double> size_override_map_;
  std::unordered_set<uint32_t> map_ids_;

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<MarkerArrayMsg>::SharedPtr map_markers_sub_;

  // pose_cov subscription and cache
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_cov_sub_;
  std::mutex world_pose_mutex_;
  bool have_world_pose_{false};
  rclcpp::Time last_pose_stamp_{0, 0, RCL_ROS_TIME};
  tf2::Transform last_pose_tf_{};

  rclcpp::Publisher<MarkerArrayMsg>::SharedPtr markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub_;

  image_transport::Publisher debug_image_pub_;
  image_transport::Publisher world_debug_image_pub_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
};

} // namespace aruco_det_loc

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<aruco_det_loc::ArucoDetectNode>();
  node->initialize();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
