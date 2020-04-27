/****************************************************************************
 *  Copyright (C) 2019 RoboMaster.
 *
 *  This program is free software: you can redistribute it and/or modify *  it under the terms of the GNU General Public License as published by *  the Free Software Foundation, either version 3 of the License, or *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of 
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <http://www.gnu.org/licenses/>.
 ***************************************************************************/

#ifndef ROBORTS_DETECTION_CVTOOLBOX_H
#define ROBORTS_DETECTION_CVTOOLBOX_H

#include <vector>
#include <thread>
#include <mutex>
//opencv
#include <opencv2/opencv.hpp>
//ros
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

namespace roborts_detection {

/**
 * This class is a toolbox for RoboMaster detection.
 */

enum BufferState {
  IDLE = 0,
  WRITE,
  READ
};

class CVToolbox {
 public:
  /**
   *  @brief The constructor of the CVToolbox.
   */
  explicit CVToolbox(std::string camera_name,
                     unsigned int buffer_size = 3) :
      get_img_info_(false)
  {

    ros::NodeHandle nh(camera_name);
    image_transport::ImageTransport it(nh);

    camera_sub_ = it.subscribeCamera("/camera/image_raw",
                                     5,
                                     boost::bind(&CVToolbox::ImageCallback, this, _1,_2));

    image_buffer_.resize(buffer_size);
    buffer_state_.resize(buffer_size);
    index_ = 0;
    capture_time_ = -1;
    for (int i = 0; i < buffer_state_.size(); ++i) {
      buffer_state_[i] = BufferState::IDLE;
    }
    latest_index_ = -1;

  }

  int GetCameraHeight() {
    if (!get_img_info_) {
      ROS_WARN("Can not get camera height info, because the first frame data wasn't received");
      return -1;
    }
    return 600; //height = 600
  }

  int GetCameraWidth() {
    if (!get_img_info_) {
      ROS_WARN("Can not get camera width info, because the first frame data wasn't received");
      return -1;
    }
    return 800; //width = 800
  }

  int GetCameraMatrix(cv::Mat &k_matrix) {
    if (!get_img_info_) {
      ROS_WARN("Can not get camera matrix info, because the first frame data wasn't received");
      return -1;
    }
    k_matrix = cv::Mat(3, 3, CV_64F, cam_matrix).clone();
    return 0;
  }

  int GetCameraDistortion(cv::Mat &distortion) {
    if (!get_img_info_) {
      ROS_WARN("Can not get camera distortion info, because the first frame data wasn't received");
      return -1;
    }
    distortion = cv::Mat(5, 1, CV_64F, cam_dist).clone();
    return 0;
  }

  int GetWidthOffSet() {
    if (!get_img_info_) {
      ROS_WARN("Can not get camera width offset info, because the first frame data wasn't received");
      return -1;
    }
    return 0; //camera_info_.roi.x_offset;
  }

  int GetHeightOffSet() {
    if (!get_img_info_) {
      ROS_WARN("Can not get camera height offset info, because the first frame data wasn't received");
      return -1;
    }
    return 0; //camera_info_.roi.y_offset;
  }

  void ImageCallback(const sensor_msgs::ImageConstPtr &img_msg) {
    if(!get_img_info_){
      capture_begin_ = std::chrono::high_resolution_clock::now();
      get_img_info_ = true;
    } else {
      capture_time_ = std::chrono::duration<double, std::ratio<1, 1000000>>(std::chrono::high_resolution_clock::now() - capture_begin_).count();
      capture_begin_ = std::chrono::high_resolution_clock::now();
//      ROS_WARN("capture time: %lf", capture_time_);
    }
    capture_begin_ = std::chrono::high_resolution_clock::now();
    auto write_begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < buffer_state_.size(); ++i) {
      if (buffer_state_[i] != BufferState::READ) {
        cv::Mat imagebbuffer = cv_bridge::toCvShare(img_msg, "bgr8")->image.clone();
        cv::resize(imagebbuffer,image_buffer_[i],cv::Size(800,600));
        //image_buffer_[i] = cv_bridge::toCvShare(img_msg, "bgr8")->image.clone();
        buffer_state_[i] = BufferState::WRITE;
        lock_.lock();
        latest_index_ = i;
        //std::cout<<" cvtoolbox latest_index_ : "<<latest_index_<<std::endl;
        lock_.unlock();
      }
    }
    /*
    ROS_WARN("write time: %lf", std::chrono::duration<double, std::ratio<1, 1000000>>
        (std::chrono::high_resolution_clock::now() - write_begin).count());*/
  }
  /**
   * @brief Get next new image.
   * @param src_img Output image
   */
  int NextImage(cv::Mat &src_img) {
    if (latest_index_ < 0) {
      ROS_WARN("Call image when no image received");
      return -1;
    }
    int temp_index = -1;
    lock_.lock();
    if (buffer_state_[latest_index_] == BufferState::WRITE ) {
      buffer_state_[latest_index_] = BufferState::READ;
    } else {
      ROS_INFO("No image is available");
      lock_.unlock();
      return temp_index;
    }
    temp_index = latest_index_;
    lock_.unlock();

    src_img = image_buffer_[temp_index];
    return temp_index;
  }

  void GetCaptureTime(double &capture_time) {
    if (!get_img_info_) {
      ROS_WARN("The first image doesn't receive");
      return;
    }
    if (capture_time_ < 0) {
      ROS_WARN("The second image doesn't receive");
      return;
    }

    capture_time = capture_time_;
  }

  /**
   * @brief Return the image after use
   * @param return_index Index gets from function 'int NextImage(cv::Mat &src_img)'
   */
  void ReadComplete(int return_index) {

    if (return_index < 0 || return_index > (buffer_state_.size() - 1)) {
      ROS_ERROR("Return index error, please check the return_index");
      return;
    }

    buffer_state_[return_index] = BufferState::IDLE;
  }

  /**
   * @brief Highlight the blue or red region of the image.
   * @param image Input image ref
   * @return Single channel image
   */
  cv::Mat DistillationColor(const cv::Mat &src_img, unsigned int color,unsigned int begin_x=0,unsigned int begin_y=0,unsigned int end_x=800,unsigned int end_y=600) {
      cv::Mat result;
      result.create(src_img.size(), CV_8UC1);
      double r, g, b;
      double r0, g0, b0;
      if (color == 0) {
        r0 = 10.0; g0 = 30.0; b0 = 240.0;
      }else{
        r0 = 240.0; g0 = 30.0; b0 = 10.0;
      }

      for (int row = begin_y; row<end_y; row++) {
          for (int col = begin_x; col<end_x; col++) {
              b = double(src_img.at<cv::Vec3b>(row, col)[0]);
              g = double(src_img.at<cv::Vec3b>(row, col)[1]);
              r = double(src_img.at<cv::Vec3b>(row, col)[2]);
              result.at<uchar>(row, col) = cv::saturate_cast<uchar>(255.0*(b*b0 + g*g0 + r*r0) / (sqrt(b0*b0 + g0*g0 + r0*r0)*sqrt(r*r + g*g + b*b)));
          }
      }
      return result;
  }
  /**
   * @brief The wrapper for function cv::findContours
   * @param binary binary image ref
   * @return Contours that found.
   */
  std::vector<std::vector<cv::Point>> FindContours(const cv::Mat &binary_img) {
    std::vector<std::vector<cv::Point>> contours;
    const auto mode = CV_RETR_EXTERNAL;
    const auto method = CV_CHAIN_APPROX_SIMPLE;
    cv::findContours(binary_img, contours, mode, method);
    return contours;
  }
  /**
   * @brief Draw rectangle.
   * @param img The image will be drew on
   * @param rect The target rectangle
   * @param color Rectangle color
   * @param thickness Thickness of the line
   */
  void DrawRotatedRect(const cv::Mat &img, const cv::RotatedRect &rect, const cv::Scalar &color, int thickness) {
    cv::Point2f vertex[4];

    cv::Point2f center = rect.center;
    float angle = rect.angle;
    std::ostringstream ss;
    ss << angle;
    std::string text(ss.str());
    int font_face = cv::FONT_HERSHEY_COMPLEX;
    double font_scale = 0.5;
    cv::putText(img, text, center, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);

    rect.points(vertex);
    for (int i = 0; i < 4; i++)
      cv::line(img, vertex[i], vertex[(i + 1) % 4], color, thickness);
  }

  void DrawRotatedRect(const cv::Mat &img, const cv::RotatedRect &rect, const cv::Scalar &color, int thickness, float angle) {
    cv::Point2f vertex[4];

    cv::Point2f center = rect.center;
    std::ostringstream ss;
    ss << (int)(angle);
    std::string text(ss.str());
    int font_face = cv::FONT_HERSHEY_COMPLEX;
    double font_scale = 0.5;
    cv::putText(img, text, center, font_face, font_scale, cv::Scalar(0, 100, 0), thickness, 8, 0);

    rect.points(vertex);
    for (int i = 0; i < 4; i++)
      cv::line(img, vertex[i], vertex[(i + 1) % 4], color, thickness);
  }
 private:
  float cam_matrix[3][3] = {{1750.0,0.0,356.3}, {0.0,1756.0,375.9}, {0.0,0.0,1.0}}
  float cam_dist[5][1] = {{0.0},{0.0},{0.0},{0.0},{0.0}}
  std::vector<cv::Mat> image_buffer_;
  std::vector<BufferState> buffer_state_;
  int latest_index_;
  std::mutex lock_;
  int index_;
  std::chrono::high_resolution_clock::time_point capture_begin_;
  double capture_time_;

  image_transport::CameraSubscriber camera_sub_;
  bool get_img_info_;
  sensor_msgs::CameraInfo camera_info_;
};
} //namespace roborts_detection

#endif //ROBORTS_DETECTION_CVTOOLBOX_H
