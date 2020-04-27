/****************************************************************************
 *  Copyright (C) 2019 RoboMaster.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of 
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <http://www.gnu.org/licenses/>.
 ***************************************************************************/
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include "constraint_set.h"
//#include "../roborts_decision/blackboard/blackboard.h"
#include "timer/timer.h"
#include "io/io.h"

namespace roborts_detection {

ConstraintSet::ConstraintSet(std::shared_ptr<CVToolbox> cv_toolbox):
    ArmorDetectionBase(cv_toolbox){
  filter_x_count_ = 0;
  filter_y_count_ = 0;
  filter_z_count_ = 0;
  filter_distance_count_ = 0;
  filter_pitch_count_ = 0;
  filter_yaw_count_ = 0;
  read_index_ = -1;
  detection_time_ = 0;
  thread_running_ = false;

  LoadParam();
  error_info_ = ErrorInfo(roborts_common::OK);
}
  //cv::VideoWriter writer("src.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, cv::Size(640, 480));

void ConstraintSet::LoadParam() {
  //read parameters
  ConstraintSetConfig constraint_set_config_;
  std::string file_name = ros::package::getPath("roborts_detection") + \
      "/armor_detection/constraint_set/config/constraint_set.prototxt";
  bool read_state = roborts_common::ReadProtoFromTextFile(file_name, &constraint_set_config_);
  ROS_ASSERT_MSG(read_state, "Cannot open %s", file_name.c_str());
  enable_debug_ = constraint_set_config_.enable_debug();
  enemy_color_ = constraint_set_config_.enemy_color();
  //using_hsv_ = constraint_set_config_.using_hsv();

  //armor info
  float armor_width = constraint_set_config_.armor_size().width();
  float armor_height = constraint_set_config_.armor_size().height();
  SolveArmorCoordinate(armor_width, armor_height);

  //algorithm threshold parameters
  light_max_aspect_ratio_ = constraint_set_config_.threshold().light_max_aspect_ratio();
  light_min_area_ = constraint_set_config_.threshold().light_min_area();
  light_max_angle_ = constraint_set_config_.threshold().light_max_angle();
  light_max_angle_diff_ = constraint_set_config_.threshold().light_max_angle_diff();
  armor_max_angle_ = constraint_set_config_.threshold().armor_max_angle();
  armor_min_area_ = constraint_set_config_.threshold().armor_min_area();
  armor_max_aspect_ratio_ = constraint_set_config_.threshold().armor_max_aspect_ratio();
  armor_max_pixel_val_ = constraint_set_config_.threshold().armor_max_pixel_val();
  //armor_max_stddev_ = constraint_set_config_.threshold().armor_max_stddev();
  //armor_max_mean_   = constraint_set_config_.threshold().armor_max_mean();

  color_thread_ = constraint_set_config_.threshold().color_thread();
  blue_thread_ = constraint_set_config_.threshold().blue_thread();
  red_thread_ = constraint_set_config_.threshold().red_thread();

  int get_intrinsic_state = -1;
  int get_distortion_state = -1;

  while ((get_intrinsic_state < 0) || (get_distortion_state < 0)) {
    ROS_WARN("Wait for camera driver launch %d", get_intrinsic_state);
    usleep(50000);
    ros::spinOnce();
    get_intrinsic_state = cv_toolbox_->GetCameraMatrix(intrinsic_matrix_);
    get_distortion_state = cv_toolbox_->GetCameraDistortion(distortion_coeffs_);
  }
}



ErrorInfo ConstraintSet::DetectArmor(bool &detected, cv::Point3f &target_3d, int &yawclass, float &ref_x) {
  std::vector<cv::RotatedRect> lights;
  std::vector<ArmorInfo> armors;
  //std::vector<cv::RotatedRect> reflights;
  //std::vector<ArmorInfo> refarmors;
  //bool detectedfirst=true;
  //auto img_begin = std::chrono::high_resolution_clock::now();
  bool sleep_by_diff_flag = true;
  //bool sleep_by_diff_flag1 = true;
  while (true) {
    // Ensure exit this thread while call Ctrl-C
    if (!thread_running_) {
      ErrorInfo error_info(ErrorCode::STOP_DETECTION);
      return error_info;
    }
    read_index_ = cv_toolbox_->NextImage(src_img_);
    if (read_index_ < 0  ) {
      //detectedfirst=false;
      // Reducing lock and unlock when accessing function 'NextImage'
      if (detection_time_ == 0) {
        usleep(20000);
        continue;
      } else {
        double capture_time = 0;
        cv_toolbox_->GetCaptureTime(capture_time);
        if (capture_time == 0) {
          // Make sure the driver is launched and the image callback is called
          usleep(20000);
          continue;
        } else if (capture_time > detection_time_ && sleep_by_diff_flag) {
          // ROS_WARN("time sleep %lf", (capture_time - detection_time_));
          usleep((unsigned int)(capture_time - detection_time_));
          sleep_by_diff_flag = false;
          continue;
        } else {
          //For real time request when image call back called, the function 'NextImage' should be called.
          usleep(500);
          continue;
        }
      }
    } else {
      break;
    }
  }
  cv_toolbox_->ReadComplete(read_index_);
  ROS_INFO("read src complete");
  /*ROS_WARN("time get image: %lf", std::chrono::duration<double, std::ratio<1, 1000>>
      (std::chrono::high_resolution_clock::now() - img_begin).count());*/

  auto detection_begin = std::chrono::high_resolution_clock::now();

  //cv::cvtColor(src_img_, gray_img_, CV_BGR2GRAY);
  if (enable_debug_) {
      show_lights_before_filter_ = src_img_.clone();
      show_lights_after_filter_ = src_img_.clone();
      show_armors_befor_filter_ = src_img_.clone();
      show_armors_after_filter_ = src_img_.clone();
      cv::waitKey(1);
  }
// first step: get enemy color light *****************************************
  DetectLights(src_img_, lights);
  if (enable_debug_){
    cv::imshow("show_lights_before_filter", show_lights_before_filter_);
    cv::waitKey(1);
  }
// second step: fliter to get armor lightbars ***************************************************
  FilterLights(lights);
  if (enable_debug_){
    cv::imshow("lights_after_filter", show_lights_after_filter_);
    cv::waitKey(1);
  }
// fliter to get possible armors according to lightbars *******************************************
  PossibleArmors(lights, armors);
  if (enable_debug_){
    cv::imshow("armors_before_filter", show_armors_befor_filter_);
    cv::waitKey(1);
  }
// get the most suitable armor by NMS
  FilterArmors(armors);
  if (enable_debug_){
    cv::imshow("armors_after_filter", show_armors_after_filter_);
    cv::waitKey(1);
  }
  ArmorInfo src_armor;
  /*ArmorInfo ref_armor;
  std::vector<cv::Point2f> final_armor;*/
  if(!armors.empty()) {
      detected = true;
      if(armors.size()==1){
        src_armor = armors[0];
        ref_x = src_armor.rect.center.x;
        //std::cout<<"armors : 1"<<std::endl;
      }else{
      //src_armor = SlectFinalArmor(armors);
        std::sort(armors.begin(),
            armors.end(),       
            [](const ArmorInfo &p1, const ArmorInfo &p2) 
            { return p1.rect.size.area() <p2.rect.size.area();});// 按照面积大小排序
        if(armors[0].rect.size.area()>1.5*armors[1].rect.size.area()){//若最大的大于1.5倍第二大的，选择第一个
          src_armor = armors[0];
//std::cout<<"armors : area()>1.5"<<std::endl;
        }else{//否则按照标准偏差排序，取最小的
          for (unsigned int i = 0; i < armors.size(); i++) {
            armors[i].stddev = abs(armors[i].rect.center.x-ref_x);
          }
          std::sort(armors.begin(),
            armors.end(),       
            [](const ArmorInfo &p1, const ArmorInfo &p2) 
            { return p1.stddev < p2.stddev;});
          src_armor = armors[0];//筛选出最终的armor
         // std::cout<<"armors : stddev"<<std::endl;
        }
      }
      CalcControlInfo(src_armor.vertex, target_3d);
      //std::cout<<"src_armor.rect.center.x "<<src_armor.rect.center.x<<std::endl;
      if(src_armor.rect.center.x < 512+src_armor.rect.size.width/5&&src_armor.rect.center.x > 512-src_armor.rect.size.width/5){
        yawclass = 4;
      }else if(src_armor.rect.center.x < 512+src_armor.rect.size.width/4&&src_armor.rect.center.x > 512-src_armor.rect.size.width/4){
        yawclass = 3;
      }else if(src_armor.rect.center.x < 512+src_armor.rect.size.width/3&&src_armor.rect.center.x > 512-src_armor.rect.size.width/3){
        yawclass = 2;
      }else if(src_armor.rect.center.x < 512+src_armor.rect.size.width/2&&src_armor.rect.center.x > 512-src_armor.rect.size.width/2){
        yawclass = 1;
      }else{
        yawclass = 0;
      }
    }else{
      detected = false;
      yawclass=0;
  }
  lights.clear();
  armors.clear();
  
  detection_time_ = std::chrono::duration<double, std::ratio<1, 1000000>>
      (std::chrono::high_resolution_clock::now() - detection_begin).count();

  return error_info_;
}

void ConstraintSet::DetectLights(const cv::Mat &src, std::vector<cv::RotatedRect> &lights, unsigned int begin_x_,unsigned int begin_y_, unsigned int end_x_, unsigned int end_y_) {
  //std::cout << "********************************************DetectLights********************************************" << std::endl;
  //cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  //cv::dilate(src, src, element, cv::Point(-1, -1), 1);
  cv::Mat binary_light_img;
  auto light = cv_toolbox_->DistillationColor(src, enemy_color_, begin_x_, begin_y_, end_x_, end_y_);
  float thresh;
  if (enemy_color_ == BLUE)
    thresh = blue_thread_;
  else
    thresh = red_thread_;
  cv::threshold(light, binary_light_img, thresh, 255, CV_THRESH_BINARY);
  /*if(enable_debug_)
    cv::imshow("light", light);
  if (enable_debug_) {
    cv::imshow("binary_light_img", binary_light_img);
  }*/
  auto contours_light = cv_toolbox_->FindContours(binary_light_img);
  lights.reserve(contours_light.size());
  lights_info_.reserve(contours_light.size());
  for (unsigned int i = 0; i < contours_light.size(); ++i) {
        cv::RotatedRect single_light = cv::minAreaRect(contours_light[i]);
        //std::cout << "single_light" << single_light.angle <<  std::endl;
        cv::Point2f vertices_point[4];
        single_light.points(vertices_point);
        LightInfo light_info(vertices_point);
        //std::cout << "vertices_point" << vertices_point[0] << vertices_point[1] << vertices_point[2] << vertices_point[3] << std::endl;
      if (enable_debug_)
          cv_toolbox_->DrawRotatedRect(show_lights_before_filter_, single_light, cv::Scalar(0, 255, 0), 1.2, light_info.angle_);
      single_light.angle = light_info.angle_;
      lights.push_back(single_light);
  }

  auto c = cv::waitKey(1);
  if (c == 'a') {
    cv::waitKey(0);
  }
}


void ConstraintSet::FilterLights(std::vector<cv::RotatedRect> &lights) {
  //std::cout << "********************************************FilterLights********************************************" << std::endl;
  std::vector<cv::RotatedRect> rects;
  rects.reserve(lights.size());

  for (auto &light : lights) {
    //float angle;
    auto light_aspect_ratio =
        std::max(light.size.width, light.size.height) / std::min(light.size.width, light.size.height);
    /*if(light.size.width < light.size.height) {
      light.angle = light.angle+90;
    } else
      light.angle = light.angle+180;*/
    if (light_aspect_ratio < light_max_aspect_ratio_ &&
        light.size.area() >= light_min_area_// &&
        //light.angle>110 &&light.angle<250      
               ) { //angle < light_max_angle_ &&
          rects.push_back(light);
          if (enable_debug_)
             cv_toolbox_->DrawRotatedRect(show_lights_after_filter_, light, cv::Scalar(0, 255, 0), 1.2, light.angle);
    }
  }

  lights = rects;
}

void ConstraintSet::PossibleArmors(const std::vector<cv::RotatedRect> &lights, std::vector<ArmorInfo> &armors) {
  //std::cout << "********************************************PossibleArmors********************************************" << std::endl;
  for (unsigned int i = 0; i < lights.size(); i++) { //任意两个灯条进行筛选匹配
    for (unsigned int j = i + 1; j < lights.size(); j++) {
      cv::RotatedRect light1 = lights[i];
      cv::RotatedRect light2 = lights[j];
      auto edge1 = std::minmax(light1.size.width, light1.size.height);
      auto edge2 = std::minmax(light2.size.width, light2.size.height);
      auto lights_dis = std::sqrt((light1.center.x - light2.center.x) * (light1.center.x - light2.center.x) +
          (light1.center.y - light2.center.y) * (light1.center.y - light2.center.y));
      auto center_angle = std::atan(std::abs(light1.center.y - light2.center.y) / std::abs(light1.center.x - light2.center.x)) * 180 / CV_PI;
      center_angle = center_angle > 90 ? 180 - center_angle : center_angle;
      //std::cout << "center_angle: " << center_angle << std::endl;

      cv::RotatedRect rect;
      rect.angle = static_cast<float>(center_angle);
      rect.center.x = (light1.center.x + light2.center.x) / 2;
      rect.center.y = (light1.center.y + light2.center.y) / 2;
      float armor_width = std::abs(static_cast<float>(lights_dis) - std::max(edge1.first, edge2.first));
      float armor_height = std::max<float>(edge1.second, edge2.second);

      rect.size.width = std::max<float>(armor_width, armor_height);
      rect.size.height = std::min<float>(armor_width, armor_height);

      float light1_angle = light1.angle; //light1.size.width < light1.size.height ? -light1.angle : light1.angle + 90
      float light2_angle = light2.angle; //light2.size.width < light2.size.height ? -light2.angle : light2.angle + 90
      //std::cout << "light1_angle: " << light1_angle << std::endl;
      //std::cout << "light2_angle: " << light2_angle << std::endl;

      /*if (enable_debug_) {
        std::cout << "*******************************" << std::endl;
        std::cout << "light_angle_diff_: " << std::abs(light1_angle - light2_angle) << std::endl;
        std::cout << "radio: " << std::max<float>(edge1.second, edge2.second)/std::min<float>(edge1.second, edge2.second) << std::endl;
        std::cout << "armor_angle_: " << std::abs(center_angle) << std::endl;
        std::cout << "armor_aspect_ratio_: " << rect.size.width / (float) (rect.size.height) << std::endl;
        std::cout << "armor_area_: " << std::abs(rect.size.area()) << std::endl;
        std::cout << "armor_pixel_val_: " << (float)(gray_img_.at<uchar>(static_cast<int>(rect.center.y), static_cast<int>(rect.center.x))) << std::endl;
        std::cout << "pixel_y" << static_cast<int>(rect.center.y) << std::endl;
        std::cout << "pixel_x" << static_cast<int>(rect.center.x) << std::endl;
      }*/
      //
      auto angle_diff = std::abs(light1_angle - light2_angle);
      // Avoid incorrect calculation at 180 and 0.
      if (angle_diff > 175) {
        angle_diff = 180 -angle_diff;
      }
      if (angle_diff < light_max_angle_diff_ &&
          std::max<float>(edge1.second, edge2.second)/std::min<float>(edge1.second, edge2.second) < 1.35 &&
          (rect.size.width) / (rect.size.height) < armor_max_aspect_ratio_ &&
          //std::abs(rect.size.area()) > armor_min_area_ && 
          std::abs(center_angle) < armor_max_angle_ 
          /*gray_img_.at<uchar>(static_cast<int>(rect.center.y), static_cast<int>(rect.center.x))
              < armor_max_pixel_val_*/) { //std::abs(center_angle) < armor_max_angle_ &&
        //std::cout<<"rect.size.width"<<rect.size.width<<std::endl;
        //std::cout<<"rect.size.height"<<rect.size.height<<std::endl;
        if (light1.center.x < light2.center.x) {
          std::vector<cv::Point2f> armor_points;
          CalcArmorInfo(armor_points, light1, light2);
          armors.emplace_back(ArmorInfo(rect, armor_points));
          if (enable_debug_)
            cv_toolbox_->DrawRotatedRect(show_armors_befor_filter_, rect, cv::Scalar(0, 255, 0), 1.2);
          armor_points.clear();
        } else {
          std::vector<cv::Point2f> armor_points;
          CalcArmorInfo(armor_points, light2, light1);
          armors.emplace_back(ArmorInfo(rect, armor_points));
          if (enable_debug_)
            cv_toolbox_->DrawRotatedRect(show_armors_befor_filter_, rect, cv::Scalar(0, 255, 0), 1.2);
          armor_points.clear();
        }
      }
    }
  }
}

void ConstraintSet::FilterArmors(std::vector<ArmorInfo> &armors) {
  //std::cout << "********************************************FilterArmors********************************************" << std::endl;
  // nms
  std::vector<bool> is_armor(armors.size(), true);
  for (int i = 0; i < armors.size() && is_armor[i] == true; i++) {
    for (int j = i + 1; j < armors.size() && is_armor[j]; j++) {
      float dx = armors[i].rect.center.x - armors[j].rect.center.x;//根据两个armor中心的间距筛选
      float dy = armors[i].rect.center.y - armors[j].rect.center.y;
      float dis = std::sqrt(dx * dx + dy * dy);
      if (dis < armors[i].rect.size.width + armors[j].rect.size.width) {
        if (armors[i].rect.angle > armors[j].rect.angle) {
          is_armor[i] = false;
        } else {
          is_armor[j] = false;
        }
      }
    }
  }
  //std::cout << armors.size() << std::endl;
  for (unsigned int i = 0; i < armors.size(); i++) {
    if (!is_armor[i]) {
      armors.erase(armors.begin() + i);
      is_armor.erase(is_armor.begin() + i);
    } else if (enable_debug_) {
      cv_toolbox_->DrawRotatedRect(show_armors_after_filter_, armors[i].rect, cv::Scalar(0, 255, 0), 1.2);
    }
    //std::cout<<"armors[i].rect.center : "<<armors[i].rect.center<<std::endl;
  }
  
}

ArmorInfo ConstraintSet::SlectFinalArmor(std::vector<ArmorInfo> &armors) {
  std::sort(armors.begin(),
            armors.end(),       
            [](const ArmorInfo &p1, const ArmorInfo &p2) { return p1.rect.center.x <p2.rect.center.x ;});
  return armors[0];
}

void ConstraintSet::CalcControlInfo(const std::vector<cv::Point2f> & armor, cv::Point3f &target_3d) {
  cv::Mat rvec;
  cv::Mat tvec;
  cv::solvePnP(armor_points_,
               armor,
               intrinsic_matrix_,
               distortion_coeffs_,
               rvec,
               tvec);
  target_3d = cv::Point3f(tvec);
  /*std::cout<<"armor_points_"<<armor_points_<<std::endl;
  std::cout<<"armor.vertex"<<armor.vertex<<std::endl;
  std::cout<<"intrinsic_matrix"<<intrinsic_matrix_<<std::endl;
  std::cout<<"distortion_coeffs"<<distortion_coeffs_<<std::endl;
  std::cout<<"rvec"<<rvec<<std::endl;
  std::cout<<"tvec"<<tvec<<std::endl;*/
  //std::cout<<"target_3d"<<target_3d<<std::endl;
}

void ConstraintSet::CalcArmorInfo(std::vector<cv::Point2f> &armor_points,
                                 cv::RotatedRect left_light,
                                 cv::RotatedRect right_light) {
  cv::Point2f left_points[4], right_points[4];
  left_light.points(left_points);
  right_light.points(right_points);

  cv::Point2f right_lu, right_ld, lift_ru, lift_rd;
  std::sort(left_points, left_points + 4, [](const cv::Point2f &p1, const cv::Point2f &p2) { return p1.x < p2.x; });
  std::sort(right_points, right_points + 4, [](const cv::Point2f &p1, const cv::Point2f &p2) { return p1.x < p2.x; });
  if (right_points[0].y < right_points[1].y) {
    right_lu = right_points[0];
    right_ld = right_points[1];
  } else {
    right_lu = right_points[1];
    right_ld = right_points[0];
  }

  if (left_points[2].y < left_points[3].y) {
    lift_ru = left_points[2];
    lift_rd = left_points[3];
  } else {
    lift_ru = left_points[3];
    lift_rd = left_points[2];
  }
  armor_points.push_back(lift_ru);
  armor_points.push_back(right_lu);
  armor_points.push_back(right_ld);
  armor_points.push_back(lift_rd);
}

void ConstraintSet::CalcFinalArmor(std::vector<cv::Point2f> &final_armors, std::vector<cv::Point2f> &old_points , std::vector<cv::Point2f> &new_points){
  
  cv::Point2f l_u_t_, r_u_t_, r_d_t_, l_d_t_, c_t_;
  cv::Point2f r_u, r_d, l_u, l_d;
  l_u_t_=new_points[0]-old_points[0];
  r_u_t_=new_points[1]-old_points[1];
  r_d_t_=new_points[2]-old_points[2];
  l_d_t_=new_points[3]-old_points[3];
  c_t_=(l_u_t_+r_u_t_+r_d_t_+l_d_t_)/4.0;
  l_u.x=new_points[0].x+3*c_t_.x;
  r_u.x=new_points[1].x+3*c_t_.x;
  r_d.x=new_points[2].x+3*c_t_.x;
  l_d.x=new_points[3].x+3*c_t_.x;
  l_u.y=new_points[0].y;//+1.2*l_u_t_.y;
  r_u.y=new_points[1].y;//+1.2*r_u_t_.y;
  r_d.y=new_points[2].y;//+1.2*r_d_t_.y;
  l_d.y=new_points[3].y;//+1.2*l_d_t_.y;
  final_armors.push_back(l_u);
  final_armors.push_back(r_u);
  final_armors.push_back(r_d);
  final_armors.push_back(l_d);
}

void ConstraintSet::SolveArmorCoordinate(const float width,
                                         const float height) {
  armor_points_.emplace_back(cv::Point3f(-width/2, height/2,  0.0));
  armor_points_.emplace_back(cv::Point3f(width/2,  height/2,  0.0));
  armor_points_.emplace_back(cv::Point3f(width/2,  -height/2, 0.0));
  armor_points_.emplace_back(cv::Point3f(-width/2, -height/2, 0.0));
}

void ConstraintSet::SignalFilter(double &new_num, double &old_num, unsigned int &filter_count, double max_diff) {
  if(fabs(new_num - old_num) > max_diff && filter_count < 2) {
    filter_count++;
    new_num += max_diff;
  } else {
    filter_count = 0;
    old_num = new_num;
  }
}

void ConstraintSet::SetThreadState(bool thread_state) {
  thread_running_ = thread_state;
}

ConstraintSet::~ConstraintSet() {

}
} //namespace roborts_detection








