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

#ifndef ROBORTS_DETECTION_ARMOR_DETECTION_NODE_H
#define ROBORTS_DETECTION_ARMOR_DETECTION_NODE_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <boost/thread.hpp>

#include <ros/ros.h>
#include "actionlib/server/simple_action_server.h"
#include "roborts_msgs/GimbalAngle.h"
#include "roborts_msgs/GimbalRate.h"
#include "roborts_msgs/ArmorDetectionAction.h"

#include "alg_factory/algorithm_factory.h"
#include "io/io.h"
#include "state/node_state.h"

#include "cv_toolbox.h"

#include "armor_detection_base.h"
#include "proto/armor_detection.pb.h"
#include "armor_detection_algorithms.h"
#include "gimbal_control.h"


#include <actionlib/client/simple_action_client.h>
#include "roborts_msgs/RemainBullets.h"
#include "roborts_msgs/ShootCmd.h"
#include "roborts_msgs/FricWhl.h"
#include "roborts_msgs/ProjectileSupply.h"
#include "roborts_msgs/RobotShoot.h"
#include "roborts_msgs/RobotHeat.h"
#include "roborts_msgs/TwistAccel.h"
#include "roborts_msgs/WhirlBool.h"
#include "roborts_msgs/SwingBool.h"
#include "roborts_msgs/SupplierStatus.h"
#include "roborts_msgs/GameStatus.h"
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include "sdk.h"

namespace roborts_detection {

using roborts_common::NodeState;
using roborts_common::ErrorInfo;

class ArmorDetectionNode {
 public:
  explicit ArmorDetectionNode();
  /**
   * @brief Initializing armor detection algorithm.
   * @return Return the error information.
   */
  ErrorInfo Init();
  /**
   * @brief Actionlib server call back function.
   * @param data Command for control the algorithm thread.
   */
  void ActionCB(const roborts_msgs::ArmorDetectionGoal::ConstPtr &data);
  /**
   * @brief Starting the armor detection thread.
   */
  void StartThread();
  /**
   * @brief Pausing the armor detection thread when received command 2 in action_lib callback function.
   */
  void PauseThread();
  /**
   * @brief Stopping armor detection thread.
   */
  void StopThread();
  /**
   * @brief Executing the armor detection algorithm.
   */
  void ExecuteLoop();
  /**
   * @brief Publishing enemy pose information that been calculated by the armor detection algorithm.
   */
  void PublishMsgs();


  bool SetFricWhl(const bool &fricwhl);

  bool shoot(unsigned int shoot_number);

  int shootclass(const int remain_bullets, const int yawclass_);

  void UpdateRobotGimbal();
    
  void checkShoot(geometry_msgs::Vector3 mark_rpy);
  
  void TfCallback(const tf2_msgs::TFMessage::ConstPtr &tf);

  void RemainCB(const roborts_msgs::RobotShoot::ConstPtr &robot_shoot_msgs);
  
  void ProjectileSupplyCallback(const roborts_msgs::ProjectileSupply::ConstPtr &projectile_supply);

  void RobotHeatCallback(const roborts_msgs::RobotHeat::ConstPtr &robot_heat);
  
  void CmdVelCallback(const geometry_msgs::Twist::ConstPtr& twist);

  void CmdVelAccCallback(const roborts_msgs::TwistAccel::ConstPtr& twist_acc);

  void WhirlBoolCallback(const roborts_msgs::WhirlBool::ConstPtr& whirl_bool);

  void SwingBoolCallback(const roborts_msgs::SwingBool::ConstPtr& swing_bool);

  void SupplierStatusCallback(const roborts_msgs::SupplierStatus::ConstPtr & supplier_stat);

  void GameStatusCallback(const roborts_msgs::GameStatus::ConstPtr & game_status);

  ~ArmorDetectionNode();
 protected:
 private:
  std::shared_ptr<ArmorDetectionBase> armor_detector_;
  std::thread armor_detection_thread_;
  unsigned int max_rotating_fps_;
  unsigned int min_rotating_detected_count_;
  unsigned int undetected_armor_delay_;
  float xgma_;
  float k_;
  //! state and error
  NodeState node_state_;
  ErrorInfo error_info_;
  bool initialized_;
  bool running_;
  std::mutex mutex_;
  std::condition_variable condition_var_;
  unsigned int undetected_count_;
  int gimbal_count_;
  int gimbal_count_plus_;
  //! enemy information
  double x_;
  double y_;
  double z_;
  //double sumpitch;
  //double avgpitch;
  //unsigned int unpitch_count_;
  double roll_cur;
  double pitch_cur;
  double yaw_cur;
  bool detected_enemy_;
  unsigned long demensions_;
  int yawclass;
  float ref_x_;
  bool still_;
  
  int remainbullet;
  //ROS
  ros::NodeHandle nh_;
  ros::NodeHandle enemy_nh_;
  ros::Publisher enemy_info_pub_;
  std::shared_ptr<CVToolbox> cv_toolbox_;
  actionlib::SimpleActionServer<roborts_msgs::ArmorDetectionAction> as_;
  roborts_msgs::GimbalAngle gimbal_angle_;
  std::vector<cv::Point2f>knowpoint;
  float gimbal_t_;

  //! control model
  GimbalContrl gimbal_control_;

//-----------shoot------------

  
  ros::ServiceClient fricwhl_client_;
  ros::ServiceClient shootcmd_client_;
  bool fricwhl_;

  //! ros gimbal tf
 tf::StampedTransform gimbal_tf_;
 geometry_msgs::Vector3 mark_rpy_;
 //! tf
 std::shared_ptr<tf::TransformListener> tf_ptr_;
 ros::Subscriber tf_sub_;

 ros::Subscriber robot_shoot_sub_;
 ros::Subscriber projectile_supply_sub_;
 ros::Subscriber robot_heat_sub_;
 ros::Subscriber cmd_vel_sub_;
 ros::Subscriber cmd_vel_acc_sub_;
 ros::Subscriber whirl_bool_sub_;
 ros::Subscriber swing_bool_sub_;
 ros::Subscriber supplier_status_pub_;
 ros::Subscriber game_status_sub_;
 
 unsigned int remain_bullets_;
 double robot_heat_;
 roborts_msgs::TwistAccel twist_acc_;
 bool if_whirl_;
 bool if_swing_;
 unsigned int projectile_supply_number_;
 unsigned int game_status_;

};
} //namespace roborts_detection

#endif //ROBORTS_DETECTION_ARMOR_DETECTION_NODE_H
