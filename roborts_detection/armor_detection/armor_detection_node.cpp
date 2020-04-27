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

#include <unistd.h>
#include "armor_detection_node.h"

namespace roborts_detection {

ArmorDetectionNode::ArmorDetectionNode():
    node_state_(roborts_common::IDLE),
    demensions_(3),
    initialized_(false),
    detected_enemy_(false),
    yawclass(0),
    undetected_count_(0), //if undetected, delay some frames to avoid missed detections
    ref_x_(512), //reference x coord(center point x on img)
    as_(nh_, "armor_detection_node_action", boost::bind(&ArmorDetectionNode::ActionCB, this, _1), false) {
  initialized_ = false;
  enemy_nh_ = ros::NodeHandle();
  knowpoint.resize(2);
  /*for (int i = 0; i < knowpoint.size(); ++i) {
      knowpoint[i].x=0;
      knowpoint[i].y=0;
  }*/
  if (Init().IsOK()) {
    initialized_ = true;
    node_state_ = roborts_common::IDLE;
  } else {
    ROS_ERROR("armor_detection_node initalized failed!");
    node_state_ = roborts_common::FAILURE;
  }
  as_.start();
}

ErrorInfo ArmorDetectionNode::Init() {
  enemy_info_pub_ = enemy_nh_.advertise<roborts_msgs::GimbalAngle>("cmd_gimbal_angle", 100);
  ArmorDetectionAlgorithms armor_detection_param;

  std::string file_name = ros::package::getPath("roborts_detection") + "/armor_detection/config/armor_detection.prototxt";
  bool read_state = roborts_common::ReadProtoFromTextFile(file_name, &armor_detection_param);
  if (!read_state) {
    ROS_ERROR("Cannot open %s", file_name.c_str());
    return ErrorInfo(ErrorCode::DETECTION_INIT_ERROR);
  }
  gimbal_control_.Init(armor_detection_param.camera_gimbal_transform().offset_x(),
                       armor_detection_param.camera_gimbal_transform().offset_y(),
                       armor_detection_param.camera_gimbal_transform().offset_z(),
                       armor_detection_param.camera_gimbal_transform().offset_pitch(),
                       armor_detection_param.camera_gimbal_transform().offset_yaw(), 
                       armor_detection_param.projectile_model_info().init_v(),
                       armor_detection_param.projectile_model_info().init_k());

  //create the selected algorithms
  std::string selected_algorithm = armor_detection_param.selected_algorithm();
  // create image receiver
  cv_toolbox_ =std::make_shared<CVToolbox>(armor_detection_param.camera_name());
  // create armor detection algorithm
  armor_detector_ = roborts_common::AlgorithmFactory<ArmorDetectionBase,std::shared_ptr<CVToolbox>>::CreateAlgorithm
      (selected_algorithm, cv_toolbox_);

  undetected_armor_delay_ = armor_detection_param.undetected_armor_delay();
  xgma_=armor_detection_param.xgma();
  k_=armor_detection_param.k();
  still_=false;
  //shoot
  tf_ptr_ = std::make_shared<tf::TransformListener>(ros::Duration(10));
  fricwhl_client_ = enemy_nh_.serviceClient<roborts_msgs::FricWhl>("cmd_fric_wheel");
  shootcmd_client_ = enemy_nh_.serviceClient<roborts_msgs::ShootCmd>("cmd_shoot");
  remain_bullets_=armor_detection_param.remain_bullets();
  projectile_supply_number_=100;
  robot_heat_=0;
  fricwhl_=true;
  if_whirl_=false;
  if_swing_=false;
  SetFricWhl(fricwhl_);
  game_status_=0;

   game_status_sub_=enemy_nh_.subscribe<roborts_msgs::GameStatus>("game_status",100,&ArmorDetectionNode::GameStatusCallback,this);

  tf_sub_=enemy_nh_.subscribe<tf2_msgs::TFMessage>("tf",100,&ArmorDetectionNode::TfCallback,this);
  robot_shoot_sub_=enemy_nh_.subscribe<roborts_msgs::RobotShoot>("robot_shoot",100,&ArmorDetectionNode::RemainCB,this);*/
  projectile_supply_sub_=enemy_nh_.subscribe<roborts_msgs::ProjectileSupply>("projectile_supply",100,&ArmorDetectionNode::ProjectileSupplyCallback,this);
  whirl_bool_sub_=enemy_nh_.subscribe<roborts_msgs::WhirlBool>("whirl",100,&ArmorDetectionNode::WhirlBoolCallback,this);
 
  cmd_vel_sub_=enemy_nh_.subscribe<geometry_msgs::Twist>("cmd_vel",100,&ArmorDetectionNode::CmdVelCallback,this);
  cmd_vel_acc_sub_=enemy_nh_.subscribe<roborts_msgs::TwistAccel>("cmd_vel_acc",100,&ArmorDetectionNode::CmdVelAccCallback,this);
  whirl_bool_sub_=enemy_nh_.subscribe<roborts_msgs::WhirlBool>("whirl",100,&ArmorDetectionNode::WhirlBoolCallback,this);
  swing_bool_sub_=enemy_nh_.subscribe<roborts_msgs::SwingBool>("swing",100,&ArmorDetectionNode::SwingBoolCallback,this);

  twist_acc_.accel.angular.x=0;
  twist_acc_.accel.angular.y=0;
  twist_acc_.accel.angular.z=0;
  twist_acc_.accel.linear.x =0;
  twist_acc_.accel.linear.y =0;
  twist_acc_.accel.linear.z =0;

  twist_acc_.twist.angular.x=0;
  twist_acc_.twist.angular.y=0;
  twist_acc_.twist.angular.z=0;
  twist_acc_.twist.linear.x =0;
  twist_acc_.twist.linear.y =0;
  twist_acc_.twist.linear.z =0;

  if (armor_detector_ == nullptr) {
    ROS_ERROR("Create armor_detector_ pointer failed!");
    return ErrorInfo(ErrorCode::DETECTION_INIT_ERROR);
  } else
    return ErrorInfo(ErrorCode::OK);
}

  void ArmorDetectionNode::GameStatusCallback(const roborts_msgs::GameStatus::ConstPtr & game_status){
    game_status_=game_status->game_status;
  }



void ArmorDetectionNode::TfCallback(const tf2_msgs::TFMessage::ConstPtr & tf){
  geometry_msgs::TransformStamped tfs=tf->transforms.at(0);
  if(tfs.child_frame_id=="gimbal"){
   // roll_cur=tfs.transform.rotation.x;
    //pitch_cur=tfs.transform.rotation.y;
    yaw_cur=tfs.transform.rotation.z;
  }
}


  void ArmorDetectionNode::SwingBoolCallback(const roborts_msgs::SwingBool::ConstPtr& swing_bool){
    if_swing_=swing_bool->is_swing;
  }

void ArmorDetectionNode::WhirlBoolCallback(const roborts_msgs::WhirlBoolConstPtr& whirl_bool){
    if_whirl_=whirl_bool->if_whirl;
}

void ArmorDetectionNode::CmdVelCallback(const geometry_msgs::TwistConstPtr& twist){
  twist_acc_.accel.angular.x=0;
  twist_acc_.accel.angular.y=0;
  twist_acc_.accel.angular.z=0;
  twist_acc_.accel.linear.x =0;
  twist_acc_.accel.linear.y =0;
  twist_acc_.accel.linear.z =0;

  twist_acc_.twist.angular.x=twist->angular.x;
  twist_acc_.twist.angular.y=twist->angular.y;
  twist_acc_.twist.angular.z=twist->angular.z;
  twist_acc_.twist.linear.x=twist->linear.x;
  twist_acc_.twist.linear.y=twist->linear.y;
  twist_acc_.twist.linear.z=twist->linear.z;

}

void ArmorDetectionNode::CmdVelAccCallback(const roborts_msgs::TwistAccelConstPtr& twist_acc){
  twist_acc_.accel.angular.x=twist_acc->accel.angular.x;
  twist_acc_.accel.angular.y=twist_acc->accel.angular.y;
  twist_acc_.accel.angular.z=twist_acc->accel.angular.z;
  twist_acc_.accel.linear.x=twist_acc->accel.linear.x;
  twist_acc_.accel.linear.y=twist_acc->accel.linear.y;
  twist_acc_.accel.linear.z=twist_acc->accel.linear.z;

  twist_acc_.twist.angular.x=twist_acc->twist.angular.x;
  twist_acc_.twist.angular.y=twist_acc->twist.angular.y;
  twist_acc_.twist.angular.z=twist_acc->twist.angular.z;
  twist_acc_.twist.linear.x= twist_acc->twist.linear.x;
  twist_acc_.twist.linear.y= twist_acc->twist.linear.y;
  twist_acc_.twist.linear.z= twist_acc->twist.linear.z;
}

  void ArmorDetectionNode::SupplierStatusCallback(const roborts_msgs::SupplierStatus::ConstPtr & supplier_stat)
  {

      if(supplier_stat->status==2){
          remain_bullets_+=100;
      }
  }

void ArmorDetectionNode::RobotHeatCallback(const roborts_msgs::RobotHeat::ConstPtr &robot_heat){
    robot_heat_=robot_heat->shooter_heat;
  }

void ArmorDetectionNode::ProjectileSupplyCallback(const roborts_msgs::ProjectileSupply::ConstPtr &projectile_supply){

  projectile_supply_number_=projectile_supply->number;
}

void ArmorDetectionNode::RemainCB(const roborts_msgs::RobotShoot::ConstPtr &robot_shoot_msgs){
  remain_bullets_--;
}


void ArmorDetectionNode::ActionCB(const roborts_msgs::ArmorDetectionGoal::ConstPtr &data) {
  roborts_msgs::ArmorDetectionFeedback feedback;
  roborts_msgs::ArmorDetectionResult result;
  bool undetected_msg_published = false;

  if(!initialized_){
    feedback.error_code = error_info_.error_code();
    feedback.error_msg  = error_info_.error_msg();
    as_.publishFeedback(feedback);
    as_.setAborted(result, feedback.error_msg);
    ROS_INFO("Initialization Failed, Failed to execute action!");
    return;
  }

  switch (data->command) {
    case 1:
      StartThread();
      break;
    case 2:
      PauseThread();
      break;
    case 3:
      StopThread();
      break;
    default:
      break;
  }
  ros::Rate rate(25);
  while(ros::ok()) {

    if(as_.isPreemptRequested()) {
      as_.setPreempted();
      return;
    }

    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (undetected_count_ != 0) {
        feedback.detected = true;
        feedback.error_code = error_info_.error_code();
        feedback.error_msg = error_info_.error_msg();

        feedback.enemy_pos.header.frame_id = "camera";
        feedback.enemy_pos.header.stamp    = ros::Time::now();

        feedback.enemy_pos.pose.position.x = x_;
        feedback.enemy_pos.pose.position.y = y_;
        feedback.enemy_pos.pose.position.z = z_;
        feedback.enemy_pos.pose.orientation.w = 1;
        as_.publishFeedback(feedback);
        undetected_msg_published = false;
      } else if(!undetected_msg_published) {
        feedback.detected = false;
        feedback.error_code = error_info_.error_code();
        feedback.error_msg = error_info_.error_msg();

        feedback.enemy_pos.header.frame_id = "camera";
        feedback.enemy_pos.header.stamp    = ros::Time::now();

        feedback.enemy_pos.pose.position.x = 0;
        feedback.enemy_pos.pose.position.y = 0;
        feedback.enemy_pos.pose.position.z = 0;
        feedback.enemy_pos.pose.orientation.w = 1;
        as_.publishFeedback(feedback);
        undetected_msg_published = true;
      }
    }
    rate.sleep();
  }
}

void ArmorDetectionNode::ExecuteLoop() {
  undetected_count_ = undetected_armor_delay_;
  //gimbal_count_  = 10;
  auto angle_begin = std::chrono::high_resolution_clock::now();
  while(running_) {
    usleep(1);
    if (node_state_ == NodeState::RUNNING) {
      cv::Point3f target_3d;
      yawclass=0;
      ErrorInfo error_info = armor_detector_->DetectArmor(detected_enemy_, target_3d, yawclass, ref_x_);
     ROS_INFO("gamen status:%d",game_status_); 
     if(game_status_==4){
          ROS_INFO("remain_bullets_:%d,  yawclass:%d",remain_bullets_,yawclass);
          shoot(shootclass(remain_bullets_, yawclass));
      }
      {
        std::lock_guard<std::mutex> guard(mutex_);
        y_ = (double)(target_3d.x/2000.0);
        z_ = (double)(target_3d.y/2000.0);
        x_ = (double)(target_3d.z/2000.0);
        error_info_ = error_info;
      };
      //gimbal_count_--;
      //gimbal_count_plus_--;
      if(detected_enemy_) {
        float pitch, yaw;
        if(gimbal_count_<0 ){
          knowpoint[0]=knowpoint[1];
          knowpoint[1].x = std::chrono::duration<double, std::ratio<1, 1>>(std::chrono::high_resolution_clock::now() - angle_begin).count();
          knowpoint[1].y= yaw_cur*180/3.1415;
          gimbal_t_=(knowpoint[1].y-knowpoint[0].y)/(knowpoint[1].x-knowpoint[0].x);
          gimbal_count_=10;
        }
        if(!if_whirl_){
          still_=true;
        }else{
          still_=false;
        }
        //std::cout<<"if_whirl_ :"<<if_whirl_<<std::endl;
        //std::cout<<"if_swing_ :"<<if_swing_<<std::endl;
        //std::cout<<"still_ :"<<still_<<std::endl;

        gimbal_control_.Transform(target_3d, pitch, yaw, gimbal_t_ ,still_);
        gimbal_angle_.yaw_mode = true;
        gimbal_angle_.pitch_mode = true;
        //gimbal_angle_.yaw_angle = 0.2*yaw;
        gimbal_angle_.yaw_angle = yaw*(k_*exp(-yaw*yaw/(2*xgma_*xgma_))/(2.5066*xgma_)+0.08);
        gimbal_angle_.pitch_angle = 0.5*pitch;
        if(gimbal_t_>50||gimbal_t_<-50){
          gimbal_angle_.yaw_mode = true;
          gimbal_angle_.pitch_mode = true;
          gimbal_angle_.yaw_angle =0;
          gimbal_angle_.pitch_angle = 0;
          ros::Duration(0.01).sleep();
        }

        std::lock_guard<std::mutex> guard(mutex_);
        undetected_count_ = undetected_armor_delay_;
        PublishMsgs();
      } else if(undetected_count_ != 0) {

        gimbal_angle_.yaw_mode = true;
        gimbal_angle_.pitch_mode = true;
        gimbal_angle_.yaw_angle = 0;
        gimbal_angle_.pitch_angle = 0;

        undetected_count_--;
        PublishMsgs();
        if(ref_x_<512){
          ref_x_++;
        }else{
            ref_x_--;
        }
          
      } else if(undetected_count_ == 0) {

        gimbal_angle_.yaw_mode = false;
        gimbal_angle_.pitch_mode = true;
        gimbal_angle_.yaw_angle = 0;
        gimbal_angle_.pitch_angle = 0;
        PublishMsgs();
        //knowpoint[1].x=0;
        //knowpoint[1].y=0;
        //gimbal_count_=50;
        ref_x_=512;
      } 

    }else if (node_state_ == NodeState::PAUSE) {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_var_.wait(lock);
    }
  }
}


int ArmorDetectionNode::shootclass(const int remain_bullets, const int yawclass_){
if(remain_bullets>100&&yawclass_>0){
  return yawclass_;
}else if(remain_bullets>80&&yawclass_>0){
  return yawclass_;
}else if(remain_bullets>40&&yawclass_>0){
  return yawclass_;
}else if(remain_bullets>20&&yawclass_>0){
  return yawclass_;
}else if(remain_bullets>0&&yawclass_>0){
  return 1;
}else{
  return 0;
}

/*
remain_bullets<10  10<remain_bullets<=20  20<remain_bullets<=30  30<remain_bullets<=40
0<heat<180
0<distance<=3m   3m<distance<=7m    distance>7m 
yawclass_: 0(不发射)  1级对准  2级对准  
*/



}


bool ArmorDetectionNode::shoot(unsigned int shoot_number){

      roborts_msgs::ShootCmd shoot_cmd_srv;
      shoot_cmd_srv.request.mode=roborts_sdk::SHOOT_ONCE;
      shoot_cmd_srv.request.number = static_cast<int8_t>(shoot_number);
      if(shoot_number==0){
	shoot_cmd_srv.request.mode=roborts_sdk::SHOOT_STOP;
      }
      shootcmd_client_.call(shoot_cmd_srv);
      return shoot_cmd_srv.response.received;
    }

bool ArmorDetectionNode::SetFricWhl(const bool &fricwhl) {

      roborts_msgs::FricWhl fricwhl_srv;
      fricwhl_srv.request.open = fricwhl;
      if(fricwhl_client_.call(fricwhl_srv) && fricwhl_srv.response.received){
        fricwhl_ = fricwhl;
        return true;
      } else
      {
        ROS_ERROR("Set shoot control failed!");
        return false;
      }
    }


void ArmorDetectionNode::PublishMsgs() {
  enemy_info_pub_.publish(gimbal_angle_);
}

void ArmorDetectionNode::StartThread() {
  ROS_INFO("Armor detection node started!");
  running_ = true;
  armor_detector_->SetThreadState(true);
  if(node_state_ == NodeState::IDLE) {
    armor_detection_thread_ = std::thread(&ArmorDetectionNode::ExecuteLoop, this);
  }
  node_state_ = NodeState::RUNNING;
  condition_var_.notify_one();
}

void ArmorDetectionNode::PauseThread() {
  ROS_INFO("Armor detection thread paused!");
  node_state_ = NodeState::PAUSE;
}

void ArmorDetectionNode::StopThread() {
  node_state_ = NodeState::IDLE;
  running_ = false;
  armor_detector_->SetThreadState(false);
  if (armor_detection_thread_.joinable()) {
    armor_detection_thread_.join();
  }
}

ArmorDetectionNode::~ArmorDetectionNode() {
  StopThread();
}
} //namespace roborts_detection

void SignalHandler(int signal){
  if(ros::isInitialized() && ros::isStarted() && ros::ok() && !ros::isShuttingDown()){
    ros::shutdown();
  }
}

int main(int argc, char **argv) {
  signal(SIGINT, SignalHandler);
  signal(SIGTERM,SignalHandler);
  ros::init(argc, argv, "armor_detection_node", ros::init_options::NoSigintHandler);
  roborts_detection::ArmorDetectionNode armor_detection;
  ros::AsyncSpinner async_spinner(1);
  async_spinner.start();
  ros::waitForShutdown();
  armor_detection.StopThread();
  return 0;
}

