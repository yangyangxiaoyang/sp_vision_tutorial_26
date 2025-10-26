#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/exiter.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

// 计算云台控制角度
std::pair<double, double> calculate_gimbal_angles(const Eigen::Vector3d & target_pos) {
  double yaw_angle = std::atan2(target_pos.y(), target_pos.x());
  double horizontal_distance = std::hypot(target_pos.x(), target_pos.y());
  double pitch_angle = -std::atan2(target_pos.z(), horizontal_distance);
  return {yaw_angle, pitch_angle};
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 初始化退出器和绘图器
  tools::Exiter exiter;
  tools::Plotter plotter;

  // 初始化相机和云台
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  // 初始化YOLO检测器和求解器
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);

  // 定义变量
  cv::Mat frame;
  Eigen::Quaterniond quaternion;
  std::chrono::steady_clock::time_point timestamp;

  nlohmann::json plot_data;

  // 主循环
  while (!exiter.exit()) {
    // 读取相机图像
    camera.read(frame, timestamp);
    
    // 获取云台姿态
    quaternion = gimbal.q(timestamp);
    
    // 检测装甲板
    std::list<auto_aim::Armor> detected_armors = yolo.detect(frame);
    
    // 检查是否检测到装甲板且云台处于自瞄模式
    bool has_target = !detected_armors.empty();
    bool is_auto_aim_mode = (gimbal.mode() == io::GimbalMode::AUTO_AIM);
    
    if (has_target && is_auto_aim_mode) {
      // 设置云台到世界坐标系的旋转矩阵
      solver.set_R_gimbal2world(quaternion);
      
      // 选择第一个装甲板作为目标
      auto_aim::Armor & target_armor = detected_armors.front();

      // 求解目标位置
      solver.solve(target_armor);

      // 获取世界坐标系下的目标位置
      Eigen::Vector3d target_position = target_armor.xyz_in_world;
      
      // 记录目标位置
      plot_data["x"] = target_position.x();
      plot_data["y"] = target_position.y();
      plot_data["z"] = target_position.z();

      // 计算云台控制角度
      auto [yaw_control, pitch_control] = calculate_gimbal_angles(target_position);

      // 发送控制命令（控制云台但不开火）
      gimbal.send(1, 0, yaw_control, pitch_control);

      // 记录发送的角度
      plot_data["yaw_sent"] = yaw_control;
      plot_data["pitch_sent"] = pitch_control;
      
      // 绘制数据
      plotter.plot(plot_data);
    }

    // ESC退出
    if (cv::waitKey(30) == 27) {
      break;
    }
  }

  return 0;
}