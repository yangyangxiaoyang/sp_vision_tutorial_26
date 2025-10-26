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

using namespace std::chrono_literals;

// 计算云台控制角度
std::pair<double, double> calculate_gimbal_angles(const Eigen::Vector3d & target_pos) {
  double yaw_angle = std::atan2(target_pos.y(), target_pos.x());
  double horizontal_distance = std::hypot(target_pos.x(), target_pos.y());
  double pitch_angle = -std::atan2(target_pos.z(), horizontal_distance);
  return {yaw_angle, pitch_angle};
}

int main(int argc, char * argv[]){
  const std::string keys = 
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   | | yaml配置文件路径 }";
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 初始化核心模块
  tools::Exiter exiter;
  tools::Plotter plotter;
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);

  // 主循环变量
  cv::Mat frame;
  Eigen::Quaterniond quaternion;
  std::chrono::steady_clock::time_point timestamp;
  nlohmann::json plot_data;

  // 主控制循环
  while (!exiter.exit()) {
    camera.read(frame, timestamp);
    quaternion = gimbal.q(timestamp);
    
    // 检测装甲板
    std::list<auto_aim::Armor> detected_armors = yolo.detect(frame);
    
    // 检测到装甲板且云台处于自瞄模式
    bool has_target = !detected_armors.empty();
    bool is_auto_aim_mode = (gimbal.mode() == io::GimbalMode::AUTO_AIM);
    
    if (has_target && is_auto_aim_mode) {
      solver.set_R_gimbal2world(quaternion);
      auto_aim::Armor & target_armor = detected_armors.front();

      // 求解目标位置
      solver.solve(target_armor);
      Eigen::Vector3d target_position = target_armor.xyz_in_world;

      // 计算云台控制角度
      auto [yaw_control, pitch_control] = calculate_gimbal_angles(target_position);
      
      // 发送控制命令
      gimbal.send(1, 0, yaw_control, pitch_control);

      // 记录数据
      plot_data["yaw_control"] = yaw_control;
      plot_data["pitch_control"] = pitch_control;
      plotter.plot(plot_data);
    }

    if (cv::waitKey(30) == 27) {
      break;
    }
  }

  return 0;
}