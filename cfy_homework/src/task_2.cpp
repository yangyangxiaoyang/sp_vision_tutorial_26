#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/exiter.hpp"

using namespace std::chrono_literals;
using json = nlohmann::json;
using namespace auto_aim;
using namespace tools;
using namespace io;

// 计算云台控制角度
std::pair<double, double> calculate_gimbal_angles(const Eigen::Vector3d& target_pos) {
  double yaw_angle = std::atan2(target_pos.y(), target_pos.x());
  double horizontal_distance = std::hypot(target_pos.x(), target_pos.y());
  double pitch_angle = -std::atan2(target_pos.z(), horizontal_distance);
  return {yaw_angle, pitch_angle};
}

int main(int argc, char* argv[]) {
  // 初始化日志
  spdlog::set_level(spdlog::level::debug);
  spdlog::default_logger()->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

  // 命令行参数
  const std::string keys = 
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   | | yaml配置文件路径 }";
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 击打相关参数：删除最大开火次数，添加发射间隔（1秒）
  const float target_stable_dist = 0.1f;    // 目标位置稳定阈值
  const int stable_frame_threshold = 3;     // 目标稳定所需连续帧数
  const std::chrono::seconds fire_interval = 1s;  // 发射间隔1秒

  // 初始化核心模块
  Exiter exiter;
  Plotter plotter;
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);

  // 主循环变量：添加上次发射时间记录，删除开火计数
  cv::Mat frame;
  Eigen::Quaterniond gimbal_quat;
  std::chrono::steady_clock::time_point timestamp, last_timestamp;
  std::chrono::steady_clock::time_point last_fire_time = std::chrono::steady_clock::time_point::min(); // 初始化为最小时间（确保首次可发射）
  json plot_data;
  int stable_frame_count = 0;                // 目标稳定帧数
  Eigen::Vector3d last_target_pos;           // 上一帧目标位置
  bool is_target_stable = false;             // 目标是否稳定

  // 主控制循环
  while (!exiter.exit()) {
    // 读取相机图像与时间戳
    camera.read(frame, timestamp);
    if (frame.empty()) {
      SPDLOG_WARN("相机未读取到图像，跳过当前帧");
      std::this_thread::sleep_for(10ms);
      continue;
    }
    if (last_timestamp.time_since_epoch().count() == 0) {
      last_timestamp = timestamp;
      continue;
    }
    last_timestamp = timestamp;

    // 获取云台状态
    gimbal_quat = gimbal.q(timestamp);

    // 检测装甲板
    std::list<auto_aim::Armor> detected_armors = yolo.detect(frame);
    bool has_target = !detected_armors.empty();
    bool is_auto_aim_mode = (gimbal.mode() == io::GimbalMode::AUTO_AIM);
    auto_aim::Armor* target_armor_ptr = nullptr;

    // 目标选择：选择第一个检测到的装甲板
    if (has_target) {
      target_armor_ptr = &detected_armors.front(); 
    }

    // 目标有效且云台处于自瞄模式
    if (target_armor_ptr != nullptr && is_auto_aim_mode) {
      auto_aim::Armor& target_armor = *target_armor_ptr;

      // 解算目标世界坐标系位姿
      solver.set_R_gimbal2world(gimbal_quat);
      solver.solve(target_armor);
      Eigen::Vector3d target_pos = target_armor.xyz_in_world;

      // 判断目标是否稳定（逻辑不变）
      if (stable_frame_count == 0) {
        last_target_pos = target_pos;
        stable_frame_count = 1;
      } else {
        float pos_diff = static_cast<float>((target_pos - last_target_pos).norm());
        if (pos_diff < target_stable_dist) {
          stable_frame_count++;
        } else {
          stable_frame_count = 1;
          is_target_stable = false;
        }
        last_target_pos = target_pos;
      }
      if (stable_frame_count >= stable_frame_threshold) {
        is_target_stable = true;
      }

      // 计算云台控制角度
      auto [target_yaw, target_pitch] = calculate_gimbal_angles(target_pos);

      // 开火控制：目标稳定且距离上次发射超过1秒
      bool fire = false;
      if (is_target_stable) {
        // 计算当前时间与上次发射时间的间隔
        auto time_since_last_fire = std::chrono::duration_cast<std::chrono::seconds>(
          timestamp - last_fire_time
        );
        if (time_since_last_fire >= fire_interval) {
          fire = true;
          last_fire_time = timestamp;  // 更新上次发射时间
          SPDLOG_INFO("开火！距离上次发射：{}秒", time_since_last_fire.count());
        }
      }

      // 发送控制命令（始终控制云台，稳定且间隔满足时开火）
      gimbal.send(1, 1, target_yaw, target_pitch);

      // 记录数据：删除开火计数相关字段
      plot_data["target_x"] = target_pos.x();
      plot_data["target_z"] = target_pos.z();
      plot_data["yaw_control"] = target_yaw; 
      plot_data["pitch_control"] = target_pitch;
      plotter.plot(plot_data);

    } else {
      // 无目标或非自瞄模式：停止控制，重置稳定状态
      gimbal.send(0, 0, 0.0, 0.0);
      stable_frame_count = 0;
      is_target_stable = false;
      // 重置上次发射时间（避免重新检测到目标时立即发射）
      last_fire_time = std::chrono::steady_clock::time_point::min();
    }

    // 图像显示与退出
    cv::imshow("Task2: Armor Shooting", frame);
    if (cv::waitKey(30) == 27) {
      break;
    }
  }

  cv::destroyAllWindows();
  SPDLOG_INFO("任务2程序正常退出");
  return 0;
}