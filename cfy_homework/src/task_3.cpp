#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <memory>
#include <spdlog/spdlog.h>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/exiter.hpp"
#include "tools/trajectory.hpp"

using namespace std::chrono_literals;
using json = nlohmann::json;
using namespace auto_aim;
using namespace tools;
using namespace io;

// 规划结果结构体：包含开火标志
struct Plan {
  bool fire;
};

// 规划函数：基于EKF预测的装甲板未来状态判断开火时机
Plan planning(const Target& target, const Eigen::Quaterniond& gimbal_quat) {
  Plan plan;
  plan.fire = false;

  // 关键参数（可根据实际场景调整）
  const double rotate_angle_thresh = 0.1f;    // 装甲板与云台夹角阈值（弧度）
  const double omega_stable_thresh = 0.05f;   // 角速度稳定性阈值（降低以要求更稳定的omega）
  const int stable_frame_threshold = 5;       // 角速度稳定所需连续帧数
  const double predict_time = 0.05;           // 预测未来时间（考虑系统延迟）

  // 预测装甲板未来状态（补偿系统延迟）
  Target target_future = target;
  target_future.predict(predict_time);

  // 获取预测的旋转状态
  Eigen::VectorXd ekf_x = target_future.ekf_x();
  float fitted_omega = std::fabs(ekf_x[7]);  // 旋转角速度
  auto predicted_armors = target_future.armor_xyza_list();  // 未来装甲板位置和角度

  // 旋转稳定性判断（多帧持续稳定检查）
  static std::deque<float> omega_history;    // 保存历史角速度
  static int stable_count = 0;               // 连续稳定帧计数
  
  omega_history.push_back(fitted_omega);
  if (omega_history.size() > stable_frame_threshold) {
    omega_history.pop_front();
  }
  
  // 检查历史窗口内的角速度是否稳定（变化率小于阈值）
  bool is_omega_stable = false;
  if (omega_history.size() >= stable_frame_threshold) {
    float max_omega = *std::max_element(omega_history.begin(), omega_history.end());
    float min_omega = *std::min_element(omega_history.begin(), omega_history.end());
    float omega_range = max_omega - min_omega;
    
    if (omega_range < omega_stable_thresh) {
      stable_count++;
      if (stable_count >= stable_frame_threshold) {
        is_omega_stable = true;
      }
    } else {
      stable_count = 0;
      is_omega_stable = false;
    }
  }

  // 计算云台朝向（从四元数转换为欧拉角）
  Eigen::Vector3d gimbal_euler = tools::eulers(gimbal_quat, 2, 1, 0, false);
  float gimbal_yaw = gimbal_euler[0];  // 云台yaw角

  // 检查是否有装甲板旋转到与云台对准的位置（基于未来状态）
  if (is_omega_stable) {
    for (const auto& xyza : predicted_armors) {
      // 计算装甲板相对于世界坐标系原点的方位角
      float armor_world_yaw = std::atan2(xyza[1], xyza[0]);
      
      // 计算装甲板与云台的夹角
      float angle_diff = tools::limit_rad(armor_world_yaw - gimbal_yaw);
      
      SPDLOG_DEBUG("装甲板世界yaw: {:.3f}, 云台yaw: {:.3f}, 夹角: {:.3f}", 
                   armor_world_yaw, gimbal_yaw, angle_diff);
      
      if (std::fabs(angle_diff) < rotate_angle_thresh) {
        plan.fire = true;
        SPDLOG_INFO("装甲板与云台对准，夹角: {:.3f} rad", angle_diff);
        break;
      }
    }
  }

  return plan;
}

int main(int argc, char* argv[]) {
  // 初始化日志
  spdlog::set_level(spdlog::level::debug);
  spdlog::default_logger()->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

  // 命令行参数
  const std::string keys = 
    "{help h| |显示帮助信息}"
    "{@config-path| |设备配置文件路径（含相机/云台参数）}";
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }
  std::string config_path = cli.get<std::string>("@config-path");

  // EKF核心参数（采用成功案例的稳定配置）
  const Eigen::VectorXd ekf_P0_dig{{0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.05, 0.5, 0.1, 0.001, 0.001, 0.001}};  // 初始协方差
  const int armor_num = 4;                          // 装甲板数量
  const float gyro_radius = 0.2f;                   // 旋转半径初始值
  const auto wait_time = 2s;                        // 收敛后稳定等待时间
  const auto send_interval = 0.5s;                  // 最小开火间隔

  // 初始化核心模块
  Exiter exiter;
  Plotter plotter;
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);

  // 主循环变量
  cv::Mat frame;
  Eigen::Quaterniond gimbal_quat;
  std::chrono::steady_clock::time_point timestamp;
  json plot_data;
  std::unique_ptr<Target> target_ptr = nullptr;     // EKF目标跟踪器
  bool ekf_initialized = false;                     // EKF初始化标志

  // 收敛与开火控制变量
  std::chrono::steady_clock::time_point first_converged_time;
  bool first_converged = true;                      // 首次收敛标志
  auto last_send_time = std::chrono::steady_clock::now();  // 上次开火时间

  // 获取初始云台姿态，固定云台在当前位置
  Eigen::Quaterniond initial_gimbal_quat = gimbal.q(std::chrono::steady_clock::now());
  Eigen::Vector3d initial_gimbal_euler = tools::eulers(initial_gimbal_quat, 2, 1, 0, false);
  float fixed_yaw = initial_gimbal_euler[0];
  float fixed_pitch = initial_gimbal_euler[1];
  
  gimbal.send(1, 0, fixed_yaw, fixed_pitch);
  SPDLOG_INFO("非自瞄模式：云台固定在 yaw={:.3f}, pitch={:.3f}，基于EKF预测触发开火", fixed_yaw, fixed_pitch);

  // 主控制循环
  while (!exiter.exit()) {
    // 读取相机图像与时间戳
    camera.read(frame, timestamp);
    if (frame.empty()) {
      SPDLOG_WARN("相机未读取到图像，跳过当前帧");
      std::this_thread::sleep_for(10ms);
      continue;
    }

    // 获取云台姿态（仅用于坐标转换）
    gimbal_quat = gimbal.q(timestamp);
    solver.set_R_gimbal2world(gimbal_quat);

    // 装甲板检测
    std::list<auto_aim::Armor> detected_armors = yolo.detect(frame);
    bool has_target = !detected_armors.empty();
    auto_aim::Armor* target_armor_ptr = nullptr;

    if (has_target) {
      // 解算首个装甲板位置
      auto& first_armor = detected_armors.front();
      solver.solve(first_armor);
      target_armor_ptr = &first_armor;
    }

    // 目标跟踪逻辑
    if (target_armor_ptr != nullptr) {
      auto& target_armor = *target_armor_ptr;

      // 初始化EKF（首次检测到目标）
      if (!ekf_initialized) {
        SPDLOG_INFO("首次检测到装甲板，初始化EKF跟踪器");
        target_ptr = std::make_unique<Target>(
          target_armor, timestamp, ekf_P0_dig, 
          gyro_radius, armor_num
        );
        ekf_initialized = true;
        continue;
      }

      // EKF预测与更新（基于时间戳的精准更新）
      target_ptr->predict(timestamp);  // 使用时间戳计算真实时间差
      target_ptr->update(target_armor);

      // 记录EKF核心状态（用于调试和Plotter）
      Eigen::VectorXd ekf_x = target_ptr->ekf_x();
      plot_data["omega"] = -ekf_x[7];       // 旋转角速度
      plot_data["radius"] = ekf_x[9];      // 旋转半径
      plot_data["converged"] = target_ptr->convergened();

      // 检查EKF收敛性
      if (target_ptr->convergened()) {
        // 记录首次收敛时间，进入稳定等待期
        if (first_converged) {
          first_converged_time = timestamp;
          first_converged = false;
          SPDLOG_INFO("EKF已收敛，进入稳定等待期（{}s）", 
                     std::chrono::duration_cast<std::chrono::seconds>(wait_time).count());
        }

        // 稳定等待期结束后，执行开火逻辑
        bool should_fire = false;
        if (timestamp - first_converged_time > wait_time) {
          Plan plan = planning(*target_ptr, gimbal_quat);  // 传入云台姿态进行开火判断

          // 检查开火间隔，避免频繁触发
          if (plan.fire && (timestamp - last_send_time > send_interval)) {
            SPDLOG_INFO("装甲板旋转到位，触发开火");
            should_fire = true;
            last_send_time = timestamp;
            plot_data["fire_triggered"] = true;
          } else {
            plot_data["fire_triggered"] = false;
          }
        } else {
          // 稳定等待期内不开火
          plot_data["fire_triggered"] = false;
          SPDLOG_DEBUG("稳定等待中... 剩余时间: {:.2f}s",
                     std::chrono::duration_cast<std::chrono::duration<double>>(
                       wait_time - (timestamp - first_converged_time)
                     ).count());
        }
        
        // 持续发送云台控制命令以保持固定位置，根据should_fire决定是否开火
        gimbal.send(true, should_fire, fixed_yaw, fixed_pitch);
      } else {
        // EKF未收敛时，也要保持云台固定位置
        gimbal.send(true, false, fixed_yaw, fixed_pitch);
      }

      // EKF发散处理
      if (target_ptr->diverged()) {
        SPDLOG_WARN("EKF跟踪发散，重置跟踪器");
        target_ptr.reset();
        ekf_initialized = false;
        first_converged = true;  // 重置收敛状态
      }

    } else {
      // 无目标时重置状态，但保持云台固定位置
      if (ekf_initialized) {
        SPDLOG_INFO("未检测到装甲板，重置EKF");
        target_ptr.reset();
        ekf_initialized = false;
        first_converged = true;
      }
      plot_data["fire_triggered"] = false;
      plot_data["converged"] = false;
      
      // 即使无目标也要保持云台固定位置
      gimbal.send(true, false, fixed_yaw, fixed_pitch);
    }

    // 数据可视化与图像显示
    plotter.plot(plot_data);
    cv::imshow("Task3: Rotating Armor Catch (EKF Enhanced)", frame);
    if (cv::waitKey(1) == 27) {  // ESC键退出
      break;
    }
  }

  // 资源清理
  cv::destroyAllWindows();
  SPDLOG_INFO("程序正常退出");
  return 0;
}