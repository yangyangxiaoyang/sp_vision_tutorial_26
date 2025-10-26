#include <chrono>
#include <cmath>
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

using namespace std::chrono_literals;
using json = nlohmann::json;
using namespace auto_aim;
using namespace tools;
using namespace io;

struct Plan {
  bool fire;
};

// 角度最小差(处理环绕)
static inline double angle_diff(double a, double b) {
  double d = a - b;
  while (d > M_PI) d -= 2 * M_PI;
  while (d < -M_PI) d += 2 * M_PI;
  return d;
}
// 固定云台(fixed_yaw)，预测目标在predict_time后的朝向，若与固定云台夹角小于阈值且角速度稳定则fire
Plan planning_simple(const Target& target, double fixed_gimbal_yaw) {
  Plan plan;
  plan.fire = false;

  const double rotate_angle_thresh = 0.1;   
  const double omega_stable_thresh = 0.3;   
  const double predict_time = 0.02;         
  const double ema_alpha = 0.3;             

  // 预测未来状态
  Target future = target;
  future.predict(predict_time);

  Eigen::VectorXd ekf_x = future.ekf_x();
  float fitted_omega = static_cast<float>(ekf_x[7]);

  // EMA平滑角速度(避免瞬时跳变影响稳定性判定)
  static float omega_ema = NAN;
  if (std::isnan(omega_ema)) omega_ema = fitted_omega;
  omega_ema = static_cast<float>(ema_alpha * fitted_omega + (1.0 - ema_alpha) * omega_ema);

  float omega_diff = std::fabs(fitted_omega - omega_ema);
  bool is_omega_stable = (omega_diff < omega_stable_thresh);

  if (!is_omega_stable) {
    SPDLOG_DEBUG("角速度不稳定: fitted_omega={:.4f}, omega_ema={:.4f}, diff={:.4f}",
                 fitted_omega, omega_ema, omega_diff);
    return plan;
  }

  // 检查所有预测到的装甲板是否有某个在固定云台方向附近
  for (const auto &xyza : future.armor_xyza_list()) {
    double armor_world_yaw = std::atan2(xyza[1], xyza[0]);
    double diff = angle_diff(armor_world_yaw, fixed_gimbal_yaw);
    SPDLOG_DEBUG("pred armor yaw={:.4f}, fixed_gimbal_yaw={:.4f}, diff={:.4f}, omega={:.4f}",
                 armor_world_yaw, fixed_gimbal_yaw, diff, fitted_omega);
    if (std::fabs(diff) < rotate_angle_thresh) {
      plan.fire = true;
      SPDLOG_INFO("装甲板(预测)与固定云台对准: 夹角 {:.4f} rad, 角速度 {:.4f}", diff, fitted_omega);
      break;
    }
  }

  return plan;
}

int main(int argc, char* argv[]) {
  spdlog::set_level(spdlog::level::debug);
  spdlog::default_logger()->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

  const std::string keys = 
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   | | yaml配置文件路径 }";
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // EKF初始协方差
  const Eigen::VectorXd ekf_P0_dig{{1, 10, 1, 10, 1, 10, 5, 50, 10, 0.1, 0.1, 0.1}};
  const auto wait_time = 10s;        
  const auto send_interval = 0.7s;  

  // 去抖参数
  const int init_detect_threshold = 4;       // 连续检测帧数才初始化EKF
  const int miss_threshold = 6;              // 连续丢失帧数才重置EKF
  const int converged_stable_threshold = 5;  // 连续收敛帧数才认为稳定

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
  std::unique_ptr<Target> target_ptr = nullptr;
  bool ekf_initialized = false;

  // 收敛和发送控制
  std::chrono::steady_clock::time_point first_converged_time;
  bool first_converged = true;
  auto last_send_time = std::chrono::steady_clock::now();

  // 去抖计数器
  int detect_count = 0;
  int miss_count = 0;
  int converged_count = 0;

  // 获取初始云台姿态并固定云台在该姿态
  Eigen::Quaterniond initial_gimbal_q = gimbal.q(std::chrono::steady_clock::now());
  Eigen::Vector3d initial_gimbal_euler = tools::eulers(initial_gimbal_q, 2, 1, 0, false);
  float fixed_yaw = static_cast<float>(initial_gimbal_euler[0]);
  float fixed_pitch = static_cast<float>(initial_gimbal_euler[1]);

  gimbal.send(1, 0, fixed_yaw, fixed_pitch);
  SPDLOG_INFO("云台固定在 yaw={:.3f}, pitch={:.3f}", fixed_yaw, fixed_pitch);

 // 主控制循环
  while (!exiter.exit()) {
    camera.read(frame, timestamp);
    if (frame.empty()) {
      SPDLOG_WARN("相机未读取到图像，跳过当前帧");
      std::this_thread::sleep_for(10ms);
      continue;
    }
    gimbal_quat = gimbal.q(timestamp);
    solver.set_R_gimbal2world(gimbal_quat);

    std::list<auto_aim::Armor> detected_armors = yolo.detect(frame);
    bool has_target = !detected_armors.empty();
    auto_aim::Armor* target_armor_ptr = nullptr;

    if (has_target) {
      auto& first_armor = detected_armors.front();
      solver.solve(first_armor);
      target_armor_ptr = &first_armor;
      miss_count = 0;

      // 初始化去抖(只有连续detect_count达到阈值才初始化EKF)
      detect_count++;
      if (!ekf_initialized && detect_count < init_detect_threshold) {
        SPDLOG_DEBUG("目标检测去抖: detect_count={}/{}", detect_count, init_detect_threshold);
        target_armor_ptr = nullptr; 
      }
    } else {
      detect_count = 0;
    }

    if (target_armor_ptr != nullptr) {
      auto& target_armor = *target_armor_ptr;

      if (!ekf_initialized) {
        SPDLOG_INFO("稳定检测到装甲板，初始化EKF跟踪器 (detect_count={})", detect_count);
        target_ptr = std::make_unique<Target>(target_armor, timestamp, ekf_P0_dig, 0.2, 4);
        ekf_initialized = true;
        continue;
      }

      // 按时间戳进行预测与更新
      target_ptr->predict(timestamp);
      target_ptr->update(target_armor);

      Eigen::VectorXd ekf_x = target_ptr->ekf_x();
      plot_data["omega"] = ekf_x[7];
      plot_data["radius"] = ekf_x[9];
      plot_data["converged"] = target_ptr->convergened();

      // 连续收敛去抖
      if (target_ptr->convergened()) {
        converged_count++;
      } else {
        converged_count = 0;
      }

      if (converged_count >= converged_stable_threshold) {
        if (first_converged) {
          first_converged_time = timestamp;
          first_converged = false;
          SPDLOG_INFO("EKF稳定收敛（连续 {} 帧），进入稳定等待期（{}s）",
                      converged_stable_threshold,
                      std::chrono::duration_cast<std::chrono::seconds>(wait_time).count());
        }

        if (timestamp - first_converged_time > wait_time) {
          Plan plan = planning_simple(*target_ptr, static_cast<double>(fixed_yaw));

          if (plan.fire && (timestamp - last_send_time > send_interval)) {
            SPDLOG_INFO("准备发送开火指令: time_since_last_send={:.3f}s, fixed_yaw={:.3f}, fixed_pitch={:.3f}",
                        std::chrono::duration_cast<std::chrono::duration<double>>(timestamp - last_send_time).count(),
                        fixed_yaw, fixed_pitch);

            gimbal.send(1, 1, fixed_yaw, fixed_pitch);
            SPDLOG_INFO("开火指令已下发");
            last_send_time = timestamp;

            plot_data["fire_triggered"] = true;
            plot_data["yaw_sent"] = fixed_yaw;
            plot_data["pitch_sent"] = fixed_pitch;
          } else {
            plot_data["fire_triggered"] = false;
          }
        } else {
          plot_data["fire_triggered"] = false;
          SPDLOG_DEBUG("稳定等待中... 剩余时间: {:.2f}s",
                       std::chrono::duration_cast<std::chrono::duration<double>>(
                         wait_time - (timestamp - first_converged_time)
                       ).count());
        }
      } else {
        SPDLOG_DEBUG("converged 连续计数: {}/{}", converged_count, converged_stable_threshold);
      }

      // 发散处理
      if (target_ptr->diverged()) {
        SPDLOG_WARN("EKF 跟踪发散，重置跟踪器");
        target_ptr.reset();
        ekf_initialized = false;
        first_converged = true;
        detect_count = 0;
        miss_count = 0;
        converged_count = 0;
      }
    } else {
      if (ekf_initialized) {
        miss_count++;
        SPDLOG_DEBUG("未检测到目标，miss_count={}", miss_count);
        if (miss_count >= miss_threshold) {
          SPDLOG_INFO("连续 %d 帧未检测到装甲板，重置EKF", miss_count);
          target_ptr.reset();
          ekf_initialized = false;
          first_converged = true;
          detect_count = 0;
          miss_count = 0;
          converged_count = 0;
        }
      }
      plot_data["fire_triggered"] = false;
      plot_data["converged"] = false;
    }

    plotter.plot(plot_data);
    cv::imshow("Task3: Fixed Gimbal Fire on Predicted Angle", frame);
    if (cv::waitKey(1) == 27) {
      break;
    }
  }

  cv::destroyAllWindows();
  SPDLOG_INFO("程序3正常退出");
  return 0;
}