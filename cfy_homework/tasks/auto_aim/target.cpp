#include "target.hpp"

#include <cmath>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
// 状态向量索引:
// 0 x, 1 vx, 2 y, 3 vy, 4 z, 5 vz, 6 a, 7 w, 8 alpha, 9 r, 10 l, 11 h

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, Eigen::VectorXd P0_dig, double radius, int armor_num)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // 状态: x vx y vy z vz a w alpha r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0(12);
  x0 << center_x, 0, center_y, 0, center_z, 0, ypr[0], 0.0, 0.0, r, 0.1, 0.0;

  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    // a index = 6
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  // 初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  // 状态转移矩阵 F 12x12
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(12, 12);

  F(0, 1) = dt;
  F(2, 3) = dt;
  F(4, 5) = dt;

  // a' = a + w*dt + 0.5*alpha*dt^2
  // w' = w + alpha*dt
  // alpha' = alpha
  F(6, 7) = dt;
  F(6, 8) = 0.5 * dt * dt;
  F(7, 8) = dt;
  // alpha 8, r 9, l 10, h 11

  // Piecewise White Noise Model for translation
  double v1, v2;

  v1 = 1;     // 加速度方差
  v2 = 0.05;  // 角加速度方差

  auto a = dt * dt * dt * dt / 4.0;
  auto b = dt * dt * dt / 2.0;
  auto c = dt * dt;
  // 预测过程噪声偏差的方差

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(12, 12);

  // translational blocks
  Q(0, 0) = a * v1;
  Q(0, 1) = b * v1;
  Q(1, 0) = b * v1;
  Q(1, 1) = c * v1;

  Q(2, 2) = a * v1;
  Q(2, 3) = b * v1;
  Q(3, 2) = b * v1;
  Q(3, 3) = c * v1;

  Q(4, 4) = a * v1;
  Q(4, 5) = b * v1;
  Q(5, 4) = b * v1;
  Q(5, 5) = c * v1;

  Q(8, 8) = v2;                // uncertainty propagated to angular motion
  Q(6, 6) = v2 * dt * dt / 2;  // uncertainty propagated to angle
  Q(7, 7) = v2 * dt;           // uncertainty propagated to angular velocity

  // Small uncertainties for r,l,h
  Q(9, 9) = 1e-6;
  Q(10, 10) = 1e-6;
  Q(11, 11) = 1e-6;

  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  ekf_.predict(F, Q, f);
}

void Target::update(const Armor & armor)
{
  // 装甲板匹配
  int id = 0;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 取前3个distance最小的装甲板
  for (int i = 0; i < std::min(3, (int)xyza_i_list.size()); i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  update_ypda(armor, id);
}

void Target::update_ypda(const Armor & armor, int id)
{
  // 观测jacobi (现在返回 4 x 12 矩阵)
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);

  // 计算 measurement noise R (4x4)
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

  // 测量过程噪声偏差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};  // 获得观测量

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  // r 9, l 10
  auto r_ok = ekf_.x[9] > 0.05 && ekf_.x[9] < 0.5;
  auto l_ok = ekf_.x[9] + ekf_.x[10] > 0.05 && ekf_.x[9] + ekf_.x[10] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[9], ekf_.x[10]);
  return true;
}

bool Target::convergened()
{
  if (update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  // angle 6, r 9, l 10, h 11
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[9] + x[10] : x[9];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[11] : x[4];

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[9] + x[10] : x[9];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // H_armor_xyza 4 x 12
  Eigen::MatrixXd H_armor_xyza = Eigen::MatrixXd::Zero(4, 12);
  // row yaw (mapped from x)
  H_armor_xyza(0, 0) = 1;       // d(x)/d(x)
  H_armor_xyza(0, 6) = dx_da;   // d(x)/d(a)
  H_armor_xyza(0, 9) = dx_dr;   // d(x)/d(r)
  H_armor_xyza(0, 10) = dx_dl;  // d(x)/d(l)

  // row pitch (mapped from y)
  H_armor_xyza(1, 2) = 1;
  H_armor_xyza(1, 6) = dy_da;
  H_armor_xyza(1, 9) = dy_dr;
  H_armor_xyza(1, 10) = dy_dl;

  // row depth (z)
  H_armor_xyza(2, 4) = 1;
  H_armor_xyza(2, 11) = dz_dh;

  // row angle a
  H_armor_xyza(3, 6) = 1;

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);  // 3x3

  // H_armor_ypda is 4 x 4, mapping [ypd(3); angle] to [x,y,z,a] states
  Eigen::MatrixXd H_armor_ypda = Eigen::MatrixXd::Zero(4, 4);
  H_armor_ypda(0, 0) = H_armor_ypd(0, 0);
  H_armor_ypda(0, 1) = H_armor_ypd(0, 1);
  H_armor_ypda(0, 2) = H_armor_ypd(0, 2);
  H_armor_ypda(1, 0) = H_armor_ypd(1, 0);
  H_armor_ypda(1, 1) = H_armor_ypd(1, 1);
  H_armor_ypda(1, 2) = H_armor_ypd(1, 2);
  H_armor_ypda(2, 0) = H_armor_ypd(2, 0);
  H_armor_ypda(2, 1) = H_armor_ypd(2, 1);
  H_armor_ypda(2, 2) = H_armor_ypd(2, 2);
  H_armor_ypda(3, 3) = 1;

  // (4x4) * (4x12) = 4x12
  Eigen::MatrixXd H = H_armor_ypda * H_armor_xyza;
  return H;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim