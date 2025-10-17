#include "buff_solver.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>
#include <iostream>

namespace auto_buff
{
Buff_Solver::Buff_Solver() 
    : is_camera_params_set_(false),
      is_feature_points_set_(false)
{
    // 相机内参
    camera_matrix_ = (cv::Mat_<double>(3, 3) << 
        1286.307063384126 , 0                  , 645.34450819155256, 
        0                 , 1288.1400736562441 , 483.6163720308021 , 
        0                 , 0                  , 1                  );
    dist_coeffs_ = (cv::Mat_<double>(1, 5) <<  -0.47562935060124745, 0.21831745829617311, 0.0004957613589406044, -0.00034617769548693592, 0);
    is_camera_params_set_ = true;
}

void Buff_Solver::setCameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs)
{
    camera_matrix_ = camera_matrix.clone();
    dist_coeffs_ = dist_coeffs.clone();
    is_camera_params_set_ = true;
}

// 传入1-4号点、5号内环中心、6号点
void Buff_Solver::setFeaturePoints(const std::vector<cv::Point2f>& points_1_4,
                                  const cv::Point2f& inner_center_2d,
                                  const cv::Point2f& point6_2d)
{
    // 确保1-4号点数量正确
    if (points_1_4.size() == 4) {
        points_1_4_ = points_1_4;
        inner_center_2d_ = inner_center_2d;
        point6_2d_ = point6_2d;
        is_feature_points_set_ = true;
    } else {
        std::cerr << "[错误] 1-4号点数量不正确（需4个点）" << std::endl;
        is_feature_points_set_ = false;
    }
}

// 使用1-4号点进行PnP解算
bool Buff_Solver::solvePnPWith4Points(cv::Mat& rvec, cv::Mat& tvec)
{
    if (!is_camera_params_set_ || !is_feature_points_set_) {
        return false;
    }

    // 构造3D模型：1-4号点在以5号点为中心的圆上
    std::vector<cv::Point3f> object_points;
    const double angle_step = 2 * M_PI / 4; 
    for (int i = 0; i < 4; ++i) {
        double angle = angle_step * i;
        // 3D坐标：以5号点为原点
        double x = kInnerRadius_ * cos(angle);
        double y = kInnerRadius_ * sin(angle);
        double z = 0.0;
        object_points.emplace_back(x, y, z);
    }

    // 1-4号点的2D坐标
    std::vector<cv::Point2f> image_points = points_1_4_;

    // 调用solvePnP
    return cv::solvePnP(object_points, image_points, 
                       camera_matrix_, dist_coeffs_, rvec, tvec, 
                       false, cv::SOLVEPNP_ITERATIVE);
}

// 解算内环中心3D坐标
bool Buff_Solver::solveInnerCircleCenter3D()
{
    if (!is_feature_points_set_) {
        return false;
    }

    cv::Mat rvec, tvec;
    if (!solvePnPWith4Points(rvec, tvec)) {
        std::cerr << "[Error] PnP解算失败（1-4号点）" << std::endl;
        return false;
    }

    // PnP解算的tvec即为5号内环中心在相机系下的3D坐标
    inner_circle_center_cam_ = cv::Point3f(
        (float)tvec.at<double>(0, 0),
        (float)tvec.at<double>(1, 0),
        (float)tvec.at<double>(2, 0)
    );

    return true;
}

// 解算R标中心3D坐标
bool Buff_Solver::solveRotationCenter3D()
{
    if (!solveInnerCircleCenter3D()) {
        return false;
    }

    // 6号点相对内环中心的方向向量
    cv::Point2f dir_2d = point6_2d_ - inner_center_2d_;
    double dir_len = cv::norm(dir_2d);
    if (dir_len < 1e-6) {  
        return false;
    }
    // 归一化方向向量
    cv::Point2f dir_norm_2d = dir_2d / dir_len;

    // 计算R标中心
    rotation_center_cam_ = cv::Point3f(
        inner_circle_center_cam_.x - dir_norm_2d.x * kInnerToRDist_,
        inner_circle_center_cam_.y - dir_norm_2d.y * kInnerToRDist_,
        inner_circle_center_cam_.z  
    );

    return true;
}

cv::Point3f Buff_Solver::getInnerCircleCenter3D() const { 
    return inner_circle_center_cam_; 
}

cv::Point3f Buff_Solver::getRotationCenter3D() const { 
    return rotation_center_cam_; 
}
} // namespace auto_buff
    