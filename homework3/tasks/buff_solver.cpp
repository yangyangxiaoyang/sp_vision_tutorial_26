#include "buff_solver.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>
#include <iostream>

namespace auto_buff
{
// 构造函数：初始化默认相机内参与3D模型点
Buff_Solver::Buff_Solver() 
    : is_camera_params_set_(false),
      is_feature_points_set_(false)
{
    // 初始化默认相机内参
    camera_matrix_ = (cv::Mat_<double>(3, 3) << 
        1200.0, 0.0, 640.0,   
        0.0, 1200.0, 360.0,   
        0.0, 0.0, 1.0);       
    dist_coeffs_ = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0); 
    is_camera_params_set_ = true;

    // 初始化内环3D世界点
    initInnerObjectPoints();
    std::cout << "[Buff_Solver] 初始化完成：3D模型点数量=" << inner_object_points_.size() << std::endl;
}

// 设置相机内参
void Buff_Solver::setCameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs)
{
    if (camera_matrix.size() != cv::Size(3, 3))
    {
        std::cerr << "[Error] 相机内参格式错误（需3x3矩阵）" << std::endl;
        return;
    }
    if (dist_coeffs.size() != cv::Size(5, 1) && dist_coeffs.size() != cv::Size(1, 5))
    {
        std::cerr << "[Error] 畸变系数格式错误（需1x5或5x1矩阵）" << std::endl;
        return;
    }
    camera_matrix.copyTo(camera_matrix_);
    dist_coeffs.copyTo(dist_coeffs_);
    is_camera_params_set_ = true;
    std::cout << "[Buff_Solver] 相机参数更新成功" << std::endl;
}

// 输入Yolo11识别的内环2D特征点
void Buff_Solver::setInnerFeaturePoints(const std::vector<cv::Point2f>& image_points)
{
    if (image_points.size() < 3) 
    {
        std::cerr << "[Error] 特征点数量不足（需≥3，当前=" << image_points.size() << "）" << std::endl;
        is_feature_points_set_ = false;
        return;
    }
    inner_image_points_.assign(image_points.begin(), image_points.end());
    is_feature_points_set_ = true;
}

// 初始化内环3D世界点
void Buff_Solver::initInnerObjectPoints()
{
    inner_object_points_.clear();
    for (int i = 0; i < kInnerVertexNum_; ++i)
    {
        double angle = 2 * M_PI * i / kInnerVertexNum_; 
        double x = kInnerRadius_ * cos(angle);         
        double y = kInnerRadius_ * sin(angle);         
        double z = 0.0;                                 
        inner_object_points_.emplace_back(x, y, z);
    }
}

// 2D圆拟合
bool Buff_Solver::fit2DCircle(const std::vector<cv::Point2f>& points, 
                             cv::Point2f& center, double& radius)
{
    size_t n = points.size();
    if (n < 3) return false;

    // 构造线性方程组
    cv::Mat A(n, 3, CV_64F), B(n, 1, CV_64F);
    for (size_t i = 0; i < n; ++i)
    {
        A.at<double>(i, 0) = points[i].x;
        A.at<double>(i, 1) = points[i].y;
        A.at<double>(i, 2) = 1.0;
        B.at<double>(i, 0) = -(points[i].x*points[i].x + points[i].y*points[i].y);
    }

    // 求解最小二乘
    cv::Mat x;
    cv::solve(A.t() * A, A.t() * B, x, cv::DECOMP_SVD);

    // 计算圆心和半径
    double D = x.at<double>(0, 0);
    double E = x.at<double>(1, 0);
    double F = x.at<double>(2, 0);
    center.x = -D / 2.0;
    center.y = -E / 2.0;
    radius = 0.5 * sqrt(D*D + E*E - 4*F);

    // 验证半径有效性
    if (radius <= 0 || std::isnan(radius))
    {
        std::cerr << "[Error] 圆拟合失败（无效半径=" << radius << "）" << std::endl;
        return false;
    }
    return true;
}

// 解算内环中心
bool Buff_Solver::solveInnerCircleCenter()
{
    if (!is_feature_points_set_)
    {
        std::cerr << "[Error] 未输入特征点，无法解算内环中心" << std::endl;
        return false;
    }

    // 拟合内环2D圆
    double inner_radius;
    if (fit2DCircle(inner_image_points_, inner_circle_center_, inner_radius))
    {
        inner_center_history_.push_back(inner_circle_center_);
        if (inner_center_history_.size() > kMaxHistoryFrames_)
        {
            inner_center_history_.erase(inner_center_history_.begin());
        }
        std::cout << "[Task1] 内环中心解算成功：(" << inner_circle_center_.x 
                  << "," << inner_circle_center_.y << ")，半径=" << inner_radius << std::endl;
        return true;
    }
    std::cerr << "[Error] 内环中心解算失败" << std::endl;
    return false;
}

// 反推旋转中心
bool Buff_Solver::solveRotationCenter()
{
    if (inner_center_history_.size() < kMinHistoryFrames_)
    {
        std::cerr << "[Error] 历史帧数不足（需≥" << kMinHistoryFrames_ 
                  << "，当前=" << inner_center_history_.size() << "）" << std::endl;
        return false;
    }

    // 拟合轨迹圆
    double rotation_radius;
    if (fit2DCircle(inner_center_history_, rotation_center_, rotation_radius))
    {
        std::cout << "[Task2] 旋转中心（R标）反推成功：(" << rotation_center_.x 
                  << "," << rotation_center_.y << ")，轨迹半径=" << rotation_radius << std::endl;
        return true;
    }
    std::cerr << "[Error] 旋转中心反推失败" << std::endl;
    return false;
}

// 获取解算结果
cv::Point2f Buff_Solver::getInnerCircleCenter() const { return inner_circle_center_; }
cv::Point2f Buff_Solver::getRotationCenter() const { return rotation_center_; }

// 清空历史数据
void Buff_Solver::clearInnerCenterHistory() { inner_center_history_.clear(); }

// 解PnP
bool Buff_Solver::solvePnP(cv::Mat& rvec, cv::Mat& tvec)
{
    if (!is_camera_params_set_ || !is_feature_points_set_) return false;
    if (inner_image_points_.size() != inner_object_points_.size()) return false;

    return cv::solvePnP(inner_object_points_, inner_image_points_, 
                       camera_matrix_, dist_coeffs_, rvec, tvec, 
                       false, cv::SOLVEPNP_ITERATIVE);
}
} // namespace auto_buff