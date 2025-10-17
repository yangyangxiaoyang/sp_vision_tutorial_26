#ifndef BUFF_SOLVER_HPP
#define BUFF_SOLVER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace auto_buff
{
class Buff_Solver
{
public:
    Buff_Solver();
    void setCameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);
    // 传入1-4号点、5号内环中心、6号点
    void setFeaturePoints(const std::vector<cv::Point2f>& points_1_4, 
                         const cv::Point2f& inner_center_2d,          
                         const cv::Point2f& point6_2d);              
    
    bool solveInnerCircleCenter3D(); 
    bool solveRotationCenter3D();     

    cv::Point3f getInnerCircleCenter3D() const;
    cv::Point3f getRotationCenter3D() const;

private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    bool is_camera_params_set_;

    // 存储1-4号点
    std::vector<cv::Point2f> points_1_4_;      
    cv::Point2f inner_center_2d_;             
    cv::Point2f point6_2d_;                    
    bool is_feature_points_set_;

    cv::Point3f inner_circle_center_cam_;      
    cv::Point3f rotation_center_cam_;          

    // 物理参数
    const double kInnerRadius_ = 0.154;         // 1-4号点所在圆的半径（米）
    const double kInnerToPoint6Dist_ = 0.529;   // 内环中心到6号点的固定距离（米）
    const double kInnerToRDist_ = 0.7;          // 内环中心到R标的固定距离（米）

    // 使用1-4号点进行PnP解算
    bool solvePnPWith4Points(cv::Mat& rvec, cv::Mat& tvec);
};
} // namespace auto_buff

#endif // BUFF_SOLVER_HPP
    