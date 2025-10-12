#ifndef AUTO_BUFF__SOLVER_HPP
#define AUTO_BUFF__SOLVER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace auto_buff
{
class Buff_Solver
{
public:
    // 构造函数：初始化相机内参与3D模型点
    Buff_Solver();

    // 设置相机内参
    void setCameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);

    // 输入识别到的2D特征点
    void setInnerFeaturePoints(const std::vector<cv::Point2f>& image_points);

    // 解算内环中心
    bool solveInnerCircleCenter();

    // 反推旋转中心
    bool solveRotationCenter();

    // 获取解算结果
    cv::Point2f getInnerCircleCenter() const;
    cv::Point2f getRotationCenter() const;

    // 清空历史数据
    void clearInnerCenterHistory();

private:
    // 相机参数
    cv::Mat camera_matrix_;      
    cv::Mat dist_coeffs_;        
    bool is_camera_params_set_;  

    // 特征点数据
    std::vector<cv::Point2f> inner_image_points_; 
    std::vector<cv::Point3f> inner_object_points_; 
    bool is_feature_points_set_;                   

    // 解算结果
    cv::Point2f inner_circle_center_;  
    cv::Point2f rotation_center_;      
    std::vector<cv::Point2f> inner_center_history_; 

    // 能量机关官方尺寸
    const double kInnerRadius_ = 0.14;     
    const int kInnerVertexNum_ = 6;        
    const size_t kMinHistoryFrames_ = 8;    
    const size_t kMaxHistoryFrames_ = 50;   

    // 初始化内环3D世界点
    void initInnerObjectPoints();

    // 2D圆拟合
    bool fit2DCircle(const std::vector<cv::Point2f>& points, 
                     cv::Point2f& center, double& radius);

    // 解PnP
    bool solvePnP(cv::Mat& rvec, cv::Mat& tvec);
};
} // namespace auto_buff

#endif // AUTO_BUFF__SOLVER_HPP