#include "tools.hpp"

void resizeAndCenterImage(
  const cv::Mat& src, 
  cv::Mat& dst, 
  double& scale, 
  int& offset_x, 
  int& offset_y
) {
  const int target_w = 640;
  const int target_h = 640;

  double scale_w = static_cast<double>(target_w) / src.cols; 
  double scale_h = static_cast<double>(target_h) / src.rows;  
  scale = std::min(scale_w, scale_h);  

  int resized_w = static_cast<int>(src.cols * scale);
  int resized_h = static_cast<int>(src.rows * scale);

  offset_x = (target_w - resized_w) / 2;  
  offset_y = (target_h - resized_h) / 2;  

  cv::Mat resized;
  cv::resize(src, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_AREA);

  dst = cv::Mat::zeros(target_h, target_w, CV_8UC3);

  resized.copyTo(dst(cv::Rect(offset_x, offset_y, resized_w, resized_h)));
}