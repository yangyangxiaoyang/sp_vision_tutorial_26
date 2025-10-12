#include <opencv2/opencv.hpp>
/*
 * @brief 图片等比例缩放并居中放置到640x640黑色画布
 * @param src 输入图像（原始图片）
 * @param dst 输出图像（640x640画布，缩放后的图片居中，缺失部分黑色）
 * @param scale 输出参数：实际缩放比例
 * @param offset_x 输出参数：缩放后图片在画布上的x轴偏移量（水平居中）
 * @param offset_y 输出参数：缩放后图片在画布上的y轴偏移量（垂直居中）
 */
void resizeAndCenterImage(
  const cv::Mat& src, 
  cv::Mat& dst, 
  double& scale, 
  int& offset_x, 
  int& offset_y
);
