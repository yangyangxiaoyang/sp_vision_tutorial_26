#include <opencv2/opencv.hpp>
#include <fmt/core.h>
#include "tools.hpp"

int main() { 
  std::string image_path = "/home/ros2/homework/images/test.webp";

  cv::Mat src = cv::imread(image_path, cv::IMREAD_COLOR);
  if (src.empty()) { 
    fmt::print("Error: Failed to read image '{}'\n", image_path);
    return 1;
  }

  cv::Mat dst;          
  double scale;         
  int offset_x, offset_y;
  resizeAndCenterImage(src, dst, scale, offset_x, offset_y);

  fmt::print("Part2: OpenCV image processing\n");
  fmt::print("Original Image Size: {}x{}\n", src.cols, src.rows);
  fmt::print("Resize Scale: {:.4f}\n", scale);                  
  fmt::print("Offset (x, y): ({}, {})\n", offset_x, offset_y);  

  cv::namedWindow("Resized Image (640x640)", cv::WINDOW_NORMAL); 
  cv::imshow("Resized Image (640x640)", dst);                    
  fmt::print("\nPress any key to close the window...\n");
  cv::waitKey(0); 

  cv::destroyAllWindows();
  return 0;
}