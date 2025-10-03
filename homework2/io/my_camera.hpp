#ifndef MY_CAMERA_HPP
#define MY_CAMERA_HPP

#include "hikrobot/include/MvCameraControl.h"
#include <opencv2/opencv.hpp>

class myCamera {
public:
    myCamera();
    
    ~myCamera();
    
    bool read(cv::Mat& img);

private:
    void* handle_;                
    bool is_opened_;              
    MV_FRAME_OUT raw_frame_;      

    cv::Mat transfer(MV_FRAME_OUT& raw);
};

#endif // MY_CAMERA_HPP
