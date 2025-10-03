#include "my_camera.hpp"
#include <unordered_map>

myCamera::myCamera() : handle_(nullptr), is_opened_(false) {
    MV_CC_DEVICE_INFO_LIST device_list_;
    int ret_ = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list_);
    if (ret_ != MV_OK || device_list_.nDeviceNum == 0) {
        return;
    }

    ret_ = MV_CC_CreateHandle(&handle_, device_list_.pDeviceInfo[0]);
    if (ret_ != MV_OK) {
        return;
    }

    ret_ = MV_CC_OpenDevice(handle_);
    if (ret_ != MV_OK) {
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
        return;
    }

    MV_CC_SetEnumValue(handle_, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
    MV_CC_SetEnumValue(handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
    MV_CC_SetEnumValue(handle_, "GainAuto", MV_GAIN_MODE_OFF);
    MV_CC_SetFloatValue(handle_, "ExposureTime", 10000);
    MV_CC_SetFloatValue(handle_, "Gain", 20);
    MV_CC_SetFrameRate(handle_, 60);

    ret_ = MV_CC_StartGrabbing(handle_);
    if (ret_ != MV_OK) {
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
        return;
    }

    is_opened_ = true;
}

myCamera::~myCamera() {
    if (handle_ != nullptr) {
        MV_CC_StopGrabbing(handle_);
        
        MV_CC_CloseDevice(handle_);
        
        MV_CC_DestroyHandle(handle_);
    }
    is_opened_ = false;
}

cv::Mat myCamera::transfer(MV_FRAME_OUT& raw) {
    cv::Mat img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);

    const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map_ = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}
    };

    auto pixel_type_ = raw.stFrameInfo.enPixelType;
    if (type_map_.find(pixel_type_) != type_map_.end()) {
        cv::cvtColor(img, img, type_map_.at(pixel_type_));
    }

    return img;
}

bool myCamera::read(cv::Mat& img) {
    if (!is_opened_ || handle_ == nullptr) {
        return false;
    }

    unsigned int n_msec_ = 100;
    int ret_ = MV_CC_GetImageBuffer(handle_, &raw_frame_, n_msec_);
    if (ret_ != MV_OK) {
        return false;
    }

    img = transfer(raw_frame_);

    ret_ = MV_CC_FreeImageBuffer(handle_, &raw_frame_);
    if (ret_ != MV_OK) {
        return false;
    }

    return true;
}
