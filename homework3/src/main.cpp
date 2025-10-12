#include "io/camera.hpp"               
#include "tasks/buff_detector.hpp"     
#include "tasks/buff_solver.hpp"       
#include "tools/plotter.hpp"           
#include "tools/img_tools.hpp"         
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>                      
#include <nlohmann/json.hpp>          

// 优先用视频测试
#define USE_VIDEO 1

int main(int argc, char** argv)
{
    auto_buff::Buff_Solver buff_solver;     
    tools::Plotter plotter;                 
    cv::Mat frame;
    std::chrono::steady_clock::time_point frame_timestamp;  

    // 初始化Buff_Detector
    auto_buff::Buff_Detector buff_detector;  

    cv::VideoCapture cap;
#if USE_VIDEO
    // 视频输入
    std::string video_path = "../assets/test.avi";
    cap.open(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "[Fatal] 测试视频打开失败！路径：" << video_path << std::endl;
        std::cerr << "请确认assets文件夹下存在test.avi" << std::endl;
        return -1;
    }
#else
    // 相机输入
    double exposure_ms = 10.0;    
    double gain = 20.0;           
    std::string camera_vid_pid = ""; 
    io::Camera camera(exposure_ms, gain, camera_vid_pid);
    // 读一帧判断相机是否正常
    camera.read(frame, frame_timestamp);
    if (frame.empty())
    {
        std::cerr << "[Fatal] 相机初始化失败！请检查相机连接" << std::endl;
        return -1;
    }
#endif

    std::cout << "程序运行中，按 'q' 退出" << std::endl;
    while (true)
    {
        // 读取帧
#if USE_VIDEO
        cap.read(frame); 
#else
        camera.read(frame, frame_timestamp); 
#endif
        if (frame.empty())
        {
            std::cerr << "[Warn] 帧为空（视频结束或相机断开）" << std::endl;
            break;
        }

        // 能量机关识别
        std::vector<auto_buff::FanBlade> fanblades = buff_detector.detect(frame);
        if (fanblades.empty())
        {
            // 未识别到，仅显示图像
            cv::imshow("Buff Detection & Solving", frame);
            if (cv::waitKey(30) == 'q') break;
            continue;
        }

        // 取第一个有效扇叶
        auto_buff::FanBlade target_fan = fanblades[0];
        std::vector<cv::Point2f> inner_points = target_fan.points; 

        // 绘制识别到的特征点
        for (const auto& pt : inner_points)
        {
            cv::circle(frame, pt, 4, cv::Scalar(0, 255, 0), -1);
        }

        // 解算内环中心
        buff_solver.setInnerFeaturePoints(inner_points);
        bool inner_solved = buff_solver.solveInnerCircleCenter();
        cv::Point2f inner_center(-1.0f, -1.0f); 
        if (inner_solved)
        {
            inner_center = buff_solver.getInnerCircleCenter();
            // 绘制内环中心
            cv::circle(frame, inner_center, 6, cv::Scalar(0, 0, 255), -1);
            cv::putText(frame, "Inner Center", inner_center + cv::Point2f(10, 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }

        // 解算旋转中心
        cv::Point2f rotation_center(-1.0f, -1.0f);  
        bool rotation_solved = false;
        if (inner_solved)  
        {
            rotation_solved = buff_solver.solveRotationCenter();
            if (rotation_solved)
            {
                rotation_center = buff_solver.getRotationCenter();
                // 绘制旋转中心
                cv::circle(frame, rotation_center, 6, cv::Scalar(255, 0, 0), -1);
                cv::putText(frame, "R-Center", rotation_center + cv::Point2f(10, 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
        }

        // PlotJuggler发布数据
        if (inner_solved)
        {
            nlohmann::json plot_json;
            plot_json["inner_center_x"] = inner_center.x;    
            plot_json["inner_center_y"] = inner_center.y;    
            if (rotation_solved)
            {
                plot_json["rotation_center_x"] = rotation_center.x;  
                plot_json["rotation_center_y"] = rotation_center.y;  
            }
            plotter.plot(plot_json); 
        }

        cv::imshow("Buff Detection & Solving", frame);
        if (cv::waitKey(30) == 'q') break;
    }

#if USE_VIDEO
    cap.release();
#endif
    cv::destroyAllWindows();
    std::cout << "程序退出" << std::endl;
    return 0;
}