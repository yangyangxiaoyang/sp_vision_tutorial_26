#include "io/camera.hpp"
#include "tasks/buff_detector.hpp"
#include "tasks/buff_solver.hpp"
#include "tools/plotter.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "nlohmann/json.hpp"

// 模式切换：1=视频，0=实际相机
const int USE_VIDEO = 0;  

int main(int argc, char**argv)
{
    // 组件初始化
    auto_buff::Buff_Solver buff_solver;
    tools::Plotter plotter;
    auto_buff::Buff_Detector detector;  
    cv::Mat frame;
    std::chrono::steady_clock::time_point frame_timestamp;  

    cv::VideoCapture cap;       
    io::Camera camera(10.0, 10.0, "");  // 曝光10ms，增益10，自动识别相机

    bool is_source_ready = false;

    // 初始化数据源
    if (USE_VIDEO == 1) {
        cap.open("../assets/test.avi");
        is_source_ready = cap.isOpened();
        if (!is_source_ready) {
            std::cerr << "无法打开文件 ../assets/test.avi" << std::endl;
            return -1;
        }
        std::cout << "视频模式启动成功" << std::endl;
    } else {
        is_source_ready = true;
        std::cout << "相机模式启动成功，曝光：15ms，增益：10" << std::endl;
    }

    std::cout << "按 ESC 退出 | 输出：内环中心(x/y/z) + R标中心(x/y/z)" << std::endl;

    while (is_source_ready) {
        // 读取帧
        if (USE_VIDEO == 1) {
            cap >> frame;
            if (frame.empty()) {
                std::cout << "视频播放结束" << std::endl;
                break;
            }
        } else {
            // 调用read，传入图像和时间戳
            camera.read(frame, frame_timestamp);
            if (frame.empty()) {
                std::cerr << "相机读取失败，可能已断开连接" << std::endl;
                break;
            }
        }

        // 识别扇叶（获取1-4号点、5号中心、6号点）
        auto fanblades = detector.detect(frame);
        cv::Mat display_img = frame.clone();

        if (!fanblades.empty()) {
            auto& fan = fanblades[0];
            // 提取特征点
            if (fan.points.size() >= 6) {
                std::vector<cv::Point2f> points_1_4;
                for (int i = 0; i < 4; ++i) {
                    points_1_4.push_back(fan.points[i]);
                }
                cv::Point2f inner_center = fan.center;  
                cv::Point2f point6 = fan.points[5];    

                // 传入解算器
                buff_solver.setFeaturePoints(points_1_4, inner_center, point6);
                bool success = buff_solver.solveRotationCenter3D();

                if (success) {
                    // 获取解算结果并发布
                    auto inner_3d = buff_solver.getInnerCircleCenter3D();
                    auto r_3d = buff_solver.getRotationCenter3D();

                    nlohmann::json data;
                    data["inner_x"] = inner_3d.x;
                    data["inner_y"] = inner_3d.y;
                    data["inner_z"] = inner_3d.z;
                    data["r_x"] = r_3d.x;
                    data["r_y"] = r_3d.y;
                    data["r_z"] = r_3d.z;
                    plotter.plot(data);

                    // 绘制特征点
                    for (int i = 0; i < 4; ++i) {
                        cv::circle(display_img, points_1_4[i], 4, cv::Scalar(255,0,0), -1);
                        cv::putText(display_img, std::to_string(i+1), 
                                   points_1_4[i] + cv::Point2f(5,-5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
                    }
                    cv::circle(display_img, inner_center, 5, cv::Scalar(0,255,0), -1);
                    cv::putText(display_img, "5(中心)", inner_center + cv::Point2f(5,-5),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
                    cv::circle(display_img, point6, 5, cv::Scalar(0,0,255), -1);
                    cv::putText(display_img, "6", point6 + cv::Point2f(5,-5),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
                }
            }
        } else {
            cv::putText(display_img, "未识别到扇叶", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
        }

        // 显示图像
        cv::resize(display_img, display_img, {}, 0.8, 0.8);  
        cv::imshow("解算结果（相机模式）", display_img);

        // 退出逻辑（按ESC键）
        char key = cv::waitKey(10);  
        if (key == 27) {  
            std::cout << "用户手动退出" << std::endl;
            break;
        }
    }

    // 资源释放
    if (USE_VIDEO == 1) {
        cap.release();  
    }
    cv::destroyAllWindows();
    return 0;
}