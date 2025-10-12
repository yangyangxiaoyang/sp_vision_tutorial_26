#include "io/my_camera.hpp"  
#include "tasks/yolo.hpp"
#include "opencv2/opencv.hpp"
#include "tools/img_tools.hpp"

int main()
{
    myCamera camera;
    
    auto_aim::YOLO yolo("./configs/yolo.yaml"); 

    cv::Mat img;
    while (true) {
        if (!camera.read(img)) {
            break;  
        }

        auto armors = yolo.detect(img);

        for (const auto& armor : armors) {
            tools::draw_points(img, armor.points, cv::Scalar(0, 0, 255), 2);
        }

        cv::resize(img, img, cv::Size(640, 480));  
        cv::imshow("Armor Detection", img);
        
        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
