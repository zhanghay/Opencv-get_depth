
#include <iostream>
#include <functional>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <strstream>

#include "mynteyed/camera.h"
#include "mynteyed/utils.h"

#include "util/cam_utils.h"
#include "util/counter.h"
#include "util/cv_painter.h"
#define CVUI_IMPLEMENTATION
#include "util/cvui.h"

#define WINDOW_NAME "CVUI hello world"

#include <fstream>
typedef cv::Vec<float, 3> Vec3f;

namespace 
{

class DepthRegion {
 public:
  explicit DepthRegion(std::uint32_t n)
    : n_(std::move(n)),
      show_(false),
      selected_(false),
      point_(0, 0) {
  }

  ~DepthRegion() = default;

  /**
   * 鼠标事件：默认不选中区域，随鼠标移动而显示。单击后，则会选中区域来显示。你可以再单击已选中区域或双击未选中区域，取消选中。
   */
  void OnMouse(const int& event, const int& x, const int& y, const int& flags) 
  {
    if (event != cv::EVENT_MOUSEMOVE && event != cv::EVENT_LBUTTONDOWN) 
    {
      return;
    }
    show_ = true;

    if (event == cv::EVENT_MOUSEMOVE) {
      if (!selected_) {
        point_.x = x;
        point_.y = y;
      }
    } else if (event == cv::EVENT_LBUTTONDOWN) {
      if (selected_) {
        if (x >= static_cast<int>(point_.x - n_) &&
            x <= static_cast<int>(point_.x + n_) &&
            y >= static_cast<int>(point_.y - n_) &&
            y <= static_cast<int>(point_.y + n_)) {
          selected_ = false;
        }
      } else {
        selected_ = true;
      }
      point_.x = x;
      point_.y = y;
    }
  }

  template <typename T>
  void ShowElems(const cv::Mat &depth,
                 std::function<std::string(const T &elem)> elem2string,
                 int elem_space = 40,
                 std::function<std::string(const cv::Mat &depth, const cv::Point &point,
                                           const std::uint32_t &n)>
                     getinfo = nullptr)
  {

    if (!show_)
      return;

    int space = std::move(elem_space);
    int n = 2 * n_ + 1;
    cv::Mat im(space * n, space * n, CV_8UC3, cv::Scalar(255, 255, 255)); //白色背景图片

    int x, y;
    std::string str;
    int baseline = 0;
    for (int i = -n_; i <= n; ++i)
    {
      x = point_.x + i;
      if (x < 0 || x >= depth.cols)
        continue;
      for (int j = -n_; j <= n; ++j)
      {
        y = point_.y + j;
        if (y < 0 || y >= depth.rows)
          continue;
        str = elem2string(depth.at<T>(y, x)); //视差图的深度值，截取无效后的
        cv::Scalar color(0, 0, 0);
        if (i == 0 && j == 0)
        {
          color = cv::Scalar(255, 0, 0); //视图界面中心数据的颜色
        }
        cv::Size sz = cv::getTextSize(str, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

        cv::putText(im, str,
                    cv::Point((i + n_) * space + (space - sz.width) / 2,
                              (j + n_) * space + (space + sz.height) / 2),
                    cv::FONT_HERSHEY_PLAIN, 1, color, 1);
        std::cout << str[24];
      }
    }

    if (getinfo)
    {
      std::string info = getinfo(depth, point_, n_);
      if (!info.empty())
      {
        cv::Size sz = cv::getTextSize(info,
                                      cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

        cv::putText(im, info,
                    cv::Point(5, 5 + sz.height),
                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 255), 1); //img,text,左下角,字体，，色彩

      } //info 是(x,y)数据
    }
    //--------------------------------
    //----------------------------------
    //--------------------------to get the distance-----------------------------------
    //------------------------------
    //
    //标定数据，下一步应自动调用标定例程
    //
    cv::Mat cameraMatrix1 = (cv::Mat_<double>(3, 3) << 516.44329833984375000, 0.000000000000, 307.72140502929687500,
                             0.000000000, 517.11633300781250000, 243.89788818359375000,
                             0, 0, 1);
    cv::Mat distCoeffs1 = (cv::Mat_<double>(5, 1) << 0.02105712890625000, -0.03374481201171875, -0.00154495239257812, -0.00125122070312500, 0.00000000000000000);
    cv::Mat cameraMatrix2 = (cv::Mat_<double>(3, 3) << 516.39331054687500000, 0.00000000, 318.16775512695312500,
                             0.0000, 517.14025878906250000, 247.56990051269531250,
                             0.0000, 0.0000000000, 1.0000000000);
    cv::Mat distCoeffs2 = (cv::Mat_<double>(5, 1) << 0.01934051513671875, -0.02861785888671875, 0.00047683715820312, -0.00082397460937500, 0.00000000000000000);
    cv::Size imagesize = cv::Size(640, 480);
    cv::Mat R = (cv::Mat_<double>(3, 3) << 0.99997091293334961, -0.00350868701934814, -0.00676858425140381, 0.00344896316528320, 0.99995505809783936, -0.00882411003112793, 0.00679922103881836, 0.00880050659179688, 0.99993813037872314);
    cv::Mat Tr = (cv::Mat_<double>(3, 1) << -120.16201782226562500, 0.00000000000000000, 0.00000000000000000);
    cv::Mat Q;

    //得到Q矩阵
    cv::Mat R1;
    cv::Mat R2;
    cv::Mat P1;
    cv::Mat P2;

    cv::stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imagesize, R, Tr, R1, R2, P1, P2, Q, 0, 0, imagesize, 0, 0);
    cv::Mat depth1;
    depth.convertTo(depth1, CV_8U);                   //depth1是depth的CV-8U的形式
    cv::Point origin = cv::Point(point_.y, point_.x); //目标点
    double dist = 520.7167975394783 * (1 / 0.008322097266035616) / (float)(depth.at<T>(point_.y, point_.x));
    dist /= 2.00;                                  //计算距离；单位：M
    std::cout << "Distance:" << dist << std::endl; //定义法输出
                                                   ///dist is right !!!!!!!!!!!!!!!!
    //---------------\\
    //---------------\\


    //------------
    //------------
    //-------------------------------------to get (x,y)---------//
    //------------
    //------------
    double theWorldx,theWorldy;
    double f_x=Q.at<double>(2,3),
           //b=1.0/Q.at<double>(3,2),
           u_0=Q.at<double>(0,3),
           v_0=Q.at<double>(1,3);

    theWorldx=(point_.x+u_0)*dist/f_x;
    theWorldy=(point_.y+v_0)*dist/f_x;
    std::cout<<"("<<theWorldx<<","<<theWorldy<<")"<<std::endl<<std::endl;
    //-------------------------------------\\
    //-------------------------------------\\
    //-------------------------------------\\


    cv::imshow("region", im); //10/29/18:31
  }
/*
  void DrawLine(const cv::Mat &im)
  {
    if (!show_)
     {
        return;
     }
    std::uint32_t n = (n_ > 1) ? n_ : 1;
    n += 1;
#ifdef WITH_OPENCV2
    cv::rectangle(const_cast<cv::Mat &>(im),
#else
    cv::line(im,
#endif
                  cv::Point(point_.x - n, point_.y - n),
                  cv::Point(point_.x + n, point_.y + n),
                  selected_ ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);
  cv::line(im, start, end, cv::Scalar(0, 255, 255));
  }
*/


  void DrawRect(const cv::Mat &im)
  {
    if (!show_)
      return;
    std::uint32_t n = (n_ > 1) ? n_ : 1;
    n += 1; // outside the region
#ifdef WITH_OPENCV2
    cv::rectangle(const_cast<cv::Mat &>(im),
#else
    cv::rectangle(im,
#endif
                  cv::Point(point_.x - n, point_.y - n),
                  cv::Point(point_.x + n, point_.y + n),
                  selected_ ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);
  }

private:
  std::uint32_t n_;
  bool show_;
  bool selected_;
  cv::Point point_;
};

void OnDepthMouseCallbackShoot(int event, int x, int y, int flags, void *userdata)
{
  DepthRegion *region = reinterpret_cast<DepthRegion *>(userdata);
  region->OnMouse(event, x, y, flags);
}

void OnDepthMouseCallback(int event, int x, int y, int flags, void *userdata)
{
  DepthRegion *region = reinterpret_cast<DepthRegion *>(userdata);
  region->OnMouse(event, x, y, flags);
}

} // namespace

using namespace std;

MYNTEYE_USE_NAMESPACE

int main(int argc, char const *argv[])
{




  cv::Mat frame;
  cvui::init(WINDOW_NAME);

  Camera cam;
  DeviceInfo dev_info;
  if (!util::select(cam, &dev_info))
  {
    return 1;
  }
  util::print_stream_infos(cam, dev_info.index);

  cout << "Open device: " << dev_info.index << ", "
       << dev_info.name << endl
       << endl;

  OpenParams params(dev_info.index);
  {
    // Framerate: 10(default), [0,60], [0,30](STREAM_2560x720)
    params.framerate = 10;

    // Color mode: raw(default), rectified
    // params.color_mode = ColorMode::COLOR_RECTIFIED;

    // Depth mode: colorful(default), gray, raw
    // Note: must set DEPTH_RAW to get raw depth values
    params.depth_mode = DepthMode::DEPTH_RAW;

    // Stream mode: left color only
    // params.stream_mode = StreamMode::STREAM_640x480;  // vga
    params.stream_mode = StreamMode::STREAM_1280x720; // hd
    // Stream mode: left+right color
    // params.stream_mode = StreamMode::STREAM_1280x480;  // vga
    // params.stream_mode = StreamMode::STREAM_2560x720;  // hd

    // Auto-exposure: true(default), false
    // params.state_ae = false;

    // Auto-white balance: true(default), false
    // params.state_awb = false;

    // Infrared intensity: 0(default), [0,10]
    params.ir_intensity = 4;
  }

  cam.Open(params);
  //----------------------
  //----------------------
  //终端打印
  //---------------------
  //---------------------
  cout << endl;
  if (!cam.IsOpened())
  {
    cerr << "Error: Open camera failed" << endl;
    return 1;
  }
  cout << "Open device success" << endl
       << endl;

  cout << "Press ESC/Q on Windows to terminate" << endl;
  //--------------- \\
  //----------------\\
  //----------------\\

  //--------------
  //----------------------------------------------------|
  //窗口                                                 |
  //-----------------------------------------------------|
  cv::namedWindow("color");
  cv::namedWindow("depth");
  cv::namedWindow("region"); //数据窗口
                             // cv::namedWindow("xyzs");
                             //创建窗口
  DepthRegion depth_region(3);
  
  
  auto depth_info = [](
                        const cv::Mat &depth, const cv::Point &point, const std::uint32_t &n) {
    /*
    int row_beg = point.y - n, row_end = point.y + n + 1;
    int col_beg = point.x - n, col_end = point.x + n + 1;
    if (row_beg < 0) row_beg = 0;
    if (row_end >= depth.rows) row_end = depth.rows;
    if (col_beg < 0) col_beg = 0;
    if (col_end >= depth.cols) col_end = depth.cols;
    cout << "[" << point.y << ", " << point.x << "]" << endl
      << depth.rowRange(row_beg, row_end).colRange(col_beg, col_end)
      << endl << endl;
    */
    std::ostringstream os;
    os << "depth pos: [" << point.y << ", " << point.x << "]" //输出形式：（y，x）

    //此处有x.y
       << "±" << n << ", unit: mm";
    return os.str();
  };
  //depth 窗口的信息
  CVPainter painter;
  util::Counter counter;
  //
  //
  //死循环
  //
  for (;;)
  {
    cam.WaitForStream();
    counter.Update();

    auto image_color = cam.GetStreamData(ImageType::IMAGE_LEFT_COLOR);
    if (image_color.img)
    {
      cv::Mat color = image_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
      painter.DrawSize(color, CVPainter::TOP_LEFT);

      painter.DrawStreamData(color, image_color, CVPainter::TOP_RIGHT);
      painter.DrawInformation(color, util::to_string(counter.fps()),
                              CVPainter::BOTTOM_RIGHT);

      cv::setMouseCallback("color", OnDepthMouseCallback, &depth_region);
      depth_region.DrawRect(color);
      cv::imshow("color", color);
    }

    auto image_depth = cam.GetStreamData(ImageType::IMAGE_DEPTH);
    //-------------------------------------
    //当读取到image depth
    //  //  //  //  //  //  //  
    //-------------------------------------
  /*   if (image_depth.img)
        {
           cv::Mat depth = image_depth.img->To(ImageFormat::DEPTH_RAW)->ToMat(); //深度图的Mat

          cv::setMouseCallback("depth", OnDepthMouseCallback, &depth_region);
          // Note: DrawRect will change some depth values to show the rect.
          depth_region.DrawRect(depth);
          cv::imshow("depth", depth);

         //depth.converTo()

          depth_region.ShowElems<ushort>(depth, [](const ushort &elem) {return std::to_string(elem); }, 80, depth_info);
      
         }
*/

    if (image_depth.img)
    {  
      //cv::namedWindow("Shoot");
      cv::Mat frame=depth.clone();
      cv::Mat frameMessage=cv::Mat(200, 500, CV_8UC3);
      frameMessage=cv::Scalar(49,52,49);
      cvui::init("Message");
      cvui::init(WINDOW_NAME);
      cvui::imshow(WINDOW_NAME,frame);
      cvui::imshow("Message",frameMessage);
      //cv::Mat depthShoot = image_depth.img->To(ImageFormat::DEPTH_RAW)->ToMat(); //深度图的Mat
      cvui::printf(frameMessage,10,10,"In frame,mouse is at:%d,%d",cvui::mouse(WINDOW_NAME).x,cvui::mouse(WINDOW_NAME).y);
      if(cvui::mouse(WINDOW_NAME,cvui::LEFT_BUTTON,cvui::IS_DOWN))
      {
        cvui::printf(frameMessage,10,90,"frame:LEFT_BUTTON is Down");
        cvui::printf(frameMessage,10,110,"frame mouse is down at:%d,%d",cvui::mouse(WINDOW_NAME).x,cvui::mouse(WINDOW_NAME).y);
      }
      //cv::setMouseCallback("depth", OnDepthMouseCallbackShoot, &depth_region);
      //depth_region.DrawRect(depthShoot);
      //cv::imshow("Shoot", depthShoot);
    }
       

//          depth_region.ShowElems<ushort>(depthShoot, [](const ushort &elem) {return std::to_string(elem); }, 80, depth_info);
          
    
    char key = static_cast<char>(cv::waitKey(1));
    if (key == 27 || key == 'q' || key == 'Q')
    { // ESC/Q
      break;
    }
  }

  cam.Close();
  cv::destroyAllWindows();
  return 0;
}
