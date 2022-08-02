#pragma once
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

class thin {
public:
	Mat thinImage(const cv::Mat src, const int maxIterations);
};