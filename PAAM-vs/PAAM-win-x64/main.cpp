//#include <cv.h>
//#include <highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

extern void videoLoop(void);
extern void processVideo(cv::String filename);

cv::String PRIMARYWINDOW = "PAAM-x64";

// Just my personal file path, customize however you'd like
int main(void) {
	processVideo("C:/Berkan/misc/tennisplay.mp4");
	destroyAllWindows();
	return (0);
}