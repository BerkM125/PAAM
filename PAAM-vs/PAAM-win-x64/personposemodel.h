#pragma once

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

#include "constants.h"

using namespace cv;
using namespace std;

const string protoFile = "../../../estimation/openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
const string weightsFile = "../../../estimation/openpose/models/pose/mpi/pose_iter_160000.caffemodel";

typedef void(*preprocess)(Mat &frame);
typedef void(*preprocessOut)(Mat frame, Mat &output);

class PersonPoseModel {
public:
	Mat poseFrame;
	PersonPoseModel();
	PersonPoseModel(Mat poseFrame);
	PersonPoseModel(Mat poseFrame, dnn::Net poseNet);
	PersonPoseModel(Mat poseFrame, dnn::Net poseNet, vector<Point> loadingBuffer);
	PersonPoseModel(const PersonPoseModel& ppm);

	void loadPointBuffer(vector<Point> buffer);
	void loadPoseFrame(Mat frame);
	void loadNeuralNetwork(cv::String protoFile, cv::String weightsFile);
	void loadNeuralNetwork(dnn::Net net);

	void forwardNet(void);
	void forwardNet(Mat outputFrame);

	void renderPose(void);
	void renderPose(cv::String WINDOWNAME);
	
	void preprocessFrame(preprocess p) {
		p(poseFrame);
	}
	void preprocessFrame(preprocessOut p, Mat &output) {
		p(poseFrame, output);
	}
	Mat getFrame(void);
	~PersonPoseModel();
private:
	
	dnn::Net poseNet;
	vector<Point> keypointBuffer;

	void renderBodyLine(unsigned int pbIndex1, unsigned int pbIndex2);
	void renderPoseLines(void);
	void passNetworkFilter(void);

};