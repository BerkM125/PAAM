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
#include "personposemodel.h"
#include "preprocesslib.h"

using namespace cv;

extern cv::String PRIMARYWINDOW;
extern void loopProcessing(PersonPoseModel &person);

void processVideo(cv::String filename) {
	VideoCapture video(filename);

	PersonPoseModel mainPerson;

	bool frameSuccess = true;

	//int frame_width = static_cast<int>(video.get(3)*0.66 - video.get(3)*0.24);
	//int frame_height = static_cast<int>(video.get(4)/2);


	int frame_width = static_cast<int>(video.get(3));
	int frame_height = static_cast<int>(video.get(4));
	Size frame_size(frame_width, frame_height);
	int fps = 30;
	//Initialize video writer object
	VideoWriter output("output.mp4", VideoWriter::fourcc('m','p','4','v'),
		fps, frame_size);

	while (frameSuccess) {
		Mat rawFrame;
		frameSuccess = video.read(rawFrame);
		if (!frameSuccess) {
			break;
		}

		mainPerson.poseFrame = rawFrame;
		mainPerson.preprocessFrame(custom);

		mainPerson.loadROI(globalRegionBuffer[0]);
		mainPerson.loadOrigDimensions(Rect(0, 0, rawFrame.cols, rawFrame.rows));
		mainPerson.enableROIMode();

		mainPerson.forwardNet();
		mainPerson.poseFrame = rawFrame;
		
		mainPerson.renderPose();

		try {
			if (mainPerson.poseFrame.cols > 0 && mainPerson.poseFrame.rows > 0) {
				output.write(mainPerson.poseFrame);
			}
			else {
				throw mainPerson.poseFrame;
			}
		}
		catch (Mat m) {
			cout << "ERROR: poseFrame empty, dimensions are: " << m.cols << "x" << m.rows << ", continuing..." << endl;
		}


		if (waitKey(30) == 27) break;
	}
	video.release();
	output.release();
}

void videoLoop(void) {
	PersonPoseModel mainPerson;
	VideoCapture camera(0);
	
	camera.read(mainPerson.poseFrame);

	thread processingThread(loopProcessing, std::ref(mainPerson));
	processingThread.detach();
	// Pass frame through the network only every 5 seconds.
	// Adjust by FPS: If at 30 frames per second, pass thru network with every 30*5 = 150 frames
	while (true) {
		camera.read(mainPerson.poseFrame);

		mainPerson.renderPose();
		if (cv::waitKey(30) == 27) { //Escape upon ESC key being pressed
			break;
		}
	}
	
}