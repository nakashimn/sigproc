#include <windows.h>
#include <stdio.h>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>

int main(void) {
	cv::CascadeClassifier face_cascade;
	cv::Mat img;
	cv::Mat img_gray;
	std::vector<cv::Rect> faces;
	std::vector<int> num_detections;
	int min_neighbors = 2;
	int min_face = 72;
	int max_face = 0;
	face_cascade.load("C:\\Users\\4068979\\work_nakashima\\FaceProjectSDK_MultiPF\\Face_detector_wide_angle\\build\\VisualStudio\\product\\haarcascade_frontalface_alt2.xml");
	cv::VideoCapture cap("sub101_15fps.mp4");
	bool flag = cap.isOpened();
	cap >> img;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	face_cascade.detectMultiScale(img_gray, faces, num_detections, 1.2, min_neighbors, 0,
		cv::Size(min_face, min_face),
		cv::Size(max_face, max_face));
}