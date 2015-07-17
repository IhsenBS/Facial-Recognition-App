#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <omp.h>

using namespace cv;
using namespace std;
//circular array
int circular_array[10];
int currentI = -1;

void init(void){
	for (int i = 0; i < 10; i++)
		circular_array[i] = -1;
}

void addElement(int i){
	currentI = (currentI + 1) % 10;
	circular_array[currentI] = i;
}

int most_present_value(void){
	int maxCount = 0;
	int max = 0;
	for (int i = 0; i<10; i++)
	{
		int count = 1;
		for (int j = i + 1; j<10; j++)
			if (circular_array[i] == circular_array[j])
				count++;
		if (count > maxCount){
			maxCount = count;
			max = circular_array[i];
		}
	}
	return max;
}
//end circular array

int updated = 0;
int busy = 0;
int pos_x = 0;
int pos_y = 0;
int prediction = 0;
int avgPrediction = 0;
int im_width;
int im_height;
Rect_<int> face_rect;
Ptr<FaceRecognizer> model;
CascadeClassifier haar_cascade;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int treatement(Mat * frame){

	int b;

	#pragma omp critical
	{
		b = busy;
		if (busy == 0) busy = 1;
	}
	if (b) return 0;


	Mat original = (*frame).clone();
	Mat gray;
	cvtColor(original, gray, CV_BGR2GRAY);
	equalizeHist(gray, gray);
	vector< Rect_<int> > faces;
	haar_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE | CASCADE_FIND_BIGGEST_OBJECT, Size(75, 75));
	for (int i = 0; i < faces.size() && i < 1; i++) {
		Rect face_i = faces[i];
		Mat face = gray(face_i);
		Mat face_resized;
		cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
		#pragma omp critical
		{
			updated = 10;
			pos_x = std::max(face_i.tl().x - 10, 0);
			pos_y = std::max(face_i.tl().y - 10, 0);
			prediction = model->predict(face_resized);
			addElement(prediction);
			avgPrediction = most_present_value();
			face_rect = face_i;
		}
	}

	#pragma omp critical
	{
		busy = 0;
	}
	return 0;

}

int main(int argc, const char *argv[]) {

	// Get the path to your CSV:
	string fn_haar = "C:/projet_faces/haarcascade_frontalface_default.xml";
	string fn_csv = "C:/projet_faces/faces.csv";
	int deviceId = 0;
	vector<Mat> images;
	vector<int> labels;
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		system("pause");
	}
	im_width = images[0].cols;
	im_height = images[0].rows;
	model = createFisherFaceRecognizer();
	//model = createEigenFaceRecognizer();
	//model = createLBPHFaceRecognizer();
	model->train(images, labels);
	model->set("threshold", 4000);
	haar_cascade.load(fn_haar);
	// Get a handle to the Video device:
	VideoCapture cap(deviceId);
	// Check if we can use this device at all:
	if (!cap.isOpened()) {
		cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
		return -1;
	}
	// Holds the current frame from the Video device:
	Mat frame;
	thread t;
	double time = 0, time2 = omp_get_wtime(), delay = 0, avgDelay = 0, latency = 0;
	int i = 0, fps = 0, avgLatency=0;
	for (;;) {
		time = omp_get_wtime();
		cap >> frame;
		flip(frame, frame, 1);
		time2 = omp_get_wtime();
		t = thread(treatement, &frame);
		t.detach();
		int x, y, pred,avgPred,upd;
		Rect_<int> face;
		#pragma omp critical
		{	
			upd = updated--;
			x = pos_x;
			y = pos_y;
			pred = prediction;
			avgPred = avgPrediction;
			face = face_rect;
		}
		if (upd > 0){
			string box_text = format("Prediction = %d", avgPred);
			putText(frame, box_text, Point(x, y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.5);
			rectangle(frame, face, CV_RGB(0, 255, 0), 1);
		}

		string diff = format("%d FPS", fps);
		string lat = format("Latency : %d", avgLatency);
		putText(frame, diff, Point(0, 40), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 0), 1.5);
		putText(frame, lat, Point(0, 60), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 0), 1.5);
		imshow("face_recognizer", frame);
		latency += (omp_get_wtime() - time2) * 1000;
		char key = (char)waitKey(5);
		if (key == 27)
			break;
		delay += omp_get_wtime() - time;
		i++;
		if (i == 30){
			i = 0;
			avgLatency = (int)floor(latency / 30);
			avgDelay = delay / 30;
			fps =(int) floor( 1 / avgDelay);
			delay = 0;
			latency = 0;
		}
	}
	system("pause");
	return 0;
}