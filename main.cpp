#include<iostream>
#include <opencv2/opencv.hpp> 
#include<math.h>

using namespace std;
using namespace cv;


struct PointAndDistance {
	//��һ��ͼ��
	Mat one;
	//�ڶ���ͼ��
	Mat two;
	//����
	float distance;
};


struct Parallel {
	//��һ��ͼ��
	Mat one;
	//�ڶ���ͼ��
	Mat two;
};


/**
* @brief RGBתHSV����ת��Ϊ��ֵͼ
*
* @param img ��ת����ԭͼ
* @return Mat mask ת���õĶ�ֵͼ
*/
Mat getTwoValue(Mat img) {
	//imgHsv�洢HSVͼ��mask�����ֵͼ
	Mat imgHsv,mask;
	cvtColor(img, imgHsv, COLOR_BGR2HSV);
	//��Ƶ1�ƹ�ƫ�ƣ�ʹ�õ���[0,80,220][179,255,255]
	//��Ƶ2�ƹ��ɫ��ʹ�õ���[0,180,220][179,255,255]
	inRange(imgHsv, Scalar(0, 80, 220), Scalar(179, 255, 255), mask);
	return mask;
}


/**
* @brief ͨ�������븯ʴ����ڵ�
*
* @param img �������ԭͼ
* @param dial_interation ���͵�������
* @param erode_interation ��ʴ��������
* @return Mat imgErode ����ú��ͼ��
*/
Mat clearBlackPoint(Mat img,int dial_interation,int erode_interation) {
	//imgDial�������ͺ��ͼ��imgErode���港ʴ���ͼ��element�����
	Mat imgDial,imgErode, element = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(img,imgDial,element,Point(-1,-1),dial_interation);
	erode(imgDial, imgErode, element, Point(-1, -1), erode_interation);
	return imgErode;

}


/**
* @brief �����������ߵ��е�
*
* @param x ��һ����
* @param y �ڶ�����
* @return Point2f x,y �е�����
*/
Point2f middle(Point2f x, Point2f y) {
	return Point2f((x.x + y.x) / 2, (x.y + y.y) / 2);
}


/**
* @brief ������������
*
* @param x ��һ����
* @param y �ڶ�����
* @return float distance �������
*/
float calculateDistance(Point2f x,Point2f y){
	return sqrt(pow(double(x.x-y.x),2)+pow(double(x.y-y.y),2));
}


/**
* @brief �Ծ���boxPoints����������ĸ������������
*
* @param box ��������ĺ���
* @return Mat box �Ѿ�������ĺ���
*/
Mat correctBox(Mat box) {
	//�ĵ��е��x��y����
	float middle_x = 0;
	float middle_y = 0;

	//λ���е��Ϸ���������ļ���
	vector<Point2f> up;
	//λ���е��·���������ļ���
	vector<Point2f> low;

	//�����е������
	for (int i = 0; i < box.rows; i++) {
		middle_x += box.at<float>(i, 0);
		middle_y += box.at<float>(i, 1);
	}
	middle_x /= 4;
	middle_y /= 4;

	//���ĸ�������е��λ�ý��з���
	for (int i = 0; i < box.rows; i++) {
		if (box.at<float>(i, 1) > middle_y) {
			up.push_back(Point2f(box.at<float>(i, 0), box.at<float>(i, 1)));
		}
		else {
			low.push_back(Point2f(box.at<float>(i, 0), box.at<float>(i, 1)));
		}
	}

	//����������
	if (up[0].x<up[1].x) {
		for (int i = 0; i <2; i++) {
			float* ptr = box.ptr<float>(i);
			ptr[0] = up[i].x;
			ptr[1] = up[i].y;
		}
	}
	else {
		for (int i = 0; i <2; i++) {
			float* ptr = box.ptr<float>(i);
			ptr[0] = up[1-i].x;
			ptr[1] = up[1-i].y;
		}
	}
	if (low[0].x < low[1].x) {
		for (int i = 2; i < 4; i++) {
			float* ptr = box.ptr<float>(i);
			ptr[0] = low[3 - i].x;
			ptr[1] = low[3 - i].y;
		}
	}
	else {
		for (int i = 2; i < 4; i++) {
			float* ptr = box.ptr<float>(i);
			ptr[0] = low[i-2].x;
			ptr[1] = low[i-2].y;
		}
	}
	return box;
}


/**
* @brief �ҵ��໥ƽ�е�����
*
* @param boxes ���жϵĺ�����
* @return vector<Parallel> twoParallel ����������ƽ����ϵ�����
*/
vector<Parallel> findParallel(vector<Mat> boxes) {
	//������ʱ���м�����
	PointAndDistance temp;
	//�洢���ܴﵽƽ�����������������Լ����ֵ
	vector<PointAndDistance> results;
	//��Ϊ��ʱ�ֿ�洢�Ѿ�����������
	vector<Mat> store;
	//����������ƽ����ϵ�����
	vector<Parallel> twoParallel;

	//����ÿ�ֿ���
	for (int i = 0; i < boxes.size()-1; i++) {
		for (int j = i + 1; j < boxes.size(); j++) {
			//�Ӻ��������ó����ĺ��ӣ�����ѡ������
			Mat box = boxes[i];
			//����һ����������������
			Point2f x_up = middle(Point2f(box.at<float>(0,0), box.at<float>(0, 1)), Point2f(box.at<float>(1, 0), box.at<float>(1, 1)));
			Point2f x_down = middle(Point2f(box.at<float>(2, 0), box.at<float>(2, 1)), Point2f(box.at<float>(3, 0), box.at<float>(3, 1)));
			box = boxes[j];
			//��һ����������������
			Point2f y_up = middle(Point2f(box.at<float>(0, 0), box.at<float>(0, 1)), Point2f(box.at<float>(1, 0), box.at<float>(1, 1)));
			Point2f y_down = middle(Point2f(box.at<float>(2, 0), box.at<float>(2, 1)), Point2f(box.at<float>(3, 0), box.at<float>(3, 1)));
			
			//���������Խ��ߵ��е�
			Point2f first_point = middle(x_up, y_down);
			Point2f second_point = middle(x_down, y_up);
			
			//�����Խ����е�֮��ľ���
			float distance = calculateDistance(first_point, second_point);
			if (distance < 10) {
				//ÿ������������ƽ�е����������ֵ��
				PointAndDistance result = { boxes[i],boxes[j],distance };
				results.push_back(result);
			}
		}
	}

	//��������������ֵ��С���д�С���������
	for (int i = 0; i < results.size() - 1; i++) {
		for (int j = 1; j < results.size();j++) {
			if (results[results.size() - 1 - j].distance > results[results.size() - j].distance) {
				temp = results[results.size() - 1 - j];
				results[results.size() - 1 - j] = results[results.size() - j];
				results[results.size() - j] = temp;
			}
		}
	}

	//ͨ���㷨�������ܵ�����ƽ�����
	for (int i = 0; i < results.size(); i++) {
		PointAndDistance result = results[i];
		//�ж����������ǲ���һģһ���ĶȺ�
		int ifCorrects=0;
		for (int j = 0; j < store.size(); j++) {
			int ifCorrect = countNonZero(result.one - store[j]) && countNonZero(result.two - store[j]);
			if (ifCorrect != 0) {
				ifCorrects += 1;
			}

		}
		if (ifCorrects == 0) {
			twoParallel.push_back(Parallel{ result.one, result.two });
			store.push_back(result.one);
			store.push_back(result.two);
		}

	}
	return twoParallel;
}



/**
* @brief ��������Mat����תΪ�ɻ�����Point��
*
* @param box ��ת��������
* @return vector<Point2f> result �ɻ�����������
*/
vector<Point2f> turnToContours(Mat box) {
	//�ɻ��������������ڲ�
	vector<Point2f> result;
	for (int i = 0; i < box.rows; i++) {
		float* ptr = box.ptr<float>(i);
		result.push_back(Point2f(ptr[0], ptr[1]));
	}
	return result;
}


/**
* @brief ��̬��λ�ý���
*
* @param length ����ĳ���
* @param width ����Ŀ��
* @param fact_points ��άʵ�ʵ�
* @return vector<Mat> result ��ת������ƽ������
*/
vector<Mat> pnpCalculate(double length,double width,vector<Point2f> fact_points) {
	//װ�װ�İ볤����
	double half_length = length / 2;
	double half_width = width / 2;

	//����ڲ���������
	double camD[9] = { 1.2853517927598091e+03, 0., 3.1944768628958542e+02, 0.,
	  1.2792339468697937e+03, 2.3929354061292258e+02, 0., 0., 1. };
	double disD[5] = { 6.3687295852461456e-01, -1.9748008790347320e+00,
	   3.0970703651800782e-02, 2.1944646842516919e-03, 0. };
	Mat cam= Mat(3, 3, CV_64FC1, camD);
	Mat dis= Mat(5, 1, CV_64FC1, disD);
	

	//�Զ����������������
	vector<Point3f> obj = vector<Point3f>{
		Point3f(-half_length,half_width,0),
		Point3f(half_length,half_width,0),
		Point3f(-half_length,-half_width,0),
		Point3f(half_length,-half_width,0)
	};

	//������ת������ƽ������
	Mat rVec = Mat::zeros(3, 1, CV_64FC1);
	Mat tVec = Mat::zeros(3, 1, CV_64FC1);

	solvePnP(obj, fact_points, cam, dis, rVec, tVec, false, SOLVEPNP_ITERATIVE);

	//�������������Ľ��
	vector<Mat> result;
	result.push_back(rVec);
	result.push_back(tVec);
	return result;
}


/**
* @brief ����Ŀ���
*
* @param img һ��ʼ��ͼ��
* @param box_first ƽ�е�����һ������
* @param box_second ƽ�е�����һ������
* @param time �ڼ�������
*/
void picture(Mat &img,vector<Point2f> box_first, vector<Point2f>box_second,int time) {
	//���������
	vector<Point> center;
	//��������pnp���ĸ���
	vector<Point2f> fact_points;
	for (int i = 0; i < 4; i += 2) {
		Point2f first = middle(box_first[i], box_first[i+1]);
		Point2f second = middle(box_second[i], box_second[i+1]);
		fact_points.push_back(first);
		fact_points.push_back(second);
		center.push_back(Point(first));
		center.push_back(Point(second));
	}

	center.push_back(Point(middle(middle(Point2f(center[0]), Point2f(center[3])), middle(Point2f(center[1]), Point2f(center[2])))));
	for (int i = 0; i < 5; i++) {
		circle(img, center[i], 1, Scalar(255,0, 0), 5);
	}

	//�������������ľ���
	vector<Mat> rtVec= pnpCalculate(67.5, 26.5, fact_points);
	//��ת����
	Mat rotM= Mat::zeros(3, 3, CV_64FC1);

	Mat rVec = rtVec[0];
	Mat tVec = rtVec[1];
	Rodrigues(rVec, rotM);
	invert(rotM, rotM, DECOMP_LU);
	Mat result =rotM*tVec*-1;
	double distance= sqrt(pow(result.at<double>(0,0), 2) + pow(result.at<double>(1, 0), 2)+ pow(result.at<double>(2, 0), 2));
	
	cout << "��ͷ�����"<<time+1<<"��װ�װ����ĵľ���Ϊ��"<<distance/10<<"cm" << endl;
	
}


/**
* @brief ��һ֡����������
*
* @param img ����һ֡�Ļ���
*/
void onceTime(Mat img) {
	//�洢��ֵͼ
	Mat mask;
	//��������������
	vector<Mat> boxes;
	//�洢ƽ����ϵ�����������
	vector<Parallel> twoParallel;
	//�洢��������������
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	//ͼ��Ԥ����
	mask = getTwoValue(img);
	mask = clearBlackPoint(mask, 3, 1);
	
	//Ѱ������������Ԥɸѡ
	findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point());
	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > 100) {
			RotatedRect rect = minAreaRect(contours[i]);
			Mat box;
			boxPoints(rect, box);
			box=correctBox(box);
			boxes.push_back(box);
		}
	}

	//Ѱ��ƽ��
	twoParallel = findParallel(boxes);

	//������������
	for (int i = 0; i < twoParallel.size(); i++) {
		picture(img, turnToContours(twoParallel[i].one), turnToContours(twoParallel[i].two),i);
	}

	//չʾͼ��
	imshow("img", img);
}

int main() {
	//ʵ�������
	VideoCapture capture;
	//֡
	Mat frame;
	//��ȡ��Ƶ
	frame = capture.open("1.avi");

	//��ÿһ֡���в���
	while (capture.isOpened()) {
		capture.read(frame);
		onceTime(frame);
		waitKey(25);
	}

	destroyAllWindows();
	return 0;
}
