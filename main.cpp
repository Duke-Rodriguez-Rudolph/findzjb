#include<iostream>
#include <opencv2/opencv.hpp> 
#include<math.h>

using namespace std;
using namespace cv;


struct PointAndDistance {
	//第一个图像
	Mat one;
	//第二个图像
	Mat two;
	//距离
	float distance;
};


struct Parallel {
	//第一个图像
	Mat one;
	//第二个图像
	Mat two;
};


/**
* @brief RGB转HSV，并转化为二值图
*
* @param img 待转换的原图
* @return Mat mask 转换好的二值图
*/
Mat getTwoValue(Mat img) {
	//imgHsv存储HSV图，mask储存二值图
	Mat imgHsv,mask;
	cvtColor(img, imgHsv, COLOR_BGR2HSV);
	//视频1灯光偏黄，使用的是[0,80,220][179,255,255]
	//视频2灯光红色，使用的是[0,180,220][179,255,255]
	inRange(imgHsv, Scalar(0, 80, 220), Scalar(179, 255, 255), mask);
	return mask;
}


/**
* @brief 通过膨胀与腐蚀清除黑点
*
* @param img 待处理的原图
* @param dial_interation 膨胀迭代次数
* @param erode_interation 腐蚀迭代次数
* @return Mat imgErode 处理好后的图像
*/
Mat clearBlackPoint(Mat img,int dial_interation,int erode_interation) {
	//imgDial储存膨胀后的图，imgErode储存腐蚀后的图，element储存核
	Mat imgDial,imgErode, element = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(img,imgDial,element,Point(-1,-1),dial_interation);
	erode(imgDial, imgErode, element, Point(-1, -1), erode_interation);
	return imgErode;

}


/**
* @brief 返回两点连线的中点
*
* @param x 第一个点
* @param y 第二个点
* @return Point2f x,y 中点坐标
*/
Point2f middle(Point2f x, Point2f y) {
	return Point2f((x.x + y.x) / 2, (x.y + y.y) / 2);
}


/**
* @brief 计算两点间距离
*
* @param x 第一个点
* @param y 第二个点
* @return float distance 两点距离
*/
float calculateDistance(Point2f x,Point2f y){
	return sqrt(pow(double(x.x-y.x),2)+pow(double(x.y-y.y),2));
}


/**
* @brief 对经过boxPoints函数处理的四个点进行重排序
*
* @param box 待重排序的盒子
* @return Mat box 已经重排序的盒子
*/
Mat correctBox(Mat box) {
	//四点中点的x与y坐标
	float middle_x = 0;
	float middle_y = 0;

	//位于中点上方的两个点的集合
	vector<Point2f> up;
	//位于中点下方的两个点的集合
	vector<Point2f> low;

	//计算中点的坐标
	for (int i = 0; i < box.rows; i++) {
		middle_x += box.at<float>(i, 0);
		middle_y += box.at<float>(i, 1);
	}
	middle_x /= 4;
	middle_y /= 4;

	//对四个点相对中点的位置进行分类
	for (int i = 0; i < box.rows; i++) {
		if (box.at<float>(i, 1) > middle_y) {
			up.push_back(Point2f(box.at<float>(i, 0), box.at<float>(i, 1)));
		}
		else {
			low.push_back(Point2f(box.at<float>(i, 0), box.at<float>(i, 1)));
		}
	}

	//进行重排序
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
* @brief 找到相互平行的轮廓
*
* @param boxes 待判断的盒子们
* @return vector<Parallel> twoParallel 储存有两两平行组合的向量
*/
vector<Parallel> findParallel(vector<Mat> boxes) {
	//作互换时的中间容器
	PointAndDistance temp;
	//存储可能达到平行条件的两两轮廓以及误差值
	vector<PointAndDistance> results;
	//作为临时仓库存储已经遍历的轮廓
	vector<Mat> store;
	//储存有两两平行组合的向量
	vector<Parallel> twoParallel;

	//遍历每种可能
	for (int i = 0; i < boxes.size()-1; i++) {
		for (int j = i + 1; j < boxes.size(); j++) {
			//从盒子们中拿出来的盒子（即待选轮廓）
			Mat box = boxes[i];
			//其中一个轮廓的上下两点
			Point2f x_up = middle(Point2f(box.at<float>(0,0), box.at<float>(0, 1)), Point2f(box.at<float>(1, 0), box.at<float>(1, 1)));
			Point2f x_down = middle(Point2f(box.at<float>(2, 0), box.at<float>(2, 1)), Point2f(box.at<float>(3, 0), box.at<float>(3, 1)));
			box = boxes[j];
			//另一个轮廓的上下两点
			Point2f y_up = middle(Point2f(box.at<float>(0, 0), box.at<float>(0, 1)), Point2f(box.at<float>(1, 0), box.at<float>(1, 1)));
			Point2f y_down = middle(Point2f(box.at<float>(2, 0), box.at<float>(2, 1)), Point2f(box.at<float>(3, 0), box.at<float>(3, 1)));
			
			//计算两条对角线的中点
			Point2f first_point = middle(x_up, y_down);
			Point2f second_point = middle(x_down, y_up);
			
			//两条对角线中点之间的距离
			float distance = calculateDistance(first_point, second_point);
			if (distance < 10) {
				//每个（两个疑似平行的轮廓，误差值）
				PointAndDistance result = { boxes[i],boxes[j],distance };
				results.push_back(result);
			}
		}
	}

	//对上述结果就误差值大小进行从小到大的排序
	for (int i = 0; i < results.size() - 1; i++) {
		for (int j = 1; j < results.size();j++) {
			if (results[results.size() - 1 - j].distance > results[results.size() - j].distance) {
				temp = results[results.size() - 1 - j];
				results[results.size() - 1 - j] = results[results.size() - j];
				results[results.size() - j] = temp;
			}
		}
	}

	//通过算法揪出可能的两两平行组合
	for (int i = 0; i < results.size(); i++) {
		PointAndDistance result = results[i];
		//判断两个轮廓是不是一模一样的度衡
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
* @brief 将轮廓从Mat类型转为可画出的Point型
*
* @param box 待转换的轮廓
* @return vector<Point2f> result 可画出来的轮廓
*/
vector<Point2f> turnToContours(Mat box) {
	//可画出来的轮廓的内部
	vector<Point2f> result;
	for (int i = 0; i < box.rows; i++) {
		float* ptr = box.ptr<float>(i);
		result.push_back(Point2f(ptr[0], ptr[1]));
	}
	return result;
}


/**
* @brief 姿态与位置解算
*
* @param length 物体的长度
* @param width 物体的宽度
* @param fact_points 二维实际点
* @return vector<Mat> result 旋转向量与平移向量
*/
vector<Mat> pnpCalculate(double length,double width,vector<Point2f> fact_points) {
	//装甲板的半长与半宽
	double half_length = length / 2;
	double half_width = width / 2;

	//相机内参与畸变矩阵
	double camD[9] = { 1.2853517927598091e+03, 0., 3.1944768628958542e+02, 0.,
	  1.2792339468697937e+03, 2.3929354061292258e+02, 0., 0., 1. };
	double disD[5] = { 6.3687295852461456e-01, -1.9748008790347320e+00,
	   3.0970703651800782e-02, 2.1944646842516919e-03, 0. };
	Mat cam= Mat(3, 3, CV_64FC1, camD);
	Mat dis= Mat(5, 1, CV_64FC1, disD);
	

	//自定义物体的世界坐标
	vector<Point3f> obj = vector<Point3f>{
		Point3f(-half_length,half_width,0),
		Point3f(half_length,half_width,0),
		Point3f(-half_length,-half_width,0),
		Point3f(half_length,-half_width,0)
	};

	//定义旋转向量与平移向量
	Mat rVec = Mat::zeros(3, 1, CV_64FC1);
	Mat tVec = Mat::zeros(3, 1, CV_64FC1);

	solvePnP(obj, fact_points, cam, dis, rVec, tVec, false, SOLVEPNP_ITERATIVE);

	//储存两个向量的结果
	vector<Mat> result;
	result.push_back(rVec);
	result.push_back(tVec);
	return result;
}


/**
* @brief 画出目标点
*
* @param img 一开始的图像
* @param box_first 平行的其中一个轮廓
* @param box_second 平行的另外一个轮廓
* @param time 第几个轮廓
*/
void picture(Mat &img,vector<Point2f> box_first, vector<Point2f>box_second,int time) {
	//储存五个点
	vector<Point> center;
	//储存塞进pnp的四个点
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

	//储存两个变量的矩阵
	vector<Mat> rtVec= pnpCalculate(67.5, 26.5, fact_points);
	//旋转矩阵
	Mat rotM= Mat::zeros(3, 3, CV_64FC1);

	Mat rVec = rtVec[0];
	Mat tVec = rtVec[1];
	Rodrigues(rVec, rotM);
	invert(rotM, rotM, DECOMP_LU);
	Mat result =rotM*tVec*-1;
	double distance= sqrt(pow(result.at<double>(0,0), 2) + pow(result.at<double>(1, 0), 2)+ pow(result.at<double>(2, 0), 2));
	
	cout << "镜头距离第"<<time+1<<"个装甲板中心的距离为："<<distance/10<<"cm" << endl;
	
}


/**
* @brief 对一帧的完整操作
*
* @param img 输入一帧的画面
*/
void onceTime(Mat img) {
	//存储二值图
	Mat mask;
	//储存轮廓的向量
	vector<Mat> boxes;
	//存储平行组合的轮廓的向量
	vector<Parallel> twoParallel;
	//存储查找轮廓的内容
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	//图像预处理
	mask = getTwoValue(img);
	mask = clearBlackPoint(mask, 3, 1);
	
	//寻找轮廓并进行预筛选
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

	//寻找平行
	twoParallel = findParallel(boxes);

	//将轮廓画出来
	for (int i = 0; i < twoParallel.size(); i++) {
		picture(img, turnToContours(twoParallel[i].one), turnToContours(twoParallel[i].two),i);
	}

	//展示图像
	imshow("img", img);
}

int main() {
	//实例化相机
	VideoCapture capture;
	//帧
	Mat frame;
	//读取视频
	frame = capture.open("1.avi");

	//对每一帧进行操作
	while (capture.isOpened()) {
		capture.read(frame);
		onceTime(frame);
		waitKey(25);
	}

	destroyAllWindows();
	return 0;
}
