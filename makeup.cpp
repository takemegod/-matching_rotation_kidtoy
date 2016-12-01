#include "stdafx.h"

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/video/background_segm.hpp>  
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <ctype.h>
#include <thread>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <iomanip>
#include <crtdbg.h> //memory leak detector

#ifdef _DEBUG
#define TRACE(s) OutputDebugString(s)
#else // _DEBUG
#define TRACE(s)
#endif // _DEBUG

using namespace std;
using namespace cv;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// opencv 1st detection square 2nd object detection 3rd matching and comput position to return
// 2016-09-12 Author xyz email: yongzhexu@hotmail.com based code from opencv example 
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <xfeatures2d/nonfree.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "stats.h" // Stats structure definition
#include "utils.h" // Drawing and printing functions

using namespace cv;
using namespace std;

Mat src; Mat src_gray; Mat subimage_roi; Mat  matchMat; 
//for image rotation parameter
static int rotation_angle = 0;
Mat imgRotated;
//for subimage rect 
int x, y, w, h;
Rect bounding_rect; Rect bounding_rect2;
Mat canny_output; Mat subimage; Mat subimagerotation;

CvPoint pointx, pointy;
vector<vector<Point> > contours;
vector<vector<Point> > contours_sub;
vector<Vec4i> hierarchy;
vector<Point> approx_poly;
vector<Point2f> bb;
//check position value data
bool check_positiondata = false;
//time performance check
double t = (double)getTickCount();

//for canny edge threshold
int thresh = 80; int max_thresh = 255;
RNG rng(12345);
bool check_squre;
float queryidx_x, queryidx_y, trainidx_x, trainidx_y;
double max_dist = 0; double min_dist = 100; double angle(Point pt1, Point pt2, Point pt0);
long double average_value_max, average_value_min;
long double matching_count = 0;
long double data_avg = 0;

// Function header and parameter & thresh_callback find submatrix 
void thresh_callback(int, void*);
void cacFeature_Compare(cv::Mat img_object, cv::Mat img_scene, int angle);
//warpaffine rotation input submatrix
//Mat image2rotation(Mat subimagerotation, int angle);

//test code here
Mat sub_test1;
void thresh_square_detection(int, void*);
Mat test_drawing; Mat test_src_gray;
void thresh_callback_test(int, void*);
void surftest(Mat baseimge, Mat scene);
void useSurfDetector(Mat img_1, Mat img_2,int roi_angle);

Mat test_subimage;
Mat kaka; Mat kaka2;
Mat imagegray1, imagegray2, imageresult1, imageresult2;

//test code2 here
void cornerHarris_demo(Mat src_gray);

void cacFeature_Compare(cv::Mat img_object, cv::Mat img_scene, int angle)
{
	cout << "angle_value is : " << angle << endl;
	//rotation img_object 
	int iImageHieght = img_object.rows / 2 + 0.5;
	int iImageWidth = img_object.cols / 2 + 0.5;
	Mat matRotation = getRotationMatrix2D(Point(iImageWidth, iImageHieght), angle, 1); //change 0 to 180 rotation
	warpAffine(img_object, imgRotated, matRotation, img_object.size());
	img_object = imgRotated.clone();
	imgRotated.release();

	Mat img_matches;
	matching_count++;
	int matching_data = 0;
	CV_Assert(img_object.data != NULL && img_scene.data != NULL);
	// keypoint initial
	std::vector<KeyPoint> img_object_keypoint, img_scene_keypoint;
	//cv::Ptr<cv::ORB> detector = cv::ORB::create();
	//cv::Ptr<cv::ORB> detector = cv::ORB::create(800, 1.0f, 2, 10, 0, 2, 0, 10);
	//Ptr <xfeatures2d::SURF> detector = xfeatures2d::SURF::create(2000, 4, 3, true, true);
	//Ptr <xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create(800, 3, 0.04, 10, 1.6);
	//Ptr <xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
	Ptr <xfeatures2d::SURF> detector = xfeatures2d::SURF::create(); //maybe it's your silver buillet
	//cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
	
	detector->detect(img_object, img_object_keypoint);
	detector->detect(img_scene, img_scene_keypoint);

	// compute feature 
	Mat descriptorMat1, descriptorMat2;
	detector->compute(img_object, img_object_keypoint, descriptorMat1);
	detector->compute(img_scene, img_scene_keypoint, descriptorMat2);

	// feature matching for orb
	//BFMatcher matcher;
	//BFMatcher matcher(NORM_L2);
	//feature matching for surf
	//BFMatcher matcher(NORM_HAMMING); // orb
	FlannBasedMatcher matcher;

	vector<DMatch> mathces;
	vector <vector<DMatch>> matches2;
	vector< DMatch > good_matches;

	matcher.match(descriptorMat1, descriptorMat2, mathces);
	const float minratio = 1.f / 1.5f;
	mathces.clear();
	matcher.knnMatch(descriptorMat1, descriptorMat2, matches2, 2);
	try
	{
		//if (mathces.size() || matches2.size())
			
		for (size_t i = 0; i<matches2.size(); i++)
		{
			const cv::DMatch& bestMatch = matches2[i][0];
			const cv::DMatch& betterMatch = matches2[i][1];
			float distanceRatio = bestMatch.distance / betterMatch.distance;
			// Pass only matches where distance ratio between  
			// nearest matches is greater than 1.5  
			// (distinct criteria)  
			if (distanceRatio < minratio)
			{
				mathces.push_back(bestMatch);
			}
		}
			//drawMatches(img_object, keyPoint1, img_scene, keyPoint2, mathces, img_matches);

			//-- Show detected matches  
			// find good match 
		max_dist = 0;
		min_dist = 100;
				
		for (int i = 0; i < descriptorMat1.rows; i++)
		{
			double dist = mathces[i].distance;
				
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
				
			if (mathces[i].distance <= (3 * min_dist))
			{
				good_matches.push_back(mathces[i]);
				matching_data++;
			}
		}
			////matching data count and show avg matching dataset 
			//data_avg += matching_data;
			//cout << " data set is :   " << matching_data << endl;
			//cout << " count data is : " << matching_count << " avg data value is : " << (data_avg / matching_count) << endl;

		if (good_matches.size() == 0)
		{
			cout << good_matches.size() << endl;
			cout << "have empty good_matches resut count is: " << matching_count << endl;
		}
		//else if (good_matches.size() > 5)
		{
			////-- Localize the object
			std::vector<Point2f> obj;
			std::vector<Point2f> scene;

			drawMatches(img_object, img_object_keypoint, img_scene, img_scene_keypoint,
				good_matches, img_matches, Scalar(255, 0, 0), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::DEFAULT);
			
			//-- Show detected matches
			imshow("Good Matches & Object detection 1", img_matches);

			//-- Localize the object
			//std::vector<Point2f> obj;
			//std::vector<Point2f> scene;

			for (int i = 0; i < good_matches.size(); i++)
			{
				//-- Get the keypoints from the good matches
				obj.push_back(img_object_keypoint[good_matches[i].queryIdx].pt);
				scene.push_back(img_scene_keypoint[good_matches[i].trainIdx].pt);
			}

			Mat H = findHomography(obj, scene, CV_RANSAC);

			//-- Get the corners from the image_1 ( the object to be "detected" )
			std::vector<Point2f> obj_corners(4);
			std::vector<Point2f> scene_corners(4);
			
			obj_corners[0] = cvPoint(0, 0);
			obj_corners[1] = cvPoint(img_object.cols, 0);
			obj_corners[2] = cvPoint(img_object.cols, img_object.rows); 
			obj_corners[3] = cvPoint(0, img_object.rows);

			perspectiveTransform(obj_corners, scene_corners, H);

			cout.setf(ios::fixed);
			int distance_value_min = 15;
			int distance_value_max = 400;

			if (
				(scene_corners[1].x > scene_corners[0].x) && (scene_corners[2].x > scene_corners[3].x) &&
				(scene_corners[3].y > scene_corners[0].y) && (scene_corners[2].y > scene_corners[1].y) &&

				(scene_corners[0].x > 0) && (scene_corners[0].y > 0) &&
				(scene_corners[1].x > 0) && (scene_corners[1].y > 0) &&
				(scene_corners[2].x > 0) && (scene_corners[2].y > 0) &&
				(scene_corners[3].x > 0) && (scene_corners[3].y > 0) &&
				
				(abs(scene_corners[0].x - scene_corners[0].y > distance_value_min)) &&
				(abs(scene_corners[1].x - scene_corners[1].y > distance_value_min)) &&
				(abs(scene_corners[2].x - scene_corners[2].y > distance_value_min)) &&
				(abs(scene_corners[3].x - scene_corners[3].y > distance_value_min)) &&

				(abs(scene_corners[0].x - scene_corners[0].y) > distance_value_min) &&
				(abs(scene_corners[1].x - scene_corners[1].y) > distance_value_min) &&
				(abs(scene_corners[2].x - scene_corners[2].y) > distance_value_min) &&
				(abs(scene_corners[3].x - scene_corners[3].y) > distance_value_min) &&

				// distance from square fix small square small value each positon neeed different 
				(abs(scene_corners[0].x - scene_corners[1].x) > distance_value_min)	&&
				(abs(scene_corners[0].y - scene_corners[1].y) > distance_value_min) &&
				(abs(scene_corners[1].x - scene_corners[2].x) > distance_value_min) &&
				(abs(scene_corners[1].y - scene_corners[2].y) > distance_value_min) &&
				(abs(scene_corners[2].x - scene_corners[3].x) > distance_value_min) &&
				(abs(scene_corners[2].y - scene_corners[3].y) > distance_value_min) &&
				(abs(scene_corners[0].x - scene_corners[3].x) > distance_value_min) &&
				(abs(scene_corners[0].y - scene_corners[3].y) > distance_value_min) &&

				(abs(scene_corners[0].y - scene_corners[3].y) > distance_value_min) &&
				(abs(scene_corners[1].y - scene_corners[2].y) > distance_value_min) &&
				(abs(scene_corners[2].x - scene_corners[3].x) > distance_value_min)	&&
				(abs(scene_corners[1].y - scene_corners[3].y) > distance_value_min) &&
				(abs(scene_corners[1].x - scene_corners[3].x) > distance_value_min) &&
				(abs(scene_corners[0].x - scene_corners[2].x) > distance_value_min) &&
				(abs(scene_corners[0].y - scene_corners[2].y) > distance_value_min) &&

				// distance from square fix small square big value 
				(abs(scene_corners[0].x - scene_corners[1].x) < distance_value_max) &&
				(abs(scene_corners[0].y - scene_corners[3].y) < distance_value_max) &&
				(abs(scene_corners[1].y - scene_corners[2].y) < distance_value_max) &&
				(abs(scene_corners[2].x - scene_corners[3].x) < distance_value_max) &&
				(abs(scene_corners[0].y - scene_corners[1].y) + abs(scene_corners[3].y - scene_corners[2].y) > distance_value_min)
				)
			{
				cout << "it's value is nice to me " << endl;
				check_positiondata = true;

			}

			if (check_positiondata == true)
			{
				//-- Draw lines between the corners (the mapped object in the scene - image_2 )
				line(src, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
				line(src, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
				line(src, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
				line(src, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);

				cout << "******************" << endl;
				cout << setprecision(0) << scene_corners[0] + Point2f(img_object.cols, 0) << scene_corners[1] + Point2f(img_object.cols, 0) << endl;
				cout << setprecision(0) << scene_corners[1] + Point2f(img_object.cols, 0) << scene_corners[2] + Point2f(img_object.cols, 0) << endl;
				cout << setprecision(0) << scene_corners[2] + Point2f(img_object.cols, 0) << scene_corners[3] + Point2f(img_object.cols, 0) << endl;
				cout << setprecision(0) << scene_corners[3] + Point2f(img_object.cols, 0) << scene_corners[0] + Point2f(img_object.cols, 0) << endl;
				cout << "******************" << endl;

				cout << "angle is " << angle << endl;
				stringstream angle_w;
				angle_w << "Good Matches & Object detection result 1 angle is";
				angle_w << angle;
				cout << "***----------------------------------***" << endl;
				cout << "Good Matches & Object detection result 1" << endl;
				cout << "***----------------------------------***" << endl;
				imshow(angle_w.str(), src);

				cvWaitKey(0);
			}
			obj_corners.clear();
			scene_corners.clear();
			obj.clear();
			scene.clear();
			H.release();
			src.release();
		}
		check_positiondata = false;

		good_matches.clear();
		img_matches.release();
		img_object.release();
		img_scene.release();
		mathces.clear();
		descriptorMat1.release();
		descriptorMat2.release();
	}
	catch (exception e)
	{
		cout << "my function T^T ;" << endl;
	}
	
}

/** @function thresh_callback */
void thresh_callback(int, void*)
{
	check_squre = false;
	/// Detect edges using canny
	Canny(src_gray, canny_output, thresh, thresh * 2, 3);

	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));
	//findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Get the moments
	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());

	// Draw contours and Calculate the area with the moments 00 and compare with the result of the OpenCV function
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	///  Get the mass centers:

	for (int i = 0; i < (int)(contours.size()); i++)
	{
		Scalar color = Scalar(rng.uniform(255, 255), rng.uniform(255, 255), rng.uniform(255, 255));
		mu[i] = moments(contours[i], false);

		double eps = contours[i].size()*0.04;
		approxPolyDP(contours[i], approx_poly, eps, true);

		///*approx find square edge and rectangle and countare more than value *///
		if ((approx_poly.size() == 4) && fabs(contourArea(Mat(approx_poly))>1500 && isContourConvex(Mat(approx_poly))))
		{
			double maxCosine = 0;
			for (int j = 2; j < 5; j++)
			{
				// find the maximum cosine of the angle between joint edges
				double cosine = fabs(angle(approx_poly[j % 4], approx_poly[j - 2], approx_poly[j - 1]));
				maxCosine = MAX(maxCosine, cosine);
			}

			if (maxCosine < 0.3)
			{
				Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				mc[i] = Point2f((float)(mu[i].m10) / (float)(mu[i].m00), (float)(mu[i].m01) / (float)(mu[i].m00));

				bounding_rect = boundingRect(contours[i]);
				x = bounding_rect.x;
				y = bounding_rect.y;
				w = bounding_rect.width;
				h = bounding_rect.height;

				Rect rect(x, y, w, h);

				rectangle(src_gray, Point(x, y), Point(x + w, y + h), Scalar(255, 255, 255), 1, 1, 0);
				//subimage = src_gray(rect);
				rectangle(drawing, Point(x, y), Point(x + w, y + h), Scalar(255, 255, 255), 1, 1, 0);
				subimage = drawing(rect);

				subimage_roi = subimage.clone();
				check_squre = true;
				
				//test code improve performance thread
				//std::thread image2rotation_thread[119];
				//for (int count_t = 0; count_t < 360; count_t = count_t + 5)
				//{
					//image2rotation_thread[count_t] = (image2rotation(subimage, count_t));
				//}
				//subimagerotation = warpaffine(subimage, count_angle);

				//imshow("matrix show", subimage);
			}
		}
		else
		{
			check_squre = false;
			drawContours(src_gray, contours, i, color, 4, 8, hierarchy, 0, Point());
			drawContours(drawing, contours, i, color, 4, 8, hierarchy, 0, Point());
			mc[i] = Point2f((float)(mu[i].m10) / (float)(mu[i].m00), (float)(mu[i].m01) / (float)(mu[i].m00));
			//cornerHarris_demo(src_gray);
		}

		if (check_squre = true)
		{
			subimage_roi = subimage.clone();
			circle(drawing, mc[i], 4, color, -1, 8, 0);
		}
		//circle(drawing, mc[i], 4, color, -1, 8, 0);

	}

	if (check_squre == true)
	{
		src_gray(cv::Rect(x, y, w, h)) = 0;
		drawing(cv::Rect(x, y, w, h)) = 0;
	}

	// for count rectangle 
	vector<vector<Point> >poly(contours.size());
	findContours(canny_output, contours_sub, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (int k = 0; k < (int)(contours_sub.size()); k++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		double eps = contours[k].size()*0.16;
		approxPolyDP(contours[k], approx_poly, eps, true);
		approxPolyDP(Mat(contours[k]), poly[k], 5, true);
		drawContours(drawing, poly, k, color, 1, 8, vector<Vec4i>(), 0, Point());

		double maxCosine = 0;
		//for (int j = 0; j < 35; j++)
		//{
			// find the maximum cosine of the angle between joint edges
		//}
		for (maxCosine = 0; maxCosine < 1; maxCosine += 0.1)
		{
			double cosine = fabs(angle(approx_poly[k], approx_poly[k], approx_poly[k]));
			maxCosine = MAX(maxCosine, cosine);

			stringstream rect_sub_roi;
			rect_sub_roi << k;

			cv::Rect bounding_rect2 = boundingRect(contours_sub[k]);
			x = bounding_rect2.x;
			y = bounding_rect2.y;
			w = bounding_rect2.width;
			h = bounding_rect2.height;
			Rect rect2(x, y, w, h);

			//if (fabs(contourArea(Mat(approx_poly)) > 10))

			rectangle(src_gray, Point(x, y), Point(x + w, y + h), Scalar(255, 255, 255), 1, 1, 0);
			rectangle(drawing, Point(x, y), Point(x + w, y + h), Scalar(255, 255, 255), 1, 1, 0);

			//cout << "approx_poly.size() is : " << approx_poly.size() << endl;

			int data_conx = poly[k].size();
			char data_pchar[6];
			const char *pchar = _itoa(data_conx, data_pchar, 5);

			Mat rect_sub_image_roi = drawing(rect2);
			IplImage ipl_img(rect_sub_image_roi);
			CvFont font;
			cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1, 0.25, 0.25, 0.25, 1);
			cvPutText(&ipl_img, pchar, cvPoint(15, 15), &font, CV_RGB(255, 255, 0));

			Mat rect_sub_image_roi2 = src_gray(rect2);
			IplImage ipl_img2(rect_sub_image_roi2);
			CvFont font2;
			cvInitFont(&font2, CV_FONT_HERSHEY_COMPLEX, 1, 0.25, 0.25, 0.25, 1);
			cvPutText(&ipl_img2, pchar, cvPoint(15, 15), &font2, CV_RGB(255, 255, 0));

			//imshow(rect_sub_roi.str(), rect_sub_image_roi);
			//cvWaitKey(0);

			//imshow(rect_sub_roi.str(), rect_sub_image_roi);
		}
	}


	if ((!src_gray.empty()) || (check_squre == true))
	{
		if ((!subimage_roi.empty()) && (!src_gray.empty()))
		{
			//Mat img_test = imread("rick2.png");
			//cvtColor(img_test, img_test, CV_BGR2GRAY);
			//cacFeature_Compare(img_test, src_gray);
			
			//cacFeature_Compare(subimage_roi, drawing, 0);
			//test code for thread language 
			int angle_value = 0;
			imshow("subimage roi ", subimage_roi);

			//cacFeature_Compare(subimage_roi, drawing, 0);
			for (angle_value = 0; angle_value <= 360; angle_value = angle_value + 3)
			{
				cout << angle_value << endl;
				//cacFeature_Compare(subimage_roi, drawing, angle_value);
				//useSurfDetector(subimage_roi, drawing, angle_value);
				std::thread t_subimage_roi1(cacFeature_Compare, subimage_roi, drawing, angle_value);
				cvWaitKey(100);
				std::thread t_subimage_roi2(cacFeature_Compare, subimage_roi, drawing, angle_value +90);
				cvWaitKey(100);
				std::thread t_subimage_roi3(cacFeature_Compare, subimage_roi, drawing, angle_value +180);
				cvWaitKey(100);
				std::thread t_subimage_roi4(cacFeature_Compare, subimage_roi, drawing, angle_value +270);
				cvWaitKey(100);
	
				t_subimage_roi1.join();
				t_subimage_roi2.join();
				t_subimage_roi3.join();
				t_subimage_roi4.join();
			}

		}
		else
		{
			cout << "subimage_roi and src_gray image can't find" << endl;
		}
	}

	/// Show in a window
	//cv::namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	cv::imshow("Contours", drawing);
	//cv::imshow("blurgray", src_gray);
	subimage_roi.release();
	src_gray.release();
	canny_output.release();
	drawing.release();
	contours.clear();
	approx_poly.clear();
	matchMat.release();
	test_drawing.release();
	test_src_gray.release();
}

/** @thresh_callback 函数 */
void thresh_square_detection(int, void*)
{
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// 使用Canndy检测边缘
	Canny(src_gray, canny_output, thresh, thresh * 2, 3);
	/// 找到轮廓
	findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// 计算矩
	vector<Moments> mu(contours.size());
	vector<Moments> mu_last(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	/// caculate matrix center
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	/// draw silhouete
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		//circle(drawing, mc[i], 4, color, -1, 8, 0);
	}

	/// show window 
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);

	cout << "contours.size is : " << contours.size() << endl;
	/// 通过m00计算轮廓面积并且和OpenCV函数比较
	//printf("\t Info: Area and Contour Length \n");
	for (int i = 0; i< contours.size(); i++)
	{
		cout << " contour area is : " << mu[i].m00 << endl;
		cout << " contour position value set is :" << contours[i] << endl;
		//printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", i, mu[i].m00, contourArea(contours[i]), arcLength(contours[i], true));
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		//circle(drawing, mc[i], 4, color, -1, 8, 0);
	}
}

/** @function angle  */
double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


/** @function main */
int main()
{
	matching_count = 1;
	Mat image;
	VideoCapture cap(0);

	if (!cap.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }
	
	//Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
	
	/// Load source image and convert it to gray
	while (true)
	{
		//test code 3 here 
		//src = imread("silhouette_test.png");
		src = cap.read(image);
		cap >> src;

		/// Convert image to gray and blur it
		//定义核  

		//进行形态学操作  
		//morphologyEx(src, src, MORPH_ERODE, element);
		//imshow("after black hat ", src);
		cvtColor(src, src_gray, CV_BGR2GRAY);
		//blur(src_gray, src_gray, Size(3, 3));

		/// Create Window
		char* source_window = "Source";
		cv::namedWindow(source_window, CV_WINDOW_AUTOSIZE);
		cv::imshow(source_window, src);

		createTrackbar("Canny thresh:", "Source", &thresh, max_thresh, thresh_callback);
		thresh_callback(0, 0);
		//createTrackbar("Canny thresh:", "Source", &thresh, max_thresh, thresh_callback_test);
		//thresh_callback_test(0, 0);

		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}

		src.release();
		image.release();
		imgRotated.release();
		src_gray.release();
		_CrtDumpMemoryLeaks();
	}
	return(0);
}

//test code 

/** @function thresh_callback */
void thresh_callback_test(int, void*)
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// 对图像进行二值化  
	threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);

	/// 寻找轮廓  
	findContours(threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	/*Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);*/

	/* 对每个轮廓计算其凸包*/
	vector<vector<Point> >poly(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), poly[i], 5, true);
	}

	/* 绘出轮廓及其凸包*/
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);

	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(255, 255), rng.uniform(255, 255), rng.uniform(255, 255));

		drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(drawing, poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		cout << poly[i].size() << endl;
	}


	// matchshape use one same image vector not two!!!
	RNG rng(12345);

	kaka = imread("rick3.png", 1);
	kaka2 = drawing;

	cvtColor(kaka, imagegray1, CV_BGR2GRAY);
	cvtColor(kaka2, imagegray2, CV_BGR2GRAY);

	vector<vector<Point>>contours1, contours2;
	vector<Vec4i>hierarchy1, hierarchy2;
	double ans = 0, result = 0;

	Canny(imagegray1, imageresult1, thresh, thresh * 2);
	Canny(imagegray2, imageresult2, thresh, thresh * 2);

	try
	{
		findContours(imageresult1, contours1, hierarchy1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
		for (int i = 0; i<contours1.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(imageresult1, contours1, i, color, 1, 8, hierarchy1, 0, Point());
		}

		findContours(imageresult2, contours2, hierarchy2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
		for (int i = 0; i<contours2.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(imageresult2, contours2, i, color, 1, 8, hierarchy2, 0, Point());
		}
		//cout << contours1[1] << contours2[1] << endl;
		//ans = matchShapes(contours1[1], contours2[1], CV_CONTOURS_MATCH_I1, 0);
		//cout << ans << endl;
		//cvWaitKey(1000);
	}
	catch (exception e)
	{
		cout << "computer too slow have copy need more time hahaha" << endl;
	}

	try
	{
		hierarchy1.clear();
		hierarchy2.clear();
		contours1.clear();
		contours2.clear();
		imageresult1.release();
		imageresult2.release();
		imagegray1.release();
		imagegray2.release();
		kaka.release();
		kaka2.release();
	}
	catch (const std::exception&)
	{
		cout << "memory freeze have problem" << endl;
	}
}

void cornerHarris_demo(Mat sub_src)
{
	Mat dst, dst_norm;	// , dst_norm_scaled;
	dst = Mat::zeros(src_gray.size(), CV_32FC1);
	/// Detector parameters  
	int blockSize = 3;
	int apertureSize = 3;
	double k = 0.01;
	/// Detecting corners  
	cornerHarris(sub_src, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
	/// Normalizing  
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//convertScaleAbs(dst_norm, dst_norm_scaled);
	/// Drawing a circle around corners  
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				//circle(dst_norm_scaled, Point(i, j), 5, Scalar(255, 0, 0), -1, 8, 0);
				circle(src, Point(i, j), 5, Scalar(255, 0, 0), -1, 8, 0);
				//cout << Point(i, j) << endl; 
			}
		}
	}
	/// Showing the result  
	//imshow("corners_window", dst_norm_scaled);
	imshow("source_window", src);

}

//angle 360 to 4 quater 90, 180, 270, 360 
Mat image2rotation(Mat subimagerotation, int angle)
{
	if (rotation_angle >= 90)
	{
		rotation_angle = 0;
		angle = 0;
	}

	int iImageHieght = subimagerotation.rows / 2 + 0.5;
	int iImageWidth = subimagerotation.cols / 2 + 0.5;
	Mat matRotation = getRotationMatrix2D(Point(iImageWidth, iImageHieght), rotation_angle, 1); //change 0 to 180 rotation
	warpAffine(subimagerotation, imgRotated, matRotation, subimagerotation.size());
	subimagerotation = imgRotated.clone();
	imshow("rotation subimage ", subimagerotation);
	rotation_angle = angle + 2;
	
	return subimagerotation;
}


//test code 4 
void useSurfDetector(Mat img_1, Mat img_2, int roi_angle)
{
	imshow("subimage rotaiton", img_1);
	vector< DMatch > good_matches;
	//rotation img_object 
	int iImageHieght = img_1.rows / 2 + 0.5;
	int iImageWidth = img_1.cols / 2 + 0.5;
	Mat matRotation = getRotationMatrix2D(Point(iImageWidth, iImageHieght), roi_angle, 1); //change 0 to 180 rotation
	warpAffine(img_1, imgRotated, matRotation, img_1.size());
	img_1 = imgRotated.clone();
	imgRotated.release();
	imshow("subimage", img_1);

	good_matches.clear();
	double t = (double)getTickCount();

	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptor_1, descriptor_2;

	int minHessian = 2000;
	//Ptr <xfeatures2d::SURF> detector = xfeatures2d::SURF::create(minHessian, 4, 3, true, true);
	Ptr <xfeatures2d::SURF> detector = xfeatures2d::SURF::create();
	//cv::Ptr<cv::ORB> detector = cv::ORB::create(minHessian);

	// Step -1, Detect keypoints using SURF detector
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);
	// Step -2, Calculate descriptors (feature vector)
	
	detector->compute(img_1, keypoints_1, descriptor_1);
	detector->compute(img_2, keypoints_2, descriptor_2);

	//step - 3, Matching descriptor vectors with a brute force mathcher

	BFMatcher matcher(NORM_L2);//surf 
	//BFMatcher matcher(NORM_HAMMING); //orb

	vector<DMatch> matches;
	matcher.match(descriptor_1, descriptor_2, matches);
	// quick calcualation of max and min distances between keypoints

	double max_dist = 0; double min_dist = 1000;
	for (int i = 0; i < descriptor_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (max_dist<dist) max_dist = dist;
		if (min_dist>dist) min_dist = dist;
	}
	//cout << " SURF Time (senconds):  " << t << endl;

	for (int i = 0; i<descriptor_1.rows; i++)
	{
		if (matches[i].distance<3 * min_dist)
			good_matches.push_back(matches[i]);
	}

	// Draw Good Matches
	Mat img_goodmatches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatches);
	imshow("Good Matches & Object detection 2", img_goodmatches);

	try
	{
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;

		for (int i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
		}

		Mat H = findHomography(obj, scene, CV_RANSAC);

		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);

		obj_corners[0] = cvPoint(0, 0);
		obj_corners[1] = cvPoint(img_1.cols, 0);
		obj_corners[2] = cvPoint(img_1.cols, img_1.rows); obj_corners[3] = cvPoint(0, img_1.rows);
		std::vector<Point2f> scene_corners(4);

		perspectiveTransform(obj_corners, scene_corners, H);

		cout.setf(ios::fixed);
		int distance_value_min = 0;
		int distance_value_max = 300;

		if ((scene_corners[1].x > scene_corners[0].x) && (scene_corners[2].x > scene_corners[3].x) &&
			(scene_corners[3].y > scene_corners[0].y) && (scene_corners[2].y > scene_corners[1].y) &&

			(scene_corners[0].x > 0) && (scene_corners[0].y > 0) &&
			(scene_corners[1].x > 0) && (scene_corners[1].y > 0) &&
			(scene_corners[2].x > 0) && (scene_corners[2].y > 0) &&
			(scene_corners[3].x > 0) && (scene_corners[3].y > 0) &&

			(abs(scene_corners[0].x - scene_corners[0].y) > distance_value_min) &&
			(abs(scene_corners[1].x - scene_corners[1].y) > distance_value_min) &&
			(abs(scene_corners[2].x - scene_corners[2].y) > distance_value_min) &&
			(abs(scene_corners[3].x - scene_corners[3].y) > distance_value_min) &&

			// distance from square fix small square small value each positon neeed different 
			(abs(scene_corners[0].x - scene_corners[1].x) > distance_value_min) &&
			(abs(scene_corners[0].y - scene_corners[1].y) > distance_value_min) &&
			(abs(scene_corners[1].x - scene_corners[2].x) > distance_value_min) &&
			(abs(scene_corners[1].y - scene_corners[2].y) > distance_value_min) &&
			(abs(scene_corners[2].x - scene_corners[3].x) > distance_value_min) &&
			(abs(scene_corners[2].y - scene_corners[3].y) > distance_value_min) &&
			(abs(scene_corners[0].x - scene_corners[3].x) > distance_value_min) &&
			(abs(scene_corners[0].y - scene_corners[3].y) > distance_value_min) &&

			(abs(scene_corners[0].y - scene_corners[3].y) > distance_value_min) &&
			(abs(scene_corners[1].y - scene_corners[2].y) > distance_value_min) &&
			(abs(scene_corners[2].x - scene_corners[3].x) > distance_value_min) &&
			(abs(scene_corners[1].y - scene_corners[3].y) > distance_value_min) &&
			(abs(scene_corners[1].x - scene_corners[3].x) > distance_value_min) &&
			(abs(scene_corners[0].x - scene_corners[2].x) > distance_value_min) &&
			(abs(scene_corners[0].y - scene_corners[2].y) > distance_value_min) &&

			//// distance from square fix small square big value 
			(abs(scene_corners[0].x - scene_corners[1].x) < distance_value_max) &&
			(abs(scene_corners[0].y - scene_corners[3].y) < distance_value_max) &&
			(abs(scene_corners[1].y - scene_corners[2].y) < distance_value_max) &&
			(abs(scene_corners[2].x - scene_corners[3].x) < distance_value_max) &&
			(abs(scene_corners[0].y - scene_corners[1].y) + abs(scene_corners[3].y - scene_corners[2].y) > distance_value_min)
			)
		{
			cout << "it's value is nice to me " << endl;
			check_positiondata = true;

		}

		if (check_positiondata == true)
		{
			//-- Draw lines between the corners (the mapped object in the scene - image_2 )
			line(src, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
			line(src, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
			line(src, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
			line(src, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);

			cout << "******************" << endl;
			cout << setprecision(0) << scene_corners[0] + Point2f(img_1.cols, 0) << scene_corners[1] + Point2f(img_1.cols, 0) << endl;
			cout << setprecision(0) << scene_corners[1] + Point2f(img_1.cols, 0) << scene_corners[2] + Point2f(img_1.cols, 0) << endl;
			cout << setprecision(0) << scene_corners[2] + Point2f(img_1.cols, 0) << scene_corners[3] + Point2f(img_1.cols, 0) << endl;
			cout << setprecision(0) << scene_corners[3] + Point2f(img_1.cols, 0) << scene_corners[0] + Point2f(img_1.cols, 0) << endl;
			cout << "******************" << endl;

			cout << "angle is " << roi_angle << endl;
			stringstream angle_w;
			angle_w << "Good Matches & Object detection result 2 angle is";
			angle_w << roi_angle;
			cout << "***----------------------------------***" << endl;
			cout << "Good Matches & Object detection result 2" << endl;
			cout << "***----------------------------------***" << endl;
			imshow(angle_w.str(), src);

			cvWaitKey(0);
		}

		obj_corners.clear();
		scene_corners.clear();
		obj.clear();
		scene.clear();
		H.release();
		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "time is " << t << endl;
		check_positiondata = false;
	}
	catch (exception e)
	{
		cout << "it's also impposoble" << endl;
	}
}