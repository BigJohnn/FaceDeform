#ifndef _ANNOTATE_H
#define _ANNOTATE_H

#include "ft.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

extern void warpTextureFromTriangle(Point2f srcTri[3], Mat &originalImage, Point2f dstTri[3], Mat &warp_final);
extern void restoreTextureFromTriangle(Mat &originalImage, Point2f dstTri[3], Mat &warp_final);
//==============================================================================
class annotate{
public:
	int idx;                       //index of image to annotate
	int pidx;                      //index of point to manipulate
	int idx_dst;//Ä¿±êÍ¼ÏñË÷Òý
	vector<Vec6f> triangleList;
	vector<Vec6f> triangleList_dst;

	cv::Mat image;                     //current image to display 
	cv::Mat image_dst;
	cv::Mat image_clean;               //clean image to display
	ft_data data;                  //annotation data
	const char* wname;             //display window name
	vector<string> instructions;   //annotation instructions

	annotate(){ wname = "Annotate"; idx = 0; pidx = -1; }

	int
		set_current_image(const int idx = 0){
		if ((idx < 0) || (idx > int(data.imnames.size())))return 0;
		image = data.get_image(idx, 2); return 1;
	}
	int
		set_dst_image(const int idx = 0){
		if ((idx < 0) || (idx > int(data.imnames.size())))return 0;
		image_dst = data.get_image(idx, 2); return 1;
	}
	void
		set_clean_image(){
		image_clean = image.clone();
	}
	void
		copy_clean_image(){
		image_clean.copyTo(image);
	}
	void
		draw_instructions(){
		if (image.empty())return;
		this->draw_strings(image, instructions);
	}
	void
		draw_points(){
		data.draw_points(image, idx);
	}
	void
		draw_chosen_point(){
		if (pidx >= 0)circle(image, data.points[idx][pidx], 1, CV_RGB(0, 255, 0), 2, CV_AA);
	}
	void
		draw_connections(){
		int m = data.connections.size();
		if (m == 0)this->draw_points();
		else{
			if (data.connections[m - 1][1] < 0){
				int i = data.connections[m - 1][0];
				data.connections[m - 1][1] = i;
				data.draw_connect(image, idx); this->draw_points();
				circle(image, data.points[idx][i], 1, CV_RGB(0, 255, 0), 2, CV_AA);
				data.connections[m - 1][1] = -1;
			}
			else{ data.draw_connect(image, idx); this->draw_points(); }
		}
	}
	void
		draw_symmetry(){
		this->draw_points(); this->draw_connections();
		for (int i = 0; i < int(data.symmetry.size()); i++){
			int j = data.symmetry[i];
			if (j != i){
				circle(image, data.points[idx][i], 1, CV_RGB(255, 255, 0), 2, CV_AA);
				circle(image, data.points[idx][j], 1, CV_RGB(255, 255, 0), 2, CV_AA);
			}
		}
		if (pidx >= 0)circle(image, data.points[idx][pidx], 1, CV_RGB(0, 255, 0), 2, CV_AA);
	}
	void
		set_capture_instructions(){
		instructions.clear();
		instructions.push_back(string("Select expressive frames."));
		instructions.push_back(string("s - use this frame"));
		instructions.push_back(string("q - done"));
	}
	void
		set_pick_points_instructions(){
		instructions.clear();
		instructions.push_back(string("Pick Points"));
		instructions.push_back(string("q - done"));
	}
	void
		set_connectivity_instructions(){
		instructions.clear();
		instructions.push_back(string("Pick Connections"));
		instructions.push_back(string("q - done"));
	}
	void
		set_symmetry_instructions(){
		instructions.clear();
		instructions.push_back(string("Pick Symmetric Points"));
		instructions.push_back(string("q - done"));
	}
	void
		set_move_points_instructions(){
		instructions.clear();
		instructions.push_back(string("Move Points"));
		instructions.push_back(string("p - next image"));
		instructions.push_back(string("o - previous image"));
		instructions.push_back(string("q - done"));
	}
	void
		initialise_symmetry(const int index){
		int n = data.points[index].size(); data.symmetry.resize(n);
		for (int i = 0; i < n; i++)data.symmetry[i] = i;
	}
	void
		replicate_annotations(const int index){
		if ((index < 0) || (index >= int(data.points.size())))return;
		for (int i = 0; i < int(data.points.size()); i++){
			if (i == index)continue;
			data.points[i] = data.points[index];
		}
	}
	int
		find_closest_point(const Point2f p,
		const double thresh = 10.0){
		int n = data.points[idx].size(), imin = -1; double dmin = -1;
		for (int i = 0; i < n; i++){
			double d = norm(p - data.points[idx][i]);
			if ((imin < 0) || (d < dmin)){ imin = i; dmin = d; }
		}
		if ((dmin >= 0) && (dmin < thresh))return imin; else return -1;
	}
	int
		remove_triangle(const Point2f p)
	{
		Mat1f points(data.points[0].size(), 2);
		Mat1f points_d(data.points[0].size(), 2);
		int n = data.indices.size();
		for (int i = 0; i < data.points[0].size(); i++)
		{
			points[i][0] = data.points[idx][i].x;
			points[i][1] = data.points[idx][i].y;
			points_d[i][0] = data.points[idx_dst][i].x;
			points_d[i][1] = data.points[idx_dst][i].y;
		}
		
		for (int i = 0; i <  n / 3; i++)
		{
			int idx0, idx1, idx2;
			idx0 = data.indices[i * 3];
			idx1 = data.indices[i * 3 + 1];
			idx2 = data.indices[i * 3 + 2];
			
			Point2f tri[3] = { Point2f(points[idx0][0], points[idx0][1]),
				Point2f(points[idx1][0], points[idx1][1]),
				Point2f(points[idx2][0], points[idx2][1]) };
			Point2f dst[3] = { Point2f(points_d[idx0][0], points_d[idx0][1]),
				Point2f(points_d[idx1][0], points_d[idx1][1]),
				Point2f(points_d[idx2][0], points_d[idx2][1]) };

			if (in_triangle(p, tri))
			{
				data.indices.erase(data.indices.begin() + i * 3, data.indices.begin() + i * 3 + 3);

				Mat I = imread(annotation.data.imnames[idx]);
				restoreTextureFromTriangle(I, tri, annotation.image);
				n -= 3;
				i -= 3;
			}
		}
		
		return 0;
	}
	bool in_triangle(const Point2f p, const Point2f *triangle)
	{
		Point2f v0 = triangle[2] - triangle[0];
		Point2f v1 = triangle[1] - triangle[0];
		Point2f v2 = p - triangle[0];

		float dot00 = v0.dot(v0);
		float dot01 = v0.dot(v1);
		float dot02 = v0.dot(v2);
		float dot11 = v1.dot(v1);
		float dot12 = v1.dot(v2);

		float inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

		float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
		if (u < 0 || u > 1) // if u out of range, return directly
		{
			return false;
		}

		float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;
		if (v < 0 || v > 1) // if v out of range, return directly
		{
			return false;
		}

		return u + v <= 1;
	}
protected:
	void
		draw_strings(cv::Mat img,
		const vector<string> &text){
		for (int i = 0; i < int(text.size()); i++)this->draw_string(img, text[i], i + 1);
	}
	void
		draw_string(cv::Mat img,
		const string text,
		const int level)
	{
		Size size = getTextSize(text, FONT_HERSHEY_COMPLEX, 0.6f, 1, NULL);
		putText(img, text, Point(0, level*size.height), FONT_HERSHEY_COMPLEX, 0.6f,
			Scalar::all(0), 1, CV_AA);
		putText(img, text, Point(1, level*size.height + 1), FONT_HERSHEY_COMPLEX, 0.6f,
			Scalar::all(255), 1, CV_AA);
	}
}annotation;

//==============================================================================
static void pp_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == CV_EVENT_LBUTTONDOWN){
		annotation.data.points[0].push_back(Point2f(x, y));
		annotation.draw_points(); imshow(annotation.wname, annotation.image);
	}
}
//==============================================================================
void pc_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == CV_EVENT_LBUTTONDOWN){
		int imin = annotation.find_closest_point(Point2f(x, y));
		if (imin >= 0){ //add connection
			int m = annotation.data.connections.size();
			if (m == 0)annotation.data.connections.push_back(Vec2i(imin, -1));
			else{
				if (annotation.data.connections[m - 1][1] < 0)//1st connecting point chosen
					annotation.data.connections[m - 1][1] = imin;
				else annotation.data.connections.push_back(Vec2i(imin, -1));
			}
			annotation.draw_connections();
			imshow(annotation.wname, annotation.image);
		}
	}
}
//==============================================================================
void ps_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == CV_EVENT_LBUTTONDOWN){
		int imin = annotation.find_closest_point(Point2f(x, y));
		if (imin >= 0){
			if (annotation.pidx < 0)annotation.pidx = imin;
			else{
				annotation.data.symmetry[annotation.pidx] = imin;
				annotation.data.symmetry[imin] = annotation.pidx;
				annotation.pidx = -1;
			}
			annotation.draw_symmetry();
			imshow(annotation.wname, annotation.image);
		}
	}
}
//==============================================================================
void mv_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == CV_EVENT_LBUTTONDOWN){
		if (annotation.pidx < 0){
			annotation.pidx = annotation.find_closest_point(Point2f(x, y));
		}
		else annotation.pidx = -1;
		annotation.copy_clean_image();
		annotation.draw_connections();
		annotation.draw_chosen_point();
		imshow(annotation.wname, annotation.image);
	}
	else if (event == CV_EVENT_MOUSEMOVE){
		if (annotation.pidx >= 0){
			annotation.data.points[annotation.idx][annotation.pidx] = Point2f(x, y);
			annotation.copy_clean_image();
			annotation.draw_connections();
			annotation.draw_chosen_point();
			imshow(annotation.wname, annotation.image);
		}
	}
}
//==============================================================================
void rm_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == CV_EVENT_LBUTTONDOWN){
		annotation.remove_triangle(Point2f(x, y));
		imshow(annotation.wname, annotation.image);
	}
}

//==============================================================================
#endif