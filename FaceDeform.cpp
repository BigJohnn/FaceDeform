// FaceDeform.cpp : Defines the entry point for the console application.
//
//Copyright @Hou Peihong 2016-6-23
#include "stdafx.h"
#include "tps.h"
#include <fstream>
#include <string>
#include <map>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>
#include "annotation.h"
#include <unordered_map>
#include "RobustMatcher.h"
//CGAL Includes
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_conformer_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CDT::Vertex_handle Vertex_handle;
//CGAL

using namespace cv;
using namespace std;

string filename = "foxMask";
struct point{
	int x, y;
};

//==============================================================================
class muct_data{
public:
	string name;
	string index;
	vector<Point2f> points;

	muct_data(string str,
		string muct_dir){
		size_t p1 = 0, p2;

		//set image directory
		string idir = muct_dir; if (idir[idir.length() - 1] != '/')idir += "/";
		idir += "jpg/";

		//get image name
		p2 = str.find(",");
		if (p2 == string::npos){ cerr << "Invalid MUCT file" << endl; exit(0); }
		name = str.substr(p1, p2 - p1);

		if ((strcmp(name.c_str(), "i434xe-fn") == 0) || //corrupted data
			(name[1] == 'r'))name = "";                //ignore flipped images
		else{
			name = idir + str.substr(p1, p2 - p1) + ".jpg"; p1 = p2 + 1;

			//get index
			p2 = str.find(",", p1);
			if (p2 == string::npos){ cerr << "Invalid MUCT file" << endl; exit(0); }
			index = str.substr(p1, p2 - p1); p1 = p2 + 1;

			//get points
			for (int i = 0; i < 75; i++){
				p2 = str.find(",", p1);
				if (p2 == string::npos){ cerr << "Invalid MUCT file" << endl; exit(0); }
				string x = str.substr(p1, p2 - p1); p1 = p2 + 1;
				p2 = str.find(",", p1);
				if (p2 == string::npos){ cerr << "Invalid MUCT file" << endl; exit(0); }
				string y = str.substr(p1, p2 - p1); p1 = p2 + 1;
				points.push_back(Point2f(atoi(x.c_str()), atoi(y.c_str())));
			}
			p2 = str.find(",", p1);
			if (p2 == string::npos){ cerr << "Invalid MUCT file" << endl; exit(0); }
			string x = str.substr(p1, p2 - p1); p1 = p2 + 1;
			string y = str.substr(p1, str.length() - p1);
			points.push_back(Point2f(atoi(x.c_str()), atoi(y.c_str())));
		}
	}
};

vector<string> imnames;
vector< vector<cv::Point2f> > points_all;
void load_markdata()
{
	string ifile = "muct-landmarks/";
	string lmfile = ifile+"muct76-opencv.csv";
	
	ifstream file(lmfile.c_str());
	if (!file.is_open()){
		cerr << "Failed opening " << lmfile << " for reading!" << endl; return;
	}
	string str; getline(file, str);
	while (!file.eof()){
		getline(file, str); if (str.length() == 0)break;
		muct_data d(str, ifile); if (d.name.length() == 0)continue;
		imnames.push_back(d.name);
		points_all.push_back(d.points);
	}
	file.close();
}

struct lessPoint2f
{
	bool operator()(const Point2f& lhs, const Point2f& rhs) const
	{
		return (lhs.x == rhs.x) ? (lhs.y < rhs.y) : (lhs.x < rhs.x);
	}
};

/// Return the Delaunay triangulation, under the form of an adjacency matrix
/// points is a Nx2 mat containing the coordinates (x, y) of the points
Mat delaunay(const Mat1f& points, int imRows, int imCols, vector<int> &indices, bool issrc=true)
{
	map<Point2f, int, lessPoint2f> mappts;

	Mat1b adj_s(points.rows, points.rows, uchar(0));

	/// Create subdiv and insert the points to it
	Subdiv2D subdiv(Rect(0, 0, imCols, imRows));
	for (int p = 0; p < points.rows; p++)
	{
		float xp = points(p, 0);
		float yp = points(p, 1);
		Point2f fp(xp, yp);

		// Don't add duplicates
		if (mappts.count(fp) == 0)
		{
			// Save point and index
			mappts[fp] = p;

			subdiv.insert(fp);
		}
	}

	/// Get the number of edges
	vector<Vec4f> edgeList;
	subdiv.getEdgeList(edgeList);
	int nE = edgeList.size();

	/// Check adjacency
	for (int i = 0; i < nE; i++)
	{
		Vec4f e = edgeList[i];
		Point2f pt0(e[0], e[1]);
		Point2f pt1(e[2], e[3]);

		if (mappts.count(pt0) == 0 || mappts.count(pt1) == 0) {
			// Not a valid point
			continue;
		}

		int idx0 = mappts[pt0];
		int idx1 = mappts[pt1];

		// Symmetric matrix
		adj_s(idx0, idx1) = 1;
		adj_s(idx1, idx0) = 1;
	}

	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);

	int nT = triangleList.size();
	/// Check triangle
	for (int i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		Point2f pt0(t[0], t[1]);
		Point2f pt1(t[2], t[3]);
		Point2f pt2(t[4], t[5]);
		if (mappts.count(pt0) == 0 || mappts.count(pt1) == 0 || mappts.count(pt2) == 0) {
			triangleList.erase(triangleList.begin() + i, triangleList.begin() + i + 1);
			i--;
			// Not a valid point
			continue;
		}

		int idx0 = mappts[pt0];
		int idx1 = mappts[pt1];
		int idx2 = mappts[pt2];
	
		indices.push_back(idx0);
		indices.push_back(idx1);
		indices.push_back(idx2);
	}

	if (issrc)
		annotation.triangleList = triangleList;

	return adj_s;
}

void restoreTextureFromTriangle(Mat &originalImage, Point2f dstTri[3], Mat &warp_final)
{
	Mat warp_mat(2, 3, CV_32FC1);
	Mat warp_dst, warp_mask;
	CvPoint trianglePoints[3];
	trianglePoints[0] = dstTri[0];
	trianglePoints[1] = dstTri[1];
	trianglePoints[2] = dstTri[2];

	warp_dst = Mat::zeros(originalImage.rows, originalImage.cols, originalImage.type());
	warp_mask = Mat::zeros(originalImage.rows, originalImage.cols, originalImage.type());

	int left = 0;
	int top = 0;
	/// Get the Affine Transform
	for (int i = 0; i<3; i++){
		dstTri[i].x -= left;
		dstTri[i].y -= top;
	}

	//[1 0 0]
	//[0 1 0]
	warp_mat = getAffineTransform(dstTri, dstTri);

	Rect roi(left, top, 280, 350);
	Mat originalImageRoi = originalImage(roi);
	Mat warp_dstRoi = warp_dst(roi);
	warpAffine(originalImageRoi, warp_dstRoi, warp_mat, warp_dstRoi.size());
	cvFillConvexPoly(new IplImage(warp_mask), trianglePoints, 3, CV_RGB(255, 255, 255), CV_AA, 0);
	warp_dst.copyTo(warp_final, warp_mask);
}
void warpTextureFromTriangle(Point2f srcTri[3], Mat &originalImage, Point2f dstTri[3], Mat &warp_final){
	//int t, ellap;
	//t= clock();
	Mat warp_mat(2, 3, CV_32FC1);
	Mat warp_dst, warp_mask;
	CvPoint trianglePoints[3];
	trianglePoints[0] = dstTri[0];
	trianglePoints[1] = dstTri[1];
	trianglePoints[2] = dstTri[2];

	//vector<Point2f> trianglePT;
	//trianglePT.push_back(srcTri[0]);
	//trianglePT.push_back(srcTri[1]);
	//trianglePT.push_back(srcTri[2]);

	warp_dst = Mat::zeros(originalImage.rows, originalImage.cols, originalImage.type());
	warp_mask = Mat::zeros(originalImage.rows, originalImage.cols, originalImage.type());

	int left = 0;
	int top = 0;
	int width = originalImage.cols - left;
	int height = originalImage.rows - top;
	/// Get the Affine Transform
	for (int i = 0; i<3; i++){
		srcTri[i].x -= left;
		srcTri[i].y -= top;
		dstTri[i].x -= left;
		dstTri[i].y -= top;
	}
	
	warp_mat = getAffineTransform(srcTri, dstTri);

	/// Apply the Affine Transform just found to the src image
	
	Rect roi(left, top, width, height);
	Mat originalImageRoi = originalImage(roi);
	Mat warp_dstRoi = warp_dst(roi);
	warpAffine(originalImageRoi, warp_dstRoi, warp_mat, warp_dstRoi.size());
	//fillConvexPoly(warp_mask, trianglePT, CV_RGB(255, 255, 255));
	cvFillConvexPoly(new IplImage(warp_mask), trianglePoints, 3, CV_RGB(255, 255, 255), CV_AA, 0);
	warp_dst.copyTo(warp_final, warp_mask);
}

int loadIndices(vector<int> &indices)
{
	FileStorage fs2("test.yml", FileStorage::READ);

	// second method: use FileNode::operator >>
	
	FileNode features;// = fs2["triangleList"];
	FileNodeIterator it = features.begin(), it_end = features.end();
	int idx = 0;

	features = fs2["Indices"];
	it = features.begin(), it_end = features.end();
	idx = 0;

	// iterate through a sequence using FileNodeIterator
	for (; it != it_end; ++it, idx++)
	{
		indices.push_back((int)(*it));
	}

	fs2.release();

	if (indices.size() == 0) return -1;
	return 0;
}

//load points coordinates
int load(vector<cv::Point2f> &points)
{
	cv::Point2f pt;
	FileStorage fs2(filename + "_vertices.yml", FileStorage::READ);

	// second method: use FileNode::operator >>

	FileNode features;// = fs2["triangleList"];
	FileNodeIterator it = features.begin(), it_end = features.end();
	int idx = 0;

	features = fs2["feaCoord"];
	it = features.begin(), it_end = features.end();
	idx = 0;

	// iterate through a sequence using FileNodeIterator
	for (; it != it_end; ++it, idx++)
	{
		pt.x = (float)(*it++);
		pt.y = (float)(*it);
		points.push_back(pt);
	}

	fs2.release();

	if (points.size() == 0) return -1;
	return 0;
}

void DrawConformingTiangulation(cv::Mat img_src, cv::Mat img_dst, int pointsID_src, int pointsID_dst)
{
	Mat1f points_s(points_all[0].size(), 2);
	Mat1f points_d(points_all[0].size(), 2);

	for (int i = 0; i < points_all[0].size(); i++)
	{
		points_s[i][0] = points_all[pointsID_src][i].x;
		points_s[i][1] = points_all[pointsID_src][i].y;
		points_d[i][0] = points_all[pointsID_dst][i].x;
		points_d[i][1] = points_all[pointsID_dst][i].y;
	}

	annotation.data.imnames = imnames;
	annotation.data.points = points_all;

	namedWindow(annotation.wname, 0);//这里要先创建窗口！！！！！！
	setMouseCallback(annotation.wname, rm_MouseCallback, 0);

	//annotation.set_pick_points_instructions();
	annotation.set_current_image(pointsID_src);
	annotation.set_dst_image(pointsID_dst);
	//annotation.draw_instructions();
	annotation.idx = pointsID_src;
	annotation.idx_dst = pointsID_dst;

	vector<Vec6f> triangles, triangles_dst;
	vector<int> indices;
	Mat1b adj_s, adj_d;
	int rows = annotation.image.rows;
	int cols = annotation.image.cols;


	//Mat3b img(rows, cols, Vec3b(0, 0, 0));
	cout << points_s << endl;
	cout << points_d << endl;
	vector<int> triSrc;
	adj_s = delaunay(points_s, rows, cols, triSrc);//NxN邻接关系表
	adj_d = delaunay(points_d, rows, cols, triSrc);
	CDT cdt,cdt_d;
	vector<Vertex_handle> vss,vsd;
	for (int i = 0; i < points_all[0].size(); i++)
	{
		Vertex_handle va = cdt.insert(CDT::Point(points_s[i][0], points_s[i][1]));
		vss.push_back(va);
		Vertex_handle va_d = cdt_d.insert(CDT::Point(points_d[i][0], points_d[i][1]));
		vsd.push_back(va_d);
	}
	
	for (int i = 0; i < points_s.rows; i++)
	{
		/// Draw the edges
		for (int j = i + 1; j < points_s.rows; j++)
		{
			if (adj_s(i, j))
			{
				cdt.insert_constraint(vss[i], vss[j]);
			}
			if (adj_d(i, j))
			{
				cdt_d.insert_constraint(vsd[i], vsd[j]);
			}
		}
	}

	// make it conforming Delaunay
	CGAL::make_conforming_Delaunay_2(cdt);
	CGAL::make_conforming_Delaunay_2(cdt_d);
	std::cout << "Number of vertices after make_conforming_Delaunay_2: "
		<< cdt.number_of_vertices() <<  std::endl;
	std::cout << "Number of vertices after make_conforming_Delaunay_2: "
		<< cdt_d.number_of_vertices() << std::endl;
	// then make it conforming Gabriel
	CGAL::make_conforming_Gabriel_2(cdt);
	CGAL::make_conforming_Gabriel_2(cdt_d);
	std::cout << "Number of vertices after make_conforming_Gabriel_2: "
		<< cdt.number_of_vertices() << std::endl;
	std::cout << "Number of vertices after make_conforming_Gabriel_2: "
		<< cdt_d.number_of_vertices() << std::endl;

	points_s.resize(3*cdt.number_of_vertices());
	points_d.resize(3*cdt.number_of_vertices());
	//用Finite_faces_iterator遍历CDT中的所有三角面Face，输出每个三角面对应的三个顶点数据  
	CDT::Finite_faces_iterator f_iter;
	int k = 0;
	for (f_iter = cdt.finite_faces_begin(); f_iter != cdt.finite_faces_end() && k < points_s.rows; f_iter++)
	{
		for (int i = 0; i<3; i++)
		{
			CDT::Point p = f_iter->vertex(i)->point();
			points_s[k][0] = p.x();
			points_s[k++][1] = p.y();
			//cout << "(" << p.x() << "," << p.y() << ")" << endl;
		}
	}
	k = 0;
	for (f_iter = cdt.finite_faces_begin(); f_iter != cdt.finite_faces_end() && k < points_s.rows; f_iter++)
	{
		for (int i = 0; i<3; i++)
		{
			CDT::Point p = f_iter->vertex(i)->point();
			points_d[k][0] = p.x();
			points_d[k++][1] = p.y();
			//cout << "(" << p.x() << "," << p.y() << ")" << endl;
		}
	}
	
	annotation.data.indices = triSrc;
	//cv::imshow("1", annotation.image);
	//cv::imshow("2", img_dst);
	if (loadIndices(indices) == 0)
	{
		for (int i = 0; i < indices.size() / 3; i++)
		{
			int idx0 = indices[i * 3];
			int idx1 = indices[i * 3 + 1];
			int idx2 = indices[i * 3 + 2];
			Point2f src[3] = { Point2f(points_s[idx0][0], points_s[idx0][1]),
				Point2f(points_s[idx1][0], points_s[idx1][1]),
				Point2f(points_s[idx2][0], points_s[idx2][1]) };
			Point2f dst[3] = { Point2f(points_d[idx0][0], points_d[idx0][1]),
				Point2f(points_d[idx1][0], points_d[idx1][1]),
				Point2f(points_d[idx2][0], points_d[idx2][1]) };

			warpTextureFromTriangle(dst, img_dst, src, annotation.image);
		}
	}
	else
	{
		//这里 虽然点是一一对应的，但是生成的delaunay三角形并不是一一对应的！！ 参Delaunay剖分定义
		//考虑只生成一次剖分！ 索引共用。
		for (int i = 0; i < k/3;i++)
		{
			int idx0 = i * 3 + 1;
			int idx1 = i * 3 + 1;
			int idx2 = i * 3 + 2;
			Point2f src[3] = { Point2f(points_s[idx0][0], points_s[idx0][1]),
				Point2f(points_s[idx1][0], points_s[idx1][1]),
				Point2f(points_s[idx2][0], points_s[idx2][1]) };
			Point2f dst[3] = { Point2f(points_d[idx0][0], points_d[idx0][1]),
				Point2f(points_d[idx1][0], points_d[idx1][1]),
				Point2f(points_d[idx2][0], points_d[idx2][1]) };

			//warpTextureFromTriangle(dst, img_dst, src, annotation.image);
		}
		k = 0;
		for (f_iter = cdt.finite_faces_begin(); f_iter != cdt.finite_faces_end() && k < points_s.rows; f_iter++)
		{
			for (int i = 0; i<3; i++)
			{
				CDT::Point p = f_iter->vertex(i)->point();
				circle(annotation.image, Point(p.x(), p.y()), 1, Scalar(0, 0, 255), -1);
				//cout << "(" << p.x() << "," << p.y() << ")" << endl;
			}
		}
		//for (int i = 0; i < points_s.rows; i++)
		//{
		//	int xi = points_s.at<float>(i, 0);
		//	int yi = points_s.at<float>(i, 1);

		//	int xi_d = points_d.at<float>(i, 0);
		//	int yi_d = points_d.at<float>(i, 1);

		//	/// Draw the edges
		//	for (int j = i + 1; j < points_s.rows; j++)
		//	{
		//		if (adj_s(i, j))
		//		{
		//			int xj = points_s(j, 0);
		//			int yj = points_s(j, 1);
		//			line(annotation.image, Point(xi, yi), Point(xj, yj), Scalar(255, 0, 0), 1);
		//		}
		//		/*if (adj_d(i, j))
		//		{
		//		int xj = points_d(j, 0);
		//		int yj = points_d(j, 1);
		//		line(img_src, Point(xi_d, yi_d), Point(xj, yj), Scalar(255, 0, 0), 1);
		//		}*/
		//	}
		//}

		//for (int i = 0; i < points_s.rows; i++)
		//{
		//	int xi = points_s(i, 0);
		//	int yi = points_s(i, 1);

		//	/* Draw the nodes*/
		//	circle(annotation.image, Point(xi, yi), 1, Scalar(0, 0, 255), -1);
		//}
	}

	for (;;){
		imshow(annotation.wname, annotation.image);
		int c = waitKey(0);
		if (c == 'q')break;
		else if (c == 's')//保存结果
		{
			//save trianglelists here...................
			FileStorage fs("test.yml", FileStorage::WRITE);

			fs << "Indices" << "[:";
			for (int i = 0; i < annotation.data.indices.size(); i++)
			{
				fs << annotation.data.indices[i];
			}
			fs << "]";

			fs.release();

		}
	}
}

void DrawDelauney(cv::Mat img_src, cv::Mat img_dst, int pointsID_src, int pointsID_dst)
{
	Mat1f points_s(points_all[0].size(), 2);
	Mat1f points_d(points_all[0].size(), 2);

	for (int i = 0; i < points_all[0].size(); i++)
	{
		points_s[i][0] = points_all[pointsID_src][i].x;
		points_s[i][1] = points_all[pointsID_src][i].y;
		points_d[i][0] = points_all[pointsID_dst][i].x;
		points_d[i][1] = points_all[pointsID_dst][i].y;
	}

	annotation.data.imnames = imnames;
	annotation.data.points = points_all;
	
	namedWindow(annotation.wname, 0);//这里要先创建窗口！！！！！！
	setMouseCallback(annotation.wname, rm_MouseCallback, 0);

	//annotation.set_pick_points_instructions();
	annotation.set_current_image(pointsID_src);
	annotation.set_dst_image(pointsID_dst);
	//annotation.draw_instructions();
	annotation.idx = pointsID_src;
	annotation.idx_dst = pointsID_dst;

	vector<Vec6f> triangles, triangles_dst;
	vector<int> indices;
	Mat1b adj_s,adj_d;
	int rows = annotation.image.rows;
	int cols = annotation.image.cols;

	//Mat3b img(rows, cols, Vec3b(0, 0, 0));
	cout << points_s << endl;
	cout << points_d << endl;
	vector<int> triSrc;
	adj_s = delaunay(points_s, rows, cols, triSrc);//NxN邻接关系表

	annotation.data.indices = triSrc;
	//cv::imshow("1", annotation.image);
	//cv::imshow("2", img_dst);
	if (loadIndices(indices) == 0)
	{
		for (int i = 0; i < indices.size() / 3; i++)
		{
			int idx0 = indices[i * 3];
			int idx1 = indices[i * 3 + 1];
			int idx2 = indices[i * 3 + 2];
			Point2f src[3] = { Point2f(points_s[idx0][0], points_s[idx0][1]),
				Point2f(points_s[idx1][0], points_s[idx1][1]),
				Point2f(points_s[idx2][0], points_s[idx2][1]) };
			Point2f dst[3] = { Point2f(points_d[idx0][0], points_d[idx0][1]),
				Point2f(points_d[idx1][0], points_d[idx1][1]),
				Point2f(points_d[idx2][0], points_d[idx2][1]) };

			warpTextureFromTriangle(dst, img_dst, src, annotation.image);
		}
	}
	else
	{
		//这里 虽然点是一一对应的，但是生成的delaunay三角形并不是一一对应的！！ 参Delaunay剖分定义
		//考虑只生成一次剖分！ 索引共用。
		for (int i = 0; i < triSrc.size() / 3; i++)
		{
			int idx0 = triSrc[i * 3];
			int idx1 = triSrc[i * 3 + 1];
			int idx2 = triSrc[i * 3 + 2];
			Point2f src[3] = { Point2f(points_s[idx0][0], points_s[idx0][1]),
				Point2f(points_s[idx1][0], points_s[idx1][1]),
				Point2f(points_s[idx2][0], points_s[idx2][1]) };
			Point2f dst[3] = { Point2f(points_d[idx0][0], points_d[idx0][1]),
				Point2f(points_d[idx1][0], points_d[idx1][1]),
				Point2f(points_d[idx2][0], points_d[idx2][1]) };

			warpTextureFromTriangle(dst, img_dst, src, annotation.image);
		}
		for (int i = 0; i < points_s.rows; i++)
		{
			int xi = points_s.at<float>(i, 0);
			int yi = points_s.at<float>(i, 1);

			int xi_d = points_d.at<float>(i, 0);
			int yi_d = points_d.at<float>(i, 1);

			/// Draw the edges
			for (int j = i + 1; j < points_s.rows; j++)
			{
				if (adj_s(i, j))
				{
					int xj = points_s(j, 0);
					int yj = points_s(j, 1);
					line(annotation.image, Point(xi, yi), Point(xj, yj), Scalar(255, 0, 0), 1);
				}
				/*if (adj_d(i, j))
				{
				int xj = points_d(j, 0);
				int yj = points_d(j, 1);
				line(img_src, Point(xi_d, yi_d), Point(xj, yj), Scalar(255, 0, 0), 1);
				}*/
			}
		}

		for (int i = 0; i < points_s.rows; i++)
		{
			int xi = points_s(i, 0);
			int yi = points_s(i, 1);

			/* Draw the nodes*/
			circle(annotation.image, Point(xi, yi), 1, Scalar(0, 0, 255), -1);
		}
	}
	
	for (;;){
		imshow(annotation.wname, annotation.image);
		int c = waitKey(0);
		if (c == 'q')break;
		else if (c == 's')//保存结果
		{
			//save trianglelists here...................
			FileStorage fs("test.yml", FileStorage::WRITE);

			fs << "Indices" << "[:";
			for (int i = 0; i < annotation.data.indices.size(); i++)
			{
				fs << annotation.data.indices[i];
			}
			fs << "]";

			fs.release();

		}
	}
}

//draw points
void draw(cv::Mat img, vector<Point2f> points)
{
	Mat1f points_s(points.size(), 2);

	for (int i = 0; i < points.size(); i++)
	{
		points_s[i][0] = points[i].x;
		points_s[i][1] = points[i].y;
	}
	//annotation.data.imnames = imnames;
	annotation.wname = filename.c_str();
	annotation.data.imnames.push_back(filename + ".png");
	annotation.data.points.push_back(points);
	annotation.image = img;
	namedWindow(annotation.wname, WINDOW_KEEPRATIO);//这里要先创建窗口！！！！！！
	setMouseCallback(annotation.wname, rm_MouseCallback, 0);

	annotation.idx_dst = 0; 
	vector<int> indices;
	Mat1b adj_s, adj_d;
	int rows = img.rows;
	int cols = img.cols;

	vector<int> triSrc;
	adj_s = delaunay(points_s, rows, cols, triSrc);//NxN邻接关系表

	annotation.data.indices = triSrc;
	//cv::imshow("1", annotation.image);
	//cv::imshow("2", img_dst);
	if (loadIndices(indices) == 0)
	{
		for (int i = 0; i < indices.size() / 3; i++)
		{
			int idx0 = indices[i * 3];
			int idx1 = indices[i * 3 + 1];
			int idx2 = indices[i * 3 + 2];
			Point2f src[3] = { Point2f(points_s[idx0][0], points_s[idx0][1]),
				Point2f(points_s[idx1][0], points_s[idx1][1]),
				Point2f(points_s[idx2][0], points_s[idx2][1]) };
			
			restoreTextureFromTriangle(img, src, img);
			//warpTextureFromTriangle(dst, img_dst, src, annotation.image);
		}
	}
	else
	{
		for (int i = 0; i < points_s.rows; i++)
		{
			int xi = points_s.at<float>(i, 0);
			int yi = points_s.at<float>(i, 1);

			/// Draw the edges
			for (int j = i + 1; j < points_s.rows; j++)
			{
				if (adj_s(i, j))
				{
					int xj = points_s(j, 0);
					int yj = points_s(j, 1);
					line(img, Point(xi, yi), Point(xj, yj), Scalar(255, 0, 0), 1);
				}
				/*if (adj_d(i, j))
				{
				int xj = points_d(j, 0);
				int yj = points_d(j, 1);
				line(img_src, Point(xi_d, yi_d), Point(xj, yj), Scalar(255, 0, 0), 1);
				}*/
			}
		}

		for (int i = 0; i < points_s.rows; i++)
		{
			int xi = points_s(i, 0);
			int yi = points_s(i, 1);

			/* Draw the nodes*/
			circle(img, Point(xi, yi), 1, Scalar(0, 0, 255), -1);
		}
	}

	for (;;){
		imshow(annotation.wname, img);
		int c = waitKey(0);
		if (c == 'q')break;
		else if (c == 's')//保存结果
		{
			//save trianglelists here...................
			FileStorage fs(filename + "_indices.yml", FileStorage::WRITE);

			fs << "Indices" << "[:";
			for (int i = 0; i < annotation.data.indices.size(); i++)
			{
				fs << annotation.data.indices[i];
			}
			fs << "]";

			fs.release();

		}
	}
}

void drawRansac(cv::Mat img_src, cv::Mat img_dst, int pointsID_src, int pointsID_dst)
{
	RobustMatcher matcher;
	vector<KeyPoint> points1, points2;
	vector<DMatch> matches;
	for (int i = 0; i < points_all[0].size(); i++)
	{
		double ratio = i*1.0 / points_all[0].size();
		points1.push_back(KeyPoint(points_all[pointsID_src][i], 10*abs(ratio), -1.0 + 2 * ratio));
		points2.push_back(KeyPoint(points_all[pointsID_dst][i], 10*abs(ratio), -1.0 + 2 * ratio));
		matcher.poly_points.push_back(points_all[pointsID_dst][i]);
		matcher.pts[i] = points_all[pointsID_dst][i];
	}

	vector<int> triSrc;

	Mat1f points_s(matcher.poly_points.size(), 2);

	for (int i = 0; i < matcher.poly_points.size(); i++)
	{
		points_s[i][0] = matcher.poly_points[i].x;
		points_s[i][1] = matcher.poly_points[i].y;
	}
	matcher.poly_points.clear();
	delaunay(points_s, img_src.rows, img_src.cols, triSrc);//NxN邻接关系表

	for (int i = 0; i < triSrc.size(); i++)
	{
		int idx = triSrc[i];
		matcher.poly_points.push_back(Point(points_s[idx][0], points_s[idx][1]));
	}

	cv::Mat fundamental = matcher.match(img_src, img_dst, matches, points1, points2);

	cout << fundamental << endl;
	waitKey();
}

int _tmain(int argc, _TCHAR* argv[])
{
#ifdef Create
	vector<cv::Point2f> points;
	load(points);

	cv::Mat img = imread(filename + ".png");
	draw(img, points);
#else
	load_markdata();

	int left, right;
	left = 108;
	right = 61;

	Mat img = imread(imnames[left]);
	Mat img2 = imread(imnames[right]);

	#ifdef Ransac
		drawRansac(img, img2, left, right);
	#elif defined Conform_flag
		DrawConformingTiangulation(img, img2, left, right);
	#else
		DrawDelauney(img, img2, left, right);
	#endif
#endif
	
	return 0;
}

