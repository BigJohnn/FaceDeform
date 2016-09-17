#ifndef _ROBUSTMATCHER_H_H
#define _ROBUSTMATCHER_H_H
#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d.hpp>

class RobustMatcher
{
private:
	cv::Ptr<cv::FeatureDetector>     detector;
	cv::Ptr<cv::DescriptorExtractor> extractor;
	int normType;
	float ratio;
	bool refineF;
	double distance;
	double confidence;
	cv::Mat m_roughHomography;
public:
	vector<Point> poly_points;
	CvPoint pts[76];

public:
	RobustMatcher()
		:normType(cv::NORM_L2), ratio(0.8f),
		refineF(true), confidence(0.98), distance(3.0){
	
		detector = cv::xfeatures2d::SIFT::create();
		extractor = cv::xfeatures2d::SIFT::create();
	}

	cv::Mat match(cv::Mat &image1, cv::Mat& image2, std::vector<cv::DMatch>& matches,
		std::vector<cv::KeyPoint>& keypoints1,
		std::vector<cv::KeyPoint>& keypoints2)
	{
		cv::Mat descriptor1, descriptor2;
		extractor->compute(image1, keypoints1, descriptor1);
		extractor->compute(image2, keypoints2, descriptor2);

		cv::BFMatcher matcher(normType, true);

		std::vector<cv::DMatch> outputMatches;
		matcher.match(descriptor1, descriptor2, outputMatches);

		cv::Mat fundamental = ransactTest(outputMatches,
			keypoints1, keypoints2, matches);

		warpTexturefromFM(image1, image2);

		//warpAffine(image1, image2, fundamental, cv::Size(480, 640));
		showAndSave("ransac", getMatchesImage(image1, image2, keypoints1, keypoints2, matches, 100));
		
		return fundamental;
	}

private:
	void warpTexturefromFM(Mat &originalImage, Mat &warp_final){
		Mat warp_dst, warp_mask;

		warp_dst = Mat::zeros(originalImage.rows, originalImage.cols, originalImage.type());
		warp_mask = Mat::zeros(originalImage.rows, originalImage.cols, originalImage.type());

		int left = 0;
		int top =0;
		int width = originalImage.cols - left;
		int height = originalImage.rows - top;
		/// Get the Affine Transform
		
		/// Apply the Affine Transform just found to the src image
		Rect roi(left, top, width, height);
		//Rect roi(140, 210, 150, 200);
		Mat originalImageRoi = originalImage(roi);
		
		Mat warp_dstRoi = warp_dst(roi);
		//!�����õ�Ӧ��������ǻ������󣡣���������
		warpPerspective(originalImageRoi, warp_dstRoi, m_roughHomography, warp_dstRoi.size());
		
		Mat mask(640, 480, CV_8U);
		for (int j = 0; j < mask.rows; j++)
		{
			uchar* data = mask.ptr<uchar>(j);
			for (int i = 0; i < mask.cols; i++)
			{
				data[i] = 0;
			}
		}
		
		for (int i = 0; i < poly_points.size() / 3; i++)
		{
			vector<Point> pt(poly_points.begin() + 3 * i, poly_points.begin() + i * 3 + 3);
			fillConvexPoly(mask, pt, cv::Scalar(255, 255, 255), 8, 0);
		}

		warp_dst.copyTo(warp_final,mask);
	}

	cv::Mat ransactTest(std::vector<cv::DMatch>& matches,
		std::vector<cv::KeyPoint>& keypoints1,
		std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::DMatch>& outMatches)
	{
		std::vector<cv::Point2f> points1, points2;
		for (std::vector<cv::DMatch>::const_iterator it =
			matches.begin(); it != matches.end(); ++it)
		{
			points1.push_back(keypoints1[it->queryIdx].pt);
			points2.push_back(keypoints2[it->trainIdx].pt);
		}
		std::vector<uchar> inliers(points1.size(), 0);
		cv::Mat fundamental = cv::findFundamentalMat(points1, points2, inliers,
			cv::FM_RANSAC,
			distance,
			confidence);

		//��ȡ�����ƥ����
		std::vector<uchar>::const_iterator itIn = inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM = matches.begin();

		for (; itIn != inliers.end(); ++itIn, ++itM)
		{
			if (*itIn)//��Ч
			{
				outMatches.push_back(*itM);
			}
		}

		if (refineF)
		{
			points1.clear();
			points2.clear();

			for (std::vector<cv::DMatch>::
				const_iterator it = outMatches.begin();
				it != outMatches.end(); ++it)
			{
				points1.push_back(keypoints1[it->queryIdx].pt);
				points2.push_back(keypoints2[it->trainIdx].pt);
			}

			fundamental = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
		}

		cv::correctMatches(fundamental, points1, points2, points1, points2);
		std::vector<unsigned char> inliersMask(points1.size());
		float reprojectionThreshold = 3;
		bool homographyFound =refineMatchesWithHomography(points1,
			points2,
			reprojectionThreshold,
			outMatches,
			m_roughHomography);

		return fundamental;
	}

	bool refineMatchesWithHomography(const std::vector<cv::Point2f> srcPoints,
		const std::vector<cv::Point2f> dstPoints,
		float reprojectionThreshold,
		std::vector<cv::DMatch>& matches,
		cv::Mat& homography)
	{
		const int minNumberMatchesAllowed = 8;//������8��ƥ�䡣

		if (matches.size() < minNumberMatchesAllowed)
			return false;

		// Find homography matrix and get inliers mask
		std::vector<unsigned char> inliersMask(srcPoints.size());
		homography = cv::findHomography(srcPoints, //Ѱ��ƥ���ϵĹؼ���ĵ�Ӧ�Ա任��������ͼ��ĵ�Ӧ�Ծ�������һ��3*3�ľ��󣬵�����ĵ�Ӧ�Ծ����3D�ؽ��еĵ�Ӧ�Ծ���͸�Ӿ���3*4���ǲ�һ���ġ�֮ǰһֱ���������ߵ�����
			dstPoints, //ǰ������������ͼ��ƥ��ĵ�
			CV_FM_RANSAC, //���㵥Ӧ�Ծ���ʹ�÷���
			reprojectionThreshold, //��������ͶӰ������ʹ��RANSACʱ���С�
			inliersMask);//ָ��ƥ���ֵ�ǲ�����Ⱥֵ�������Ż�ƥ����������һ����������С��ƥ��������ͬ����ÿһ��ƥ���ж��ǲ�����Ⱥֵ�������Ԫ�ؾ�Ϊ0.

		std::vector<cv::DMatch> inliers;
		for (size_t i = 0; i<inliersMask.size(); i++)
		{
			if (inliersMask[i])
				inliers.push_back(matches[i]);
		}

		matches.swap(inliers);
		return matches.size() > minNumberMatchesAllowed;//���а�������ƥ�䣬��ô��Ϊ�ҵ���Щƥ��ĵ�Ӧ�Ծ���
	}

	inline cv::Mat getMatchesImage(cv::Mat query, cv::Mat pattern, const std::vector<cv::KeyPoint>& queryKp,
		const std::vector<cv::KeyPoint>& trainKp, std::vector<cv::DMatch> matches, int maxMatchesDrawn)
	{
		cv::Mat outImg;

		if (matches.size() > maxMatchesDrawn)
		{
			matches.resize(maxMatchesDrawn);
		}

		//��������ͼ�񣬻���Ѱ�ҵ��������ؼ��㼰��ƥ��
		cv::drawMatches(query,
			queryKp,
			pattern,
			trainKp,
			matches,
			outImg,
			cv::Scalar(0, 200, 0, 255),
			cv::Scalar::all(-1),
			std::vector<char>(),
			cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);//������ûƥ�䵽�Ľǵ㲻���ơ�

		return outImg;
	}

	inline void showAndSave(std::string name, const cv::Mat& m)
	{
		cv::imshow(name, m);
		cv::imwrite(name + ".png", m);
	}
};

#endif