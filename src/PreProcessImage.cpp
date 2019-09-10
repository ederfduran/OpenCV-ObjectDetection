#include "PreProcessImage.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

cv::Scalar _randomColor(cv::RNG &rng)
{
    unsigned icolor = (unsigned)rng;
    return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void PreProcessImg::removeNoise(){
    cv::medianBlur(m_original_img, m_img_no_noise, 3);
}

void PreProcessImg::removeLight(){
    if(m_light_method!=2){
        // if method is normalization
    if (m_light_method == 1)
    {
        // Require change our image to 32 float for division
        cv::Mat img32, pattern32;
        m_img_no_noise.convertTo(img32, CV_32F);
        m_light_pattern.convertTo(pattern32, CV_32F);

        // Divide the image by the pattern
        m_img_no_light = 1 - (img32 / pattern32);
        // convert 8 bit format and scale
        m_img_no_light.convertTo(m_img_no_light, CV_8U, 255);
    }
    else
    {
        m_img_no_light = m_light_pattern - m_img_no_noise;
    }
    }
}

void PreProcessImg::binarize(){
    if (m_light_method != 2)
    {
        cv::threshold(m_img_no_light, m_img_threshold, 30, 255, cv::THRESH_BINARY);
    }
    else
    {
        cv::threshold(m_img_no_light, m_img_threshold, 140, 255, cv::THRESH_BINARY_INV);
    }
}

void PreProcessImg::ConnectedComponents(){
    // Use connected components to divide our image in multiple connected component objects
    cv::Mat labels;
    int num_objects = connectedComponents(m_img_threshold, labels);
    // Check the number of objects detected The background counts as an object
    if (num_objects < 2)
    {
        std::cout << "No objects detected" << std::endl;
        return;
    }
    else
    {
        std::cout << "Number of objects detected: " << num_objects - 1 << std::endl;
    }
    // Create output image coloring the objects
    m_result= cv::Mat::zeros(m_img_threshold.rows, m_img_threshold.cols, CV_8UC3);
    cv::RNG rng(0xFFFFFFFF);
    for (auto i = 1; i < num_objects; i++)
    {
        cv::Mat mask = labels == i;
        m_result.setTo(_randomColor(rng), mask);
    }
}

void PreProcessImg::ConnectedComponentsStats(){
    // use connected components with stats
    cv::Mat labels, stats, centroids;
    int num_objects = cv::connectedComponentsWithStats(m_img_threshold, labels, stats, centroids);
    // check the number of objects detected
    if (num_objects < 2)
    {
        std::cout << "No objects detected\n";
        return;
    }
    else
    {
        std::cout << "Number of objects detected: " << num_objects - 1 << std::endl;
    }
    // Create output image coloring the objects and show area
    m_result = cv::Mat::zeros(m_img_threshold.rows, m_img_threshold.cols, CV_8UC3);
    cv::RNG rng(0xFFFFFFFF);
    for (auto i = 1; i < num_objects; i++)
    {
        std::cout << "Object " << i << " with pos: " << centroids.at<cv::Point2d>(i) << " with area " << stats.at<int>(i, cv::CC_STAT_AREA) << std::endl;
        cv::Mat mask = labels == i;
        m_result.setTo(_randomColor(rng), mask);
        // draw text with area
        std::stringstream ss;
        ss << "area: " << stats.at<int>(i, cv::CC_STAT_AREA);

        cv::putText(m_result,
                ss.str(),
                centroids.at<cv::Point2d>(i),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                cv::Scalar(255, 255, 255));
    }
}

void PreProcessImg::FindContoursBasic(){
    std::vector<std::vector<cv::Point>> contours;
    /**
     * img: Input image
     * contours: A contour's output where each detected contour is a vector of points
     * cv::RETR_EXTERNAL: retrieves only the external contours.
     * cv::CHAIN_APPROX_SIMPLE: compresses all horizontal, vertical, and diagonal segments, storing only the start and end points
     */
    cv::findContours(m_img_threshold, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    m_result = cv::Mat::zeros(m_img_threshold.rows, m_img_threshold.cols, CV_8UC3);
    // Check the number of objects detected
    if (contours.size() == 0)
    {
        std::cout << "No objects detected\n";
        return;
    }else
    {
        std::cout << "Number of objects detected: " << contours.size() << std::endl;
    }
    cv::RNG rng(0xFFFFFFFF);
    for (auto i = 0; i < contours.size(); i++)
    {
        /**
         *  result: Image to draw the contours
         *  contours: a vector of contours
         *  i: A number indicating the contour to draw. If this is negative, all contours are drawn.
         *  color: The color to draw the contour.
         */
        cv::drawContours(m_result, contours, i, _randomColor(rng));
    }
}

void PreProcessImg::segmentation(){
    switch(m_segment_method){
		case 1:
			ConnectedComponents();
			break;
		case 2:
			ConnectedComponentsStats();
			break;
		case 3:
			FindContoursBasic();
			break;
	}
}

void PreProcessImg::StartPreProcess(){
    removeNoise();
    removeLight();
    binarize();
    segmentation();
}