#ifndef PREPROCESS_IMAGE_H
#define PREPROCESS_IMAGE_H

#include <opencv2/core/utility.hpp>
class PreProcessImg
{

private:
    cv::Mat &m_original_img;
    cv::Mat &m_light_pattern;
    cv::Mat m_img_no_noise;
    cv::Mat m_img_no_light;
    cv::Mat m_img_threshold;
    cv::Mat m_result;
    char m_light_method;
    char m_segment_method;

    //Segmentation methods
    void ConnectedComponents();
    void ConnectedComponentsStats();
    void FindContoursBasic();


    // Step 1
    void removeNoise();
    
    // step 2
    void removeLight();
    
    // step 3
    void binarize();

    //step 4 
    void segmentation();

public:
    PreProcessImg(cv::Mat &original_img, cv::Mat &light_pattern,char light_method=1,char seg_method=1)
        : m_original_img{original_img}, m_light_pattern{light_pattern},
        m_light_method{light_method},m_segment_method{seg_method}
    {
    }
    PreProcessImg(const PreProcessImg &other) = delete;

    void StartPreProcess();

    cv::Mat getOriginalImg()const{ return m_original_img;}
    cv::Mat getLightPattern()const{ return m_light_pattern;}
    cv::Mat getImgNoNoise()const{ return m_img_no_noise;}
    cv::Mat getImgNoLight()const{ return m_img_no_light;}
    cv::Mat getBinaryImg()const{ return m_img_threshold;}
    cv::Mat getResult()const{ return m_result;}
};

#endif