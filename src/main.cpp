#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <memory>

// OpenCV includes
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "MultipleImageWindow.hpp"
#include "PreProcessImage.hpp"

/**
 *  lightMethod:   0: Difference operation is applied
 *                 1: The division operation is applied
 * 
 *   segMethod:    1: the connected components method for segment is applied
 *                 2: the connected components method with the statistics area is applied
 *                 3: the find contours method is applied for Segmentation
 */
const char *keys =
    {
        "{help h usage ? | | print this message}"
        "{@image || Image to process}"
        "{@lightPattern || Image light pattern to apply to image input}"
        "{lightMethod | 1 | Method to remove background light, 0 difference, 1 div, 2 no light removal }"
        "{segMethod | 1 | Method to segment: 1 connected Components, 2 connected components with stats, 3 find Contours }"};

/**
 * This basic function applies a blur to an input image by using a big kernel size relative to the image size. 
 * From the code, it is one-third of the original width and height.
 * Estimate the background image in case with don't have a Light pattern
 */
cv::Mat calculateLightPattern(cv::Mat img)
{
    cv::Mat pattern;
    // Basic and effective way to calculate the light pattern from one image
    cv::blur(img, pattern, cv::Size(img.cols / 3, img.cols / 3));
    return pattern;
}

bool processInput(const cv::CommandLineParser &p, cv::Mat &out_img, cv::Mat &out_light_pattern,
                  int &out_method_light, int &out_method_seg)
{

    // if requires hel show
    if (p.has("help"))
    {
        p.printMessage();
        return false;
    }

    // Check if params are correctly parsed in his variables
    if (!p.check())
    {
        p.printErrors();
        return false;
    }

    std::string img_file = p.get<std::string>(0);
    std::string light_pattern_file = p.get<std::string>(1);
    out_method_light = p.get<int>("lightMethod");
    out_method_seg = p.get<int>("segMethod");

    out_img = cv::imread(img_file, 0);
    if (out_img.data == NULL)
    {
        std::cout << "Error Loading the image\n";
        return false;
    }

    out_light_pattern = cv::imread(light_pattern_file, 0);
    if (out_light_pattern.data == NULL)
    {
        //Calculate light pattern
        out_light_pattern = calculateLightPattern(out_img);
    }
    cv::medianBlur(out_light_pattern, out_light_pattern, 3);
    return true;
}

int main(int argc, const char **argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Object Detection");

    cv::Mat img;
    cv::Mat light_pattern;
    int method_light;
    int method_seg;
    if (!processInput(parser, img, light_pattern, method_light, method_seg))
    {
        return -1;
    }
    std::unique_ptr<PreProcessImg> p;
    p = std::make_unique<PreProcessImg>(img, light_pattern, method_light, method_seg);
    p->StartPreProcess();
    std::shared_ptr<MultipleImageWindow> miw;
    // Create the Multiple Image Window
    miw = std::make_shared<MultipleImageWindow>("Main window", 3, 2, cv::WINDOW_AUTOSIZE);
    // Show images
    miw->addImage("Original Image", p->getOriginalImg());
    miw->addImage("Image without noise", p->getImgNoNoise());
    miw->addImage("Light Pattern", p->getLightPattern());
    miw->addImage("No Light", p->getImgNoLight());
    miw->addImage("Threshold", p->getBinaryImg());
    miw->addImage("Result", p->getResult());

    miw->render();
    cv::waitKey(0);

    return 0;
}