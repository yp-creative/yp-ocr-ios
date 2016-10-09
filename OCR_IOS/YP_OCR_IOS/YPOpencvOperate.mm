//
//  YPOpencvOperate.m
//  OCR_IDCardTest
//
//  Created by yp-tc-m-2548 on 16/8/2.
//  Copyright © 2016年 yp-tc-m-2548. All rights reserved.
//

#import "YPOpencvOperate.h"

#import <opencv2/opencv.hpp>
#import <opencv2/imgproc/types_c.h>
#import <opencv2/imgcodecs/ios.h>

#define GRAY_THRESH 128
#define HOUGH_VOTE 3
#define HOUGH_STEP 3
using namespace cv;
using namespace std;

@implementation YPOpencvOperate

#pragma mark Objc Method

+ (UIImage *)thresholdWithImage:(UIImage *)source{

    double t = (double)getTickCount();

    Mat srcImg,dst;

    UIImageToMat(source, srcImg);

    Mat resetImg = autoSetAlphaBetaForThresh(srcImg);

    Mat channel = getChannel(resetImg, 2);

    Mat grayImg;

    cvtColor(channel, grayImg, CV_BGR2GRAY);

    dst = threshold(grayImg);

    t = 1000*((double)getTickCount() - t)/getTickFrequency();

    return MatToUIImage(dst);
}

+ (UIImage *)correctSlopingWithImage:(UIImage *)source{
    
    double t = (double)getTickCount();
    
    Mat srcImg,dst;

    UIImageToMat(source, srcImg);
    
    double li[] = {0.0, 0.0, 0.0};
    double hi[] = {0.2, 0.2, 0.2};
    double lo[] = {0.0, 0.0, 0.0};
    double ho[] = {0.1, 0.1, 0.1};
    double ga[] = {2.2, 2.2, 2.2};
    
    vector<double> low_in(li, li+3);
    vector<double> high_in(hi, hi+3);
    vector<double> low_out(lo, lo+3);
    vector<double> high_out(ho, ho+3);
    vector<double> gamma(ga, ga+3);
    
    Mat tem = Mat(srcImg.size(), srcImg.type());
    
    ImageAdjust(srcImg, tem,
                low_in,
                high_in,
                low_out,
                high_out,
                gamma
                );
    
    normalize(tem, tem, 0, 255, NORM_MINMAX);
    
    Mat channel = getChannel(tem, 2);
    
    Mat grayImg;
    
    cvtColor(channel, grayImg, CV_BGR2GRAY);
    
    Mat fin;
    threshold(grayImg, fin, 35, 255, THRESH_BINARY);

    dst = correctSloping(fin,srcImg);

    t = 1000*((double)getTickCount() - t)/getTickFrequency();

    return MatToUIImage(dst);
}

#pragma mark C/C++ Method
void setPixel(Mat &srcImg, int rows, int cols, int channel ,uchar value){
    
    int channels = srcImg.channels();
    
    uchar *p = srcImg.ptr<uchar>(rows);
    
    p[cols * channels+channel] = value;
}

uchar getPixel(Mat &srcImg, int rows, int cols, int channel){

    int channels = srcImg.channels();

    uchar *p = srcImg.ptr<uchar>(rows);

    return p[cols * channels+channel];
}

Mat getChannel(Mat &srcImg,int channel){

    CV_Assert(channel < 3 && channel > -1 );

    CV_Assert(srcImg.data);

    vector<Mat> channels;

    Mat splitImg;
    srcImg.copyTo(splitImg);
    split(splitImg, channels);

    vector<vector<Mat>> channelImg(splitImg.channels()) ;

    Mat tem(srcImg.size(),CV_8U,Scalar(0));

    for (int i = 0; i < srcImg.channels(); i++) {
        if (i == 0) {//blue
            channelImg[0].push_back(channels[0]);
        }else{
            channelImg[0].push_back(tem);
        }
        if (i == 1) {//blue
            channelImg[1].push_back(channels[1]);
        }else{
            channelImg[1].push_back(tem);
        }
        if (i == 2) {//red
            channelImg[2].push_back(channels[2]);
        }else{
            channelImg[2].push_back(tem);
        }
    }

    merge(channelImg[channel], splitImg);
    return splitImg;
}

Mat resetAlphaAndBeta(Mat image, double alpha , int beta){
    
    Mat new_image = Mat::zeros( image.size(), image.type() );
    
    for( int y = 0; y < image.rows; y++ )
    {
        for( int x = 0; x < image.cols; x++ )
        {
            for( int c = 0; c < image.channels(); c++ )
            {
                uchar value = saturate_cast<uchar>(alpha * getPixel(image, y, x, c) + beta);
                setPixel(new_image, y, x, c, value);
            }
        }
    }
    return new_image;
}

Mat threshold(Mat grayImg){
    
    CV_Assert(grayImg.channels() == 1);
    
    Mat dst;
    
    double thresh = threshold(grayImg, dst, 0, 255, THRESH_OTSU);
    
    threshold(grayImg, dst, thresh, 255, THRESH_BINARY);
    
    return dst;
}
Mat IteratorMat(Mat srcImg, uchar *table){
    
    int nRows = srcImg.rows;
    int nCols = srcImg.cols;
    
    if (srcImg.isContinuous()) {
        nCols = nCols*nRows;
        nRows = 1;
    }
    uchar *p;
    switch (srcImg.channels()) {
        case 1:
        {
            for (int rows = 0; rows < nRows; rows++) {
                p = srcImg.ptr<uchar>(rows);
                for (int cols = 0; cols < nCols; cols++) {
                    p[cols] = table[p[cols]];
                }
            }
        }
            break;
        case 3:
        {
            for (int rows = 0; rows < nRows; rows++) {
                p = srcImg.ptr<uchar>(rows);
                for (int cols = 0; cols < nCols; cols++) {
                    p[cols * 3 + 0] = table[p[cols* 3 + 0]];
                    p[cols * 3 + 1] = table[p[cols* 3 + 1]];
                    p[cols * 3 + 2] = table[p[cols* 3 + 2]];
                }
            }
            
        }
            break;
    }
    
    return srcImg;
}

void ImageAdjust(Mat& src, Mat& dst,
                 vector<double> low_in,
                 vector<double> high_in,
                 vector<double> low_out,
                 vector<double> high_out,
                 vector<double> gamma)
{
    vector<double> low;
    vector<double> high;
    vector<double> bottom;
    vector<double> top;
    vector<double> err_in;
    vector<double> err_out;
    size_t N = low_in.size();
    
    for (int i=0; i<N; i++)
    {
        low.push_back(low_in[i]*255);
        high.push_back(high_in[i]*255);
        bottom.push_back(low_out[i]*255);
        top.push_back(high_out[i]*255);
        err_in.push_back(high[i] - low[i]);
        err_out.push_back(top[i] - bottom[i]);
    }
    
    int x,y;
    vector<double> val;
    
    // intensity transform
    for( y = 0; y < src.rows; y++)
    {
        for (x = 0; x < src.cols; x++)
        {
            for (int i=0; i<N; i++)
            {
                uchar val = getPixel(src, y, x, i);
                val = saturate_cast<uchar>(pow((val-low[i])/err_in[i], gamma[i])*err_out[i]+bottom[i]);
                setPixel(dst, y, x, i, val);
            }
        }
    }
    
}

Mat gammaCorrection(Mat I, float fGamma)
{
    CV_Assert(I.data);
    
    // accept only char type matrices
    CV_Assert(I.depth() != sizeof(uchar));
    
    // build look up table
    uchar lut[256];
    for( int i = 0; i < 256; i++ )
    {
        lut[i] = pow((float)(i/255.0), fGamma) * 255.0;
    }
    
    IteratorMat(I, lut);
    
    return I;
}

Mat correction(Mat srcImg,double approachThresh){
    
    CV_Assert(srcImg.data);
    CV_Assert(approachThresh > 0);
    
    Mat resetAB ;
    normalize(srcImg, resetAB, 0, 255, NORM_MINMAX);
    
    resetAB = srcImg;
    
    Mat grayImg;
    cvtColor(resetAB, grayImg, CV_BGR2GRAY);
    
    Mat tmp;
    double firstThresh = threshold(grayImg, tmp, 0, 255, THRESH_OTSU);
    printf("oriThresh:%f\n",firstThresh);
    
    IplImage firstSrcIpl = grayImg;
    
    float firstAvg = cvAvg(&firstSrcIpl).val[0];
    
    printf("\noriAvg:%f\n",firstAvg);
    
    double gamma = approachThresh > firstAvg ? 1/2.2 : 2.2;
    
    resetAB = gammaCorrection(resetAB, gamma);
    
    return resetAB;
}

Mat autoSetAlphaBetaForThresh(Mat srcImg,double approachThresh = 128 ){
    
    Mat resetAB = correction(srcImg,approachThresh);
    
    Mat grayImg;
    cvtColor(resetAB, grayImg, CV_BGR2GRAY);
    
    Mat tmp;
    double thresh = threshold(grayImg, tmp, 0, 255, THRESH_OTSU);
    
    IplImage srcIpl = grayImg;
    
    float avg = cvAvg(&srcIpl).val[0];
    
    double max;
    double min;
    
    cvMinMaxLoc(&srcIpl, &min, &max);
    
    double b = 0.12*(max-min+1)/128;
    
    double start = (1 - b) * thresh;
    
    double end = (1 + b) * thresh;
    
    double med = (start+end)/2;
    
    Mat dst;
    
    double alpha = med > approachThresh ? avg/approachThresh : approachThresh/avg;
    
    int beta = 0;
    
    dst = resetAlphaAndBeta(resetAB,alpha,beta);
    
    return resetAB;
}

Mat correctSloping(Mat grayImg,Mat colorImg){
    
    cv::Point center(grayImg.cols/2, grayImg.rows/2);
    
    Mat padded;
    int opWidth = getOptimalDFTSize(grayImg.rows);
    int opHeight = getOptimalDFTSize(grayImg.cols);
    copyMakeBorder(grayImg, padded, 0, opWidth-grayImg.rows, 0, opHeight-grayImg.cols, BORDER_CONSTANT, Scalar::all(0));
    
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat comImg;
    
    merge(planes,2,comImg);
    
    dft(comImg, comImg);
    
    split(comImg, planes);
    magnitude(planes[0], planes[1], planes[0]);
    
    Mat magMat = planes[0];
    magMat += Scalar::all(1);
    log(magMat, magMat);
    
    magMat = magMat(cv::Rect(0, 0, magMat.cols & -2, magMat.rows & -2));
    
    int cx = magMat.cols/2;
    int cy = magMat.rows/2;
    
    Mat q0(magMat, cv::Rect(0, 0, cx, cy));
    Mat q1(magMat, cv::Rect(0, cy, cx, cy));
    Mat q2(magMat, cv::Rect(cx, cy, cx, cy));
    Mat q3(magMat, cv::Rect(cx, 0, cx, cy));
    
    Mat tmp;
    q0.copyTo(tmp);
    q2.copyTo(q0);
    tmp.copyTo(q2);
    
    q1.copyTo(tmp);
    q3.copyTo(q1);
    tmp.copyTo(q3);
    
    normalize(magMat, magMat, 0, 1, CV_MINMAX);
    Mat magImg(magMat.size(), CV_8UC1);
    magMat.convertTo(magImg,CV_8UC1,255,0);
    
    int elementSize = 1;
    
    threshold(magImg,magImg,GRAY_THRESH,255,CV_THRESH_BINARY);
    Mat element = getStructuringElement( MORPH_RECT ,cv::Size( 2*elementSize + 1, 2*elementSize+1 ));
    
    erode(magImg, magImg, element);
    dilate(magImg, magImg,element);
    erode(magImg, magImg, element);
    
    dilate(magImg, magImg,element);
    dilate(magImg, magImg,element);
    
    vector<Vec2f> lines;
    float pi180 = (float)CV_PI/45;
    HoughLines(magImg,lines,1,pi180,HOUGH_VOTE,0,0);
    
    int count = 0;
    
    int lastCount =  0;
    
    while (lines.size() > 3) {
        count++;
        const int threshC = count*HOUGH_STEP;
        HoughLines(magImg,lines,1,pi180,threshC,0,0);
        
        if (lines.size() != 0) {
            lastCount = count;
        }else{
            const int lastThresh = lastCount*HOUGH_STEP;
            HoughLines(magImg,lines,1,pi180,lastThresh,0,0);
            break;
        }
    }
    
    size_t numLines = lines.size();
    for(int l=0; l<numLines; l++)
    {
        float rho = lines[l][0], theta = lines[l][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line(magImg,pt1,pt2,Scalar(255,0,0),3,8,0);
    }
    
    float angel = 0;
    float piThresh = (float)CV_PI/90;
    float pi2 = CV_PI/2;
    for(int l=0; l<numLines; l++)
    {
        float theta = lines[l][1];
        if(abs(theta) < piThresh || abs(theta-pi2) < piThresh)
            continue;
        else{
            if (abs(theta)>pi2) {
                 angel = (abs(theta-pi2) > abs(angel)) ?  theta  : angel;
            }else{
                angel = (abs(theta) > abs(angel)) ?  theta  : angel;
            }
        }
    }
    
    angel = angel < pi2 ? angel : angel - CV_PI;
    
    printf("%f",angel);
    
    if(angel != pi2 ){
        float angelT = grayImg.rows * tan(angel) / grayImg.cols;
        angel = atan(angelT);
    }
    float angelD = angel * 180 / (float)CV_PI;
    
    Mat dstImg;
    Mat rotMat;
    
    if ( angelD <  70 && abs(angel) < CV_PI/2) {
        rotMat = getRotationMatrix2D(center,angelD,1.0);
    }else{
        angelD = angelD - 90;
        rotMat = getRotationMatrix2D(center,angelD,1.0);
    }
    
    dstImg = Mat::ones(grayImg.size(),CV_8UC3);
    warpAffine(colorImg,dstImg,rotMat,grayImg.size(),1,0,Scalar(255,255,255));
    
    return dstImg;
}

/********************************************************************************
 *函数描述：  DefRto 计算并返回一幅图像的清晰度
 *函数参数： frame  彩色帧图
 *函数返回值：double   清晰度表示值，针对该视频，当清晰度小于10为模糊，大于14为清楚
 *********************************************************************************/
double DefRto(Mat frame)
{
    IplImage img = IplImage(frame);
    double temp = 0;
    double DR = 0;
    int i,j;//循环变量
    int height=img.height;
    int width=img.width;
    int step=img.widthStep/sizeof(uchar);
    uchar *data=(uchar*)img.imageData;
    double num = width*height;
    
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            temp += sqrt((pow((double)(data[(i+1)*step+j]-data[i*step+j]),2) + pow((double)(data[i*step+j+1]-data[i*step+j]),2)));
            temp += abs(data[(i+1)*step+j]-data[i*step+j])+abs(data[i*step+j+1]-data[i*step+j]);
        }
    }
    DR = temp/num;
    return DR;
}

+ (BOOL)isBlurryWithImage:(UIImage *)srcImage thresh:(double)thresh{
    Mat frame;
    UIImageToMat(srcImage, frame);
    double rt =  DefRto(frame);
    return rt > thresh ? YES : NO;
}

/********************************************************************************************
 *函数描述：  calcCast    计算并返回一幅图像的色偏度以及，色偏方向
 *函数参数：  InputImg    需要计算的图片，BGR存放格式，彩色（3通道），灰度图无效
 *           cast        计算出的偏差值，小于1表示比较正常，大于1表示存在色偏
 *           da          红/绿色偏估计值，da大于0，表示偏红；da小于0表示偏绿
 *           db          黄/蓝色偏估计值，db大于0，表示偏黄；db小于0表示偏蓝
 *函数返回值： 返回值通过cast、da、db三个应用返回，无显式返回值
 *********************************************************************************************/
void calcCast(Mat InputImg,float& cast,float& da,float& db)
{
    Mat LABimg;
    cvtColor(InputImg,LABimg,CV_BGR2Lab);//由于OpenCV定义的格式是uint8，这里输出的LABimg从标准的0～100，-127～127，-127～127，被映射到了0～255，0～255，0～255空间
    float a=0,b=0;
    int HistA[256],HistB[256];
    for(int i=0;i<256;i++)
    {
        HistA[i]=0;
        HistB[i]=0;
    }
    for(int i=0;i<LABimg.rows;i++)
    {
        for(int j=0;j<LABimg.cols;j++)
        {
            a+=float(LABimg.at<cv::Vec3b>(i,j)[1]-128);//在计算过程中，要考虑将CIE L*a*b*空间还原 后同
            b+=float(LABimg.at<cv::Vec3b>(i,j)[2]-128);
            int x=LABimg.at<cv::Vec3b>(i,j)[1];
            int y=LABimg.at<cv::Vec3b>(i,j)[2];
            HistA[x]++;
            HistB[y]++;
        }
    }
    da=a/float(LABimg.rows*LABimg.cols);
    db=b/float(LABimg.rows*LABimg.cols);
    float D =sqrt(da*da+db*db);
    float Ma=0,Mb=0;
    for(int i=0;i<256;i++)
    {
        Ma+=abs(i-128-da)*HistA[i];//计算范围-128～127
        Mb+=abs(i-128-db)*HistB[i];
    }
    Ma/=float((LABimg.rows*LABimg.cols));
    Mb/=float((LABimg.rows*LABimg.cols));
    float M=sqrt(Ma*Ma+Mb*Mb);
    float K=D/M;
    cast = K;
    return;
}

/*********************************************************************************************************************************************************
 *函数描述：  brightnessException     计算并返回一幅图像的色偏度以及，色偏方向
 *函数参数：  InputImg    需要计算的图片，BGR存放格式，彩色（3通道），灰度图无效
 *           cast        计算出的偏差值，小于1表示比较正常，大于1表示存在亮度异常；当cast异常时，da大于0表示过亮，da小于0表示过暗
 *函数返回值： 返回值通过cast、da两个引用返回，无显式返回值
 **********************************************************************************************************************************************************/
void brightness(Mat InputImg,float& cast,float& da)
{
    float a=0;
    int Hist[256];
    
    for(int i=0;i<256;i++)
        Hist[i]=0;

    for(int i=0;i<InputImg.rows;i++)
    {
        for(int j=0;j<InputImg.cols;j++)
        {
            a+=float(InputImg.at<uchar>(i,j) - 128);//在计算过程中，考虑128为亮度均值点
            int x=InputImg.at<uchar>(i,j);
            Hist[x]++;
        }
    }

    da=a/float(InputImg.rows*InputImg.cols);
    float D = abs(da);
    float Ma = 0;

    for(int i = 0;i < 256;i++)
    {
        Ma+=abs(i - 128 - da)*Hist[i];
    }

    Ma /= float((InputImg.rows*InputImg.cols));
    float M = abs(Ma);
    float K = D/M;
    cast = K;
    return;
}

+ (YPBrightnessType)isBrightnessWithImage:(UIImage *)image{

    Mat src;

    UIImageToMat(image, src);

    float cast , da ;

    brightness(src, cast, da);
    
    if (cast < 1) {
        return YPBrightnessTypeSuitable;
    }else{
        if (da > 0) {
            return YPBrightnessTypehigh;
        }else{
            return YPBrightnessTypelow;
        }
    }
}

+ (void)idCardMathDetector:(UIImage *)imageScene imgObject:(UIImage *)imageObject{

    Mat imgScene, imgObject;

    UIImageToMat(imageScene, imgScene);

    UIImageToMat(imageObject, imgObject);

    cv::Point *points = idCardMathWithImage(imgObject ,imgScene);
    
    for (int i = 0; i < sizeof(points); i++) {
        NSLog(@"%d:%d-%d",i,points->x,points->y);
    }
}

cv::Point* idCardMathWithImage(Mat &imgObject,Mat &imgScene) {

    ///-- Step 1: 使用SIFT算子检测特征点
    Ptr<FeatureDetector> detector = ORB::create();
    vector<KeyPoint> keyPointObject = vector<KeyPoint>();
    vector<KeyPoint> keyPointScene = vector<KeyPoint>();
    detector->detect(imgObject, keyPointObject);
    detector->detect(imgScene, keyPointScene);

    ///-- Step 2: 使用SIFT算子提取特征（计算特征向量）

    Ptr<DescriptorExtractor> extractor = ORB::create();

    Mat descriptorsObject;
    Mat descriptorsScene;

    extractor->compute(imgObject, keyPointObject, descriptorsObject);
    extractor->compute(imgScene, keyPointScene, descriptorsScene);

    ///-- Step 3: 使用BRUTEFORCE法进行匹配
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    vector<vector<DMatch>> matches_knn = vector<vector<DMatch>>();
    vector<DMatch> matche_knn = vector<DMatch>();

    matcher->knnMatch(descriptorsObject, descriptorsScene, matches_knn, 2);

    for (int i = 0 ; i < matches_knn.size(); i++) {

        double ratio = matches_knn[i][0].distance / matches_knn[i][1].distance;

        if (ratio < 0.61) {
            matche_knn.push_back(matches_knn[i][0]);
        }
    }

    vector<cv::Point> objList = vector<cv::Point>();
    vector<cv::Point> sceneList = vector<cv::Point>();

    vector<KeyPoint> keypoints_objectList = keyPointObject;
    vector<KeyPoint> keypoints_sceneList = keyPointScene;

    for (int i = 0; i < matche_knn.size(); i++) {

        cv::Point objectPoint(keypoints_objectList[matche_knn[i].queryIdx].pt.x,keypoints_objectList[matche_knn[i].queryIdx].pt.y);
        cv::Point scenePoint(keypoints_sceneList[matche_knn[i].queryIdx].pt.x,keypoints_sceneList[matche_knn[i].queryIdx].pt.y);

        objList.push_back(objectPoint);
        sceneList.push_back(scenePoint);
    }

    if (objList.size() < 4 || sceneList.size() < 4) {
        throw runtime_error("请确保身份证正面图片质量");
    }

    std::vector<cv::Point2f> obj(matche_knn.size());
    std::vector<cv::Point2f> scene(matche_knn.size());

    for (size_t i = 0; i < matche_knn.size(); i++)
    {
        obj[i] = keypoints_objectList[matche_knn[i].queryIdx].pt;
        scene[i] = keypoints_sceneList[matche_knn[i].trainIdx].pt;
    }

    Mat H = findHomography(obj, scene, 8, 10);

    vector<Point2f> obj_corners(4);
    vector<Point2f> scene_corners(4);

    Mat obj_corner(4 ,1 ,CV_32FC2);
    Mat scene_corner(4 ,1 ,CV_32FC2);

    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( imgObject.cols, 0 );
    obj_corners[2] = cvPoint( imgObject.cols, imgObject.rows );
    obj_corners[3] = cvPoint( 0, imgObject.rows );

    perspectiveTransform( obj_corners, scene_corners, H);

    cv::Point *points = new cv::Point[4];
    double v1 = scene_corners[0].x - scene_corners[1].x;
    double v2 = scene_corners[0].x - scene_corners[1].y;

    if (pow(v1, 2) + pow(v2, 2) < 20) {//认为是处理不好的,不处理
        double width = imgObject.cols;
        double height = imgObject.rows;
        cv::Point *newPoint = new cv::Point[4];
        newPoint[0] = cv::Point(0, 0);
        newPoint[1] = cv::Point(width, 0);
        newPoint[2] = cv::Point(width, height);
        newPoint[3] = cv::Point(0, height);
    }

    for (int i = 0; i < 4; i++) {
        points[i] = cv::Point(scene_corners[i].x, scene_corners[i].y);
    }

    return points;
}

@end
