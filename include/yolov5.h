//
// Created by tztek on 2022/4/7.
//

#ifndef YOLO_TRT_YOLOV5_H
#define YOLO_TRT_YOLOV5_H


#include <opencv2/opencv.hpp>
#include "loggingRT.h"
#include <map>
#include <iostream>
#include <cassert>
#include <fstream>
#include <NvInfer.h>
#include <dirent.h>
#include <NvOnnxParser.h>
#include <memory>
#include <mutex>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <cuda_runtime.h>
#include "loggingRT.h"


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0

constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}

struct ObjStu
{
    int		 id		= -1;
    float	 prob	= 0.f;
    cv::Rect rect;

};

struct ObjPos
{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct ImgInfo
{
    float width;
    float height;
};

typedef std::vector<ObjStu> detectResult;

class YoloV5
{
public:
    explicit YoloV5(const std::string& configPath);
    ~YoloV5();
    std::vector<detectResult> detect(std::vector<cv::Mat> &batchImg);

private:
    void modifyBoundaryValue(int &x1, int &y1, int &x2, int &y2, int imgWidth, int imgHeight);
    bool checkDetectRect(int &x1, int &y1, int &x2, int &y2, int imgWidth, int imgHeight);
    void bubbleSort(std::vector<float> confs, int length, std::vector<int> &indDiff);
    float getIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);
    void initInputImageSize(std::vector<cv::Mat> &batchImg);
    void imgPreProcess(std::vector<cv::Mat> &batchImg) const;
    void getTrtmodelStream();
    void readParameters(const std::string& configPath);
    nvinfer1::ICudaEngine* createEngine(nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config);
    bool createEngineIfNotExit();
    void createInferenceEngine(nvinfer1::IHostMemory **modelStream);
    void getBindingDimsInfo();
    std::vector<detectResult> postProcessing(float *boxesProb, int batch);
    void doInference(nvinfer1::IExecutionContext& context, float* boxesProb, int batchSize);
    void thresholdFilter(const float *anchorsProb,
                         int batch,
                         std::vector<std::vector<float>> &confFilterVec,
                         std::vector<std::vector<int>> &confIdFilterVec,
                         std::vector<std::vector<ObjPos>> &boxFilterVec,
                         float confThreshold);
    ObjPos xywh2xyxy(float x, float y, float w, float h, float originW, float originH);
    std::vector<detectResult> getDetResult(std::vector<std::vector<float>> &confFilterVec,
                                           std::vector<std::vector<int>> &confIdFilterVec,
                                           std::vector<std::vector<ObjPos>> &boxFilterVec,
                                           std::vector<std::vector<int>> keepVec);
    std::vector<std::vector<int>> allClassNMS(std::vector<std::vector<float>> &confFilterVec,
                                              std::vector<std::vector<int>> &confIdFilterVec,
                                              std::vector<std::vector<ObjPos>> &boxFilterVec, float nmsThreshold);

private:
    bool mAllClasssNMS;
    std::vector<ImgInfo> mImageSizeBatch;
    int mMaxSupportBatchSize;
    size_t mEngineFileSize;
    std::string mQuantizationInfer;
    std::string mOnnxFile;
    std::string mEngineFile;
    Logger mGlogger;
    char *mTrtModelStream;
    int mInputH;
    int mInputW;
    int mInputC;
    int mOutputAnchorsNum;
    int mOutputAnchorsDim;
    int mOutputAnchorsSize;
    unsigned char *mInputData;
    float *mBuff[2];
    float mConfTreshold;
    float mNMSTreshold;
    nvinfer1::IRuntime *mRuntime;
    nvinfer1::ICudaEngine *mEngine;
    nvinfer1::IExecutionContext *mContext;
    cudaStream_t  mStream;
};

typedef std::vector<ObjStu> detectResult;

#endif //YOLO_TRT_YOLOV5_H
