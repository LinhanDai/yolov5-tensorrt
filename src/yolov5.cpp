//
// Created by tztek on 2022/4/7.
//

#include "yolov5.h"


extern "C" void cudaPreProcess(float* img_dst, unsigned char* img_source, int width, int height, int channel, int num, cudaStream_t stream);

YoloV5::YoloV5(const std::string& configPath)
{
    readParameters(configPath);
    bool flag = createEngineIfNotExit();
    assert(flag == true && "engine create failure!");
    getTrtmodelStream();
}

YoloV5::~YoloV5()
{
    try
    {
        cudaStreamDestroy(mStream);
        cudaFree(mInputData);
        cudaFree(mBuff[0]);
        cudaFree(mBuff[1]);
        cudaFree(mBuff[2]);
        mContext->destroy();
        mEngine->destroy();
    }
    catch (std::exception &e)
    {
        mGlogger.log(Severity::kERROR, "~YoloV4() error!");
    }
}

void YoloV5::imgPreProcess(std::vector<cv::Mat> &batchImg) const
{
    for (int i = 0; i < batchImg.size(); i++)
    {
        cv::Mat &img = batchImg[i];
        float ratioW = mInputW / mImageSizeBatch[i].width;
        float ratioH = mInputH / mImageSizeBatch[i].height;
        int tw, th, tx1, tx2, ty1, ty2;

        if (ratioH > ratioW)
        {
            tw = mInputW;
            th = int(ratioW * mImageSizeBatch[i].height);
            tx1 = 0;
            tx2 = 0;
            ty1 = int((mInputH - th) / 2);
            ty2 = mInputH - th - ty1;
        }
        else
        {
            tw = int (ratioH * mImageSizeBatch[i].width);
            th = mInputH;
            tx1 = int((mInputW - tw) / 2);
            tx2 = mInputW - tw - tx1;
            ty1 = 0;
            ty2 = 0;
        }
        cv::resize(img, img, cv::Size(tw, th), cv::INTER_LINEAR);
        cv::copyMakeBorder(img, img, ty1, ty2, tx1, tx2, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
    }
}

void YoloV5::doInference(nvinfer1::IExecutionContext& context, float* boxesProb, int batchSize)
{
    const nvinfer1::ICudaEngine &engine = context.getEngine();
    assert(engine.getNbBindings() == 2), "yolov5 Bindings Dim should be four!";
    nvinfer1::Dims inputDims = engine.getBindingDimensions(0);
    nvinfer1::Dims d = inputDims;
    d.d[0] = batchSize;
    if (!mContext->setBindingDimensions(0, d))
    {
        mGlogger.log(Severity::kERROR, "模型输入维度不正确");
        std::abort();
    }
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    context.enqueueV2((void **)mBuff, mStream, nullptr);
    CHECK(cudaMemcpyAsync(boxesProb, mBuff[1], batchSize * mOutputAnchorsSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);
}

ObjPos YoloV5::xywh2xyxy(float x, float y, float w, float h, float originW, float originH)
{
    ObjPos pos{};
    float ratioW = mInputW / originW;
    float ratioH = mInputH / originH;
    if (ratioH > ratioW)
    {
        pos.x1 = (x - w / 2) / ratioW;
        pos.x2 = (x + w / 2) / ratioW;
        pos.y1 = (y - h / 2 - (mInputH - ratioW * originH) / 2) / ratioW;
        pos.y2 = (y + h / 2 - (mInputH - ratioW * originH) / 2) / ratioW;

    }
    else
    {
        pos.x1 = (x - w / 2 - (mInputW - ratioH * originW) / 2) / ratioH;
        pos.x2 = (x + w / 2 - (mInputW - ratioH * originW) / 2) / ratioH ;
        pos.y1 = (y - h / 2) / ratioH;
        pos.y2 = (y + h / 2) / ratioH;
    }
    return pos;
}

void YoloV5::thresholdFilter(const float *anchorsProb, int batch, std::vector<std::vector<float>> &confFilterVec,
                             std::vector<std::vector<int>> &confIdFilterVec,
                             std::vector<std::vector<ObjPos>> &boxFilterVec,
                             float confThreshold)
{
    for (int i = 0; i < batch; i++)
    {
        std::vector<float> maxConf;
        std::vector<int> maxSoreId;
        std::vector<ObjPos> obj;
        for (int j = 0; j < mOutputAnchorsNum; j ++)
        {
            std::vector<float> confVec;
            std::vector<float> score;
            float conf = anchorsProb[i * mOutputAnchorsSize + j * mOutputAnchorsDim + 4];
            if (conf > confThreshold)
            {
                for (int k = 5; k < mOutputAnchorsDim; k++)
                {
                    confVec.push_back(anchorsProb[i * mOutputAnchorsSize + j * mOutputAnchorsDim + k] * conf);
                }
                float value = *max_element(confVec.begin(),confVec.end());
                auto valueIter = max_element(confVec.begin(), confVec.end());
                int index = distance(confVec.begin(), valueIter);
                maxSoreId.push_back(index);
                maxConf.push_back(value);
                ObjPos pos = xywh2xyxy(anchorsProb[i * mOutputAnchorsSize + j * mOutputAnchorsDim],
                                       anchorsProb[i * mOutputAnchorsSize + j * mOutputAnchorsDim + 1],
                                       anchorsProb[i * mOutputAnchorsSize + j * mOutputAnchorsDim + 2],
                                       anchorsProb[i * mOutputAnchorsSize + j * mOutputAnchorsDim + 3],
                                       mImageSizeBatch[i].width, mImageSizeBatch[i].height);
                obj.push_back(pos);
            }
        }
        confFilterVec.push_back(maxConf);
        confIdFilterVec.push_back(maxSoreId);
        boxFilterVec.push_back(obj);
    }
}

void YoloV5::modifyBoundaryValue(int &x1, int &y1, int &x2, int &y2, int imgWidth, int imgHeight)
{
    if (x1 < 0)
    {
        x1 = 0;
    }
    else if (x1 > imgWidth)
    {
        x1 = imgWidth;
    }
    if (x2 < 0)
    {
        x2 = 0;
    }
    else if (x2 > imgWidth)
    {
        x2 = imgWidth;
    }
    if (y1 < 0)
    {
        y1 = 0;
    }
    else if (y1 > imgHeight)
    {
        y1 = imgHeight;
    }
    if (y2 < 0)
    {
        y2 = 0;
    }
    else if (y2 > imgHeight)
    {
        y2 = imgHeight;
    }
}

bool YoloV5::checkDetectRect(int &x1, int &y1, int &x2, int &y2, int imgWidth, int imgHeight)
{
    modifyBoundaryValue(x1, y1, x2, y2, imgWidth, imgHeight);
    if ((x2 - x1 > 0) && (y2 - y1 > 0))
        return true;
    else
        return false;
}

std::vector<detectResult> YoloV5::getDetResult(std::vector<std::vector<float>> &confFilterVec,
                                               std::vector<std::vector<int>> &confIdFilterVec,
                                               std::vector<std::vector<ObjPos>> &boxFilterVec,
                                               std::vector<std::vector<int>> keepVec)
{
    std::vector<detectResult> result;
    int batch = boxFilterVec.size();
    for (int i = 0; i < batch; i++)
    {
        detectResult det;
        for (int j = 0; j < keepVec[i].size(); ++j)
        {
            ObjStu obj{};
            int x1 = boxFilterVec[i][keepVec[i][j]].x1;
            int y1 = boxFilterVec[i][keepVec[i][j]].y1;
            int x2 = boxFilterVec[i][keepVec[i][j]].x2;
            int y2 = boxFilterVec[i][keepVec[i][j]].y2;
            if(checkDetectRect(x1, y1, x2, y2, mImageSizeBatch[i].width, mImageSizeBatch[i].height))
            {
                obj.rect.x = x1;
                obj.rect.y = y1;
                obj.rect.width = x2 - x1;
                obj.rect.height = y2 - y1;
                obj.prob = confFilterVec[i][keepVec[i][j]];
                obj.id = confIdFilterVec[i][keepVec[i][j]];
                det.push_back(obj);
            }
        }
        result.push_back(det);
    }
    return result;
}

// Computes IOU between two bounding boxes
float YoloV5::getIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < DBL_EPSILON)
        return 0;

    return in / un;
}


void YoloV5::bubbleSort(std::vector<float> confs, int length, std::vector<int> &indDiff)
{
    for (int m = 0; m < length; m++)
    {
        indDiff[m] = m;
    }
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length - i - 1; j++)
        {
            if (confs[j] < confs[j + 1])
            {
                float temp = confs[j];
                confs[j] = confs[j + 1];
                confs[j + 1] = temp;
                int ind_temp = indDiff[j];
                indDiff[j] = indDiff[j + 1];
                indDiff[j + 1] = ind_temp;
            }
        }
    }
}

std::vector<std::vector<int>> YoloV5::allClassNMS(std::vector<std::vector<float>> &confFilterVec,
                                                  std::vector<std::vector<int>> &confIdFilterVec,
                                                  std::vector<std::vector<ObjPos>> &boxFilterVec,
                                                  float nmsThreshold)
{
    int batch = boxFilterVec.size();
    std::vector<std::vector<int>> keepVec;
    for (int i = 0; i < batch; i ++)
    {
        std::vector<int> keep;
        std::vector<float> confs = confFilterVec[i];
        std::vector<int> ids = confIdFilterVec[i];
        std::vector<ObjPos> boxes = boxFilterVec[i];
        int targetNum = boxes.size();
        std::vector<int> indDiff(targetNum, 0);
        bubbleSort(confs, targetNum, indDiff);
        while (!indDiff.empty())
        {
            int idxSelf = indDiff[0];
            keep.push_back(idxSelf);
            std::vector<float> iouVec;
            for (int j = 1; j < indDiff.size(); j++)
            {
                float iou = getIOU(cv::Rect_<float>(boxes[idxSelf].x1, boxes[idxSelf].y1,
                                                    boxes[idxSelf].x2 - boxes[idxSelf].x1,
                                                    boxes[idxSelf].y2 - boxes[idxSelf].y1),
                                   cv::Rect_<float>(boxes[indDiff[j]].x1, boxes[indDiff[j]].y1,
                                                    boxes[indDiff[j]].x2 - boxes[indDiff[j]].x1,
                                                    boxes[indDiff[j]].y2 - boxes[indDiff[j]].y1));
                iouVec.push_back(iou);
            }
            std::vector<int> newIndex;
            for (int j = 0; j < iouVec.size(); j++)
            {
                if (iouVec[j] < nmsThreshold)
                {
                    newIndex.push_back(indDiff[1 + j]);
                }
            }
            indDiff = newIndex;
        }
        keepVec.push_back(keep);
    }
    return keepVec;
}

std::vector<detectResult> YoloV5::postProcessing(float *anchorsProb, int batch)
{
    std::vector<std::vector<int>> keepVec;
    std::vector<std::vector<float>> confFilterVec;
    std::vector<std::vector<int>> confIdFilterVec;
    std::vector<std::vector<ObjPos>> boxFilterVec;
    thresholdFilter(anchorsProb, batch, confFilterVec, confIdFilterVec, boxFilterVec, mConfTreshold);
    if (mAllClasssNMS)
    {
        keepVec = allClassNMS(confFilterVec, confIdFilterVec, boxFilterVec, mNMSTreshold);
    }
    std::vector<detectResult> result = getDetResult(confFilterVec, confIdFilterVec, boxFilterVec, keepVec);
    return result;
}

void YoloV5::initInputImageSize(std::vector<cv::Mat> &batchImg)
{
    int batch = batchImg.size();
    for (int i = 0; i < batch; i++)
    {
        ImgInfo info{};
        info.width = batchImg[i].cols;
        info.height = batchImg[i].rows;
        mImageSizeBatch.push_back(info);
    }
}

std::vector<detectResult> YoloV5::detect(std::vector<cv::Mat> &batchImg)
{
    std::vector<cv::Mat> detectMatVec;
    for (auto & img : batchImg)
    {
        detectMatVec.push_back(img);
    }
    int batch = detectMatVec.size();
    initInputImageSize(detectMatVec);
    imgPreProcess(detectMatVec);
    int inputSingleByteNum = mInputH * mInputW * mInputC;
    for (size_t i = 0; i < batch; i++)
    {
        cudaMemcpyAsync(mInputData + i * inputSingleByteNum, detectMatVec[i].data, inputSingleByteNum,
                        cudaMemcpyHostToDevice, mStream);
    }
    cudaPreProcess(mBuff[0], mInputData, mInputW, mInputH, mInputC, batch, mStream);
    float *anchorsProb = (float *) malloc(batch * mOutputAnchorsSize * sizeof(float));
    doInference(*mContext, anchorsProb, batch);
    std::vector<detectResult> detectResult = postProcessing(anchorsProb, batch);
    free(anchorsProb);
    mImageSizeBatch.clear();
    return detectResult;
}

void YoloV5::readParameters(const std::string& configPath)
{
    std::string yamlFile = configPath + "/" + "yolo.yaml";
    cv::FileStorage fs(yamlFile, cv::FileStorage::READ);
    mConfTreshold = fs["confTreshold"];
    mNMSTreshold = fs["nmsTreshold"];
    fs["allClassNMS"] >> mAllClasssNMS;
    mMaxSupportBatchSize = fs["maxSupportBatchSize"];
    mQuantizationInfer = (std::string) fs["quantizationInfer"];
    mOnnxFile = configPath + "/"  + (std::string) fs["onnxFile"];
    mEngineFile = configPath + "/" + (std::string) fs["engineFile"];
}


nvinfer1::ICudaEngine *YoloV5::createEngine(nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config)
{
    mGlogger.setReportableSeverity(Severity::kERROR);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    assert(network);
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, mGlogger);
    assert(parser);
    bool parsed = parser->parseFromFile(mOnnxFile.c_str(), (int) nvinfer1::ILogger::Severity::kWARNING);
    if (!parsed) {
        mGlogger.log(Severity::kERROR, "onnx file parse error, please check onnx file!");
        std::abort();
    }
    // Build engine
    builder->setMaxBatchSize(mMaxSupportBatchSize);
    config->setMaxWorkspaceSize(1_GiB);
    if (strcmp(mQuantizationInfer.c_str(), "FP16") == 0)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
    if (inputDims.d[0] == -1)
    {
        nvinfer1::IOptimizationProfile *profileCalib = builder->createOptimizationProfile();
        const auto inputName = "input";
        nvinfer1::Dims batchDim = inputDims;
        batchDim.d[0] = 1;
        // We do not need to check the return of setDimension and setCalibrationProfile here as all dims are explicitly set
        profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, batchDim);
        profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, batchDim);
        batchDim.d[0] = mMaxSupportBatchSize;
        profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, batchDim);
        config->addOptimizationProfile(profileCalib);
    }
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);
    mGlogger.log(Severity::kINFO, "success create engine!");
    //release network
    network->destroy();
    return engine;
}

void YoloV5::getBindingDimsInfo()
{
    int nb = mEngine->getNbBindings();
    assert(nb == 2), "binding total dim should be three!";
    nvinfer1::Dims inputDims = mEngine->getBindingDimensions(0);
    nvinfer1::Dims dInput = inputDims;
    mInputC = dInput.d[1];
    mInputH = dInput.d[2];
    mInputW = dInput.d[3];
    nvinfer1::Dims outPutBoxesDims = mEngine->getBindingDimensions(1);
    nvinfer1::Dims dOutPutBoxes = outPutBoxesDims;
    mOutputAnchorsNum = dOutPutBoxes.d[1];
    mOutputAnchorsDim = dOutPutBoxes.d[2];
    mOutputAnchorsSize = mOutputAnchorsNum * mOutputAnchorsDim;
}

void YoloV5::getTrtmodelStream()
{
    cudaSetDevice(DEVICE);
    std::ifstream file(mEngineFile, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        mEngineFileSize = file.tellg();
        file.seekg(0, file.beg);
        mTrtModelStream = new char[mEngineFileSize];
        assert(mTrtModelStream);
        file.read(mTrtModelStream, mEngineFileSize);
        file.close();
    }
    mRuntime = nvinfer1::createInferRuntime(mGlogger);
    assert(mRuntime);
    mEngine = mRuntime->deserializeCudaEngine(mTrtModelStream, mEngineFileSize, nullptr);
    assert(mEngine);
    mContext = mEngine->createExecutionContext();
    assert(mContext);
    //create stream
    CHECK(cudaStreamCreate(&mStream));
    getBindingDimsInfo();
    //create fixed maximum input buffer
    int inputSingleByteNum = mInputW * mInputH * mInputC;
    int outputSingleAnchorByteNum = mOutputAnchorsNum * mOutputAnchorsDim;
    CHECK(cudaMalloc(&(mInputData), mMaxSupportBatchSize * inputSingleByteNum));
    CHECK(cudaMalloc(&(mBuff[0]), mMaxSupportBatchSize * inputSingleByteNum * sizeof(float)));
    CHECK(cudaMalloc(&(mBuff[1]), mMaxSupportBatchSize * outputSingleAnchorByteNum * sizeof(float)));
    delete mTrtModelStream;
}

void YoloV5::createInferenceEngine(nvinfer1::IHostMemory **modelStream)
{
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(mGlogger);
    assert(builder);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    assert(config);
    nvinfer1::ICudaEngine *engine = createEngine(builder, config);
    assert(engine != nullptr && "engine create failure!");

    // Serialize the engine
    (*modelStream) = engine->serialize();

    //release all memory
    builder->destroy();
    config->destroy();
    engine->destroy();
}

bool YoloV5::createEngineIfNotExit()
{
    std::ifstream cache(mEngineFile.c_str(), std::ios::binary);
    if (cache)
        return true;
    else
    {
        nvinfer1::IHostMemory* modelStream{nullptr};
        createInferenceEngine(&modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(mEngineFile.c_str(), std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return false;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
    }
    return true;
}