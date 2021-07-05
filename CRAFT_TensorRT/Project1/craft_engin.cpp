
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "sampleEngines.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "craft_engin.h"
#include "opencv2/opencv.hpp"

#include "ROI.h"

bool SampleOnnxCRAFT::build()
{
	bool trt파일있음 = false;
	const std::ifstream checkFile("craft.trt");
	trt파일있음 = checkFile.is_open();

	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
	if (!builder)
	{
		return false;
	}

	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		return false;
	}

	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		return false;
	}

	auto parser
		= SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
	if (!parser)
	{
		return false;
	}

	auto constructed = constructNetwork(builder, network, config, parser);
	if (!constructed)
	{
		return false;
	}
	if (trt파일있음 == false)
	{
		mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
			builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
		if (!mEngine)
		{
			return false;
		}

#if 1
		const std::ifstream checkFile("craft.trt");
		if (checkFile.is_open() == false)
		{
			nvinfer1::IHostMemory* serializedModel = mEngine->serialize();
			// Save Engine 
			std::cout << "Saving serialized model to disk...";
			std::ofstream fileEngine("craft.trt", std::ios::out | std::ios::binary);
			fileEngine.write((char*)(serializedModel->data()), serializedModel->size());
			fileEngine.close();
			serializedModel->destroy();
			std::cout << " DONE!" << std::endl;
		}
#endif

	}
	else
	{

		std::string loadEngine("craft.trt");
		//std::string loadEngine("craft_1402.trt");
		//std::string loadEngine("craft_526.trt");
		//std::string loadEngine("D:\\Python\\TensorRT-7.2.3.4\\bin\\engine.trt");
		sample::gLogInfo << "Loading engine from: " << loadEngine << std::endl;
		mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
			sample::loadEngine(loadEngine, -1, std::cerr), samplesCommon::InferDeleter());

		//			mEngine->
	}
	// assert(network->getNbInputs() == 1);

	if (!mEngine)
	{
		return false;
	}
	mInputDims = network->getInput(0)->getDimensions();
	auto inputName = network->getInput(0)->getName();

	//assert(mInputDims.nbDims == 4);
	sample::gLogInfo << "Input : " << inputName << std::endl;
	sample::gLogInfo << "InputDims : " << mInputDims << std::endl;

	// assert(network->getNbOutputs() == 1);
	mOutputDims = network->getOutput(0)->getDimensions();
	auto outputName = network->getOutput(0)->getName();
	// assert(mOutputDims.nbDims == 2);
	sample::gLogInfo << "Output : " << outputName << std::endl;
	sample::gLogInfo << "OutputDims : " << mOutputDims << std::endl;
	return true;
}

bool SampleOnnxCRAFT::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
	SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
	SampleUniquePtr<nvonnxparser::IParser>& parser)
{
	auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
		static_cast<int>(sample::gLogger.getReportableSeverity()));
	if (!parsed)
	{
		return false;
	}

	if (mParams.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (mParams.int8)
	{
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
	}

	samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

	return true;
}


void preprocessImage(cv::Mat & frame, float* gpu_input, const nvinfer1::Dims& dims)
{

	auto input_width = dims.d[3];
	auto input_height = dims.d[2];
	auto channels = dims.d[1];
	auto input_size = cv::Size(input_width, input_height);

	// 크기 조정
	cv::Mat resized;
	cv::resize(frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

	// 정규화
	cv::Mat flt_image;
	resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
	cv::subtract(flt_image, cv::Scalar(0.485f, 0.486f, 0.406f), flt_image, cv::noArray(), -1);
	cv::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

	// 전치 - width*height를 height*width로 변환
	cv::Mat transpose;
	cv::transpose(flt_image, transpose);
	// 채널분리 - 복사
	std::vector<cv::Mat> chw;
	for (size_t i = 0; i < transpose.channels(); i++)
	{
		chw.emplace_back(cv::Mat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
	}
	cv::split(flt_image, chw);
}
void 영상파일목록(std::string &폴더, std::queue<string> &영상파일, vector<string> &확장자, bool 경로포함 = true)
{
	WIN32_FIND_DATA fd;
	HANDLE hFind;

	std::string fp = 폴더 + "\\*";

	hFind = FindFirstFile(fp.c_str(), &fd);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		return;
	}
	else
	{
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				std::string extension;
				std::string name(fd.cFileName);
				size_t i = name.rfind('.', name.length());
				if (i != string::npos)
				{
					extension = name.substr(i + 1, name.length() - 1);
				}

				auto 있음 = find(확장자.begin(), 확장자.end(), extension);
				if (있음 != 확장자.end())
				{
					if (경로포함)
					{
						영상파일.push(폴더 + "\\" + name);
					}
					else
					{

						영상파일.push(name);
					}
				}
			}
		} while (::FindNextFile(hFind, &fd));
		FindClose(hFind);
	}

}

void SaveHeatimage(cv::Mat &render_img, const char *폴더이름, string &파일이름)
{
	cv::Mat heatmap;
	render_img.convertTo(heatmap, CV_8UC1, 255.f);
	cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_JET);
	cv::imwrite(폴더이름 + 파일이름, heatmap);
}

bool SampleOnnxCRAFT::infer()
{
	// Create RAII buffer manager object
	samplesCommon::BufferManager buffers(mEngine);

	sample::gLogInfo << "buffers" << std::endl;
	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context)
	{
		return false;
	}

	vector<string> 확장자필터;
	확장자필터.push_back("jpg");

	//std::string 경로(".\\");
	std::string 경로("Y:\\영상");

	std:queue<string> 영상목록;
	sample::gLogInfo << "context" << std::endl;
	영상파일목록(경로, 영상목록, 확장자필터, false);

	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	float* gpu_output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

	경로 = 경로.append("\\");

	while(영상목록.size() > 0)
	{
		std::string 파일이름 = 영상목록.front();
		영상목록.pop();

		cv::Mat frame = cv::imread(경로 + 파일이름);
		if (frame.empty())
		{
			std::cerr << "Input image " << 파일이름 << " load failed\n";
			continue;
		}

		DWORD tick = GetTickCount(); 
		preprocessImage(frame, hostDataBuffer, mInputDims);
		tick = GetTickCount() - tick;

/*
		// Read the input data into the managed buffers
		assert(mParams.inputTensorNames.size() == 1);
		if (!processInput(buffers))
		{
			return false;
		}
		*/

		// Memcpy from host input buffers to device input buffers
		DWORD tick4 = GetTickCount(); 
		buffers.copyInputToDevice();

		bool status = context->executeV2(buffers.getDeviceBindings().data());
		if (!status)
		{
			return false;
		}

		// Memcpy from device output buffers to host output buffers
		buffers.copyOutputToHost();
		
		tick4 = GetTickCount() - tick4;

		DWORD tick1 = GetTickCount();

		const int outputSize = mOutputDims.d[1];
		auto output_width = mOutputDims.d[2];
		auto output_height = mOutputDims.d[1];
		auto channels = mOutputDims.d[3];
		auto output_size = cv::Size(output_width, output_height);


		cv::Mat cuda(output_size, CV_32FC2, gpu_output);
		vector<cv::Mat> cpuMat(2);
		cv::split(cuda, cpuMat);
		tick1 = GetTickCount() - tick1;
		

		cv::Mat sum;
		cv::add(cpuMat[0], cpuMat[1], sum);

		DWORD tick2 = GetTickCount();

		// ROI 찾기 - 
		cv::Mat ROI;
		// 시도_20210623_01(sum, cpuMat[1], ROI, 파일이름); //시도_20210623_01(cpuMat[0], cpuMat[1], ROI, 파일이름);

		// 단어확률 + 자간확률 => 사각형
		// 시도_20210630_01(cpuMat[0], cpuMat[1], ROI, 파일이름); //시도_20210623_01(cpuMat[0], cpuMat[1], ROI, 파일이름);
		시도_20210624_01(cpuMat[0], cpuMat[1], ROI, 파일이름); //시도_20210623_01(cpuMat[0], cpuMat[1], ROI, 파일이름);
		//시도_20210623_02(cpuMat[0], cpuMat[1], ROI, 파일이름); //시도_20210623_01(cpuMat[0], cpuMat[1], ROI, 파일이름);
		//시도_20210623_01(frame, cpuMat[1], ROI, 파일이름); //시도_20210623_01(cpuMat[0], cpuMat[1], ROI, 파일이름);
		tick2 = GetTickCount() - tick2;
		//cv::Mat score_text(output_size, CV_32FC1, gpu_output);
		//cv::Mat score_link(output_size, CV_32FC1, gpu_output + output_width * output_height);
		cv::Mat render_img;
		cv::hconcat(cpuMat[0], cpuMat[1], render_img);
		SaveHeatimage(sum, "Y:\\결과2\\res1_", 파일이름);
		// SaveHeatimage(render_img, "data\\result\\res_", 파일이름);

		sample::gLogInfo << 파일이름 << " tick : " << tick << "," << tick4 << "," << tick1 << "," << tick2 << std::endl;

		
	}
	// Verify results

/*   if (!verifyOutput(buffers))
	{
		return false;
	}
	*/
	return true;
}
bool SampleOnnxCRAFT::processInput(const samplesCommon::BufferManager& buffers)
{

	const int inputH = 1280;
	const int inputW = 640;
	const int c = 3;
	//	const int inputH = mInputDims.d[2];
	//		const int inputW = mInputDims.d[3];

	sample::gLogInfo << "InputH : " << inputH << std::endl;
	sample::gLogInfo << "InputW : " << inputW << std::endl;
	sample::gLogInfo << mParams.inputTensorNames[0] << std::endl;
	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

	std::vector<uint8_t> fileData(inputH * inputW * c);
	for (int i = 0; i < inputH * inputW * c; i++)
	{
		hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
	}
	sample::gLogInfo << "hostdata" << std::endl;
	/*
		// Read a random digit file
		srand(unsigned(time(nullptr)));
		mNumber = rand() % 10;
		readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

		// Print an ascii representation
		sample::gLogInfo << "Input:" << std::endl;
		for (int i = 0; i < inputH * inputW; i++)
		{
			sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
		}
		sample::gLogInfo << std::endl;

		float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
		for (int i = 0; i < inputH * inputW; i++)
		{
			hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
		}
		*/
	return true;
}
bool SampleOnnxCRAFT::verifyOutput(const samplesCommon::BufferManager& buffers)
{
	const int outputSize = mOutputDims.d[1];
	float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
	float val{ 0.0f };
	int idx{ 0 };

	// Calculate Softmax
	float sum{ 0.0f };
	for (int i = 0; i < outputSize; i++)
	{
		output[i] = exp(output[i]);
		sum += output[i];
	}

	sample::gLogInfo << "Output:" << std::endl;
	for (int i = 0; i < outputSize; i++)
	{
		output[i] /= sum;
		val = std::max(val, output[i]);
		if (val == output[i])
		{
			idx = i;
		}

		sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
			<< " "
			<< "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
			<< std::endl;
	}
	sample::gLogInfo << std::endl;

	return idx == mNumber && val > 0.9f;
}

