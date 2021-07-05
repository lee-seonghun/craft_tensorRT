#pragma once

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


class SampleOnnxCRAFT
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	SampleOnnxCRAFT(const samplesCommon::OnnxSampleParams& params)
		: mParams(params)
		, mEngine(nullptr)
	{
	}//!
		//! \brief Function builds the network engine
		//!
	bool build();

	//!
	//! \brief Runs the TensorRT inference engine for this sample
	//!
	bool infer();
	// nvinfer1::ICudaEngine* loadEngine(const std::string& engine, int DLACore, std::ostream& err);

private:
	samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
	int mNumber{ 0 };             //!< The number to classify

	std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

	//!
	//! \brief Parses an ONNX model for MNIST and creates a TensorRT network
	//!
	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
		SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
		SampleUniquePtr<nvonnxparser::IParser>& parser);

	//!
	//! \brief Reads the input  and stores the result in a managed buffer
	//!
	bool processInput(const samplesCommon::BufferManager& buffers);

	//!
	//! \brief Classifies digits and verify result
	//!
	bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

