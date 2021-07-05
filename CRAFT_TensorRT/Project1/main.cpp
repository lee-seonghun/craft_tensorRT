
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "opencv2/opencv.hpp"

#include <cuda_runtime_api.h>
#include "logger.h"
#include "sampleEngines.h"
#include "sampleOptions.h"
#include "sampleUtils.h"
#include "craft_engin.h"

const std::string gSampleName = "craft_526.onnx";
//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
	samplesCommon::OnnxSampleParams params;
	if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
	{
		params.dataDirs.push_back("data/craft/");
		params.dataDirs.push_back("data/samples/mnist/");
	}
	else //!< Use the data directory provided by the user
	{
		params.dataDirs = args.dataDirs;
	}
	//    params.onnxFileName = "mnist.onnx";
//	params.onnxFileName = "craft_2140.onnx";
	params.onnxFileName = "craft_1280.onnx";
//	params.onnxFileName = "craft_526.onnx";
	//params.onnxFileName = "test.onnx";
	params.inputTensorNames.push_back("input.1");
	params.outputTensorNames.push_back("291");
	params.dlaCore = args.useDLACore;
	params.int8 = args.runInInt8;
	params.fp16 = args.runInFp16;

	return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
	std::cout
		<< "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
		<< std::endl;
	std::cout << "--help          Display help information" << std::endl;
	std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
		"multiple times to add multiple directories. If no data directories are given, the default is to use "
		"(data/samples/mnist/, data/mnist/)"
		<< std::endl;
	std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
		"where n is the number of DLA engines on the platform."
		<< std::endl;
	std::cout << "--int8          Run in Int8 mode." << std::endl;
	std::cout << "--fp16          Run in FP16 mode." << std::endl;
}
int main(int argc, char** argv)
{
	samplesCommon::Args args;
	bool argsOK = samplesCommon::parseArgs(args, argc, argv);
	if (!argsOK)
	{
		sample::gLogError << "Invalid arguments" << std::endl;
		printHelpInfo();
		return EXIT_FAILURE;
	}
	if (args.help)
	{
		printHelpInfo();
		return EXIT_SUCCESS;
	}

	auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

	sample::gLogger.reportTestStart(sampleTest);

	SampleOnnxCRAFT sample(initializeSampleParams(args));

	sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

	if (!sample.build())
	{
		return sample::gLogger.reportFail(sampleTest);
	}

	if (!sample.infer())
	{
		return sample::gLogger.reportFail(sampleTest);
	}
	/*
		return sample::gLogger.reportPass(sampleTest);
		*/
	return 0;
}

