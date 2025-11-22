#include <iostream>
#include <string>
#include <vector>

// argparse
#include <cxxopts.hpp>

// C IPU/xmem/logger APIs
extern "C"
{
#include "ipu/ipu.h"
#include "xmem/xmem.h"
#include "logging/logger.h"
}

int main(int argc, char **argv)
{
    logger__init();

    cxxopts::Options options("ipu-as", "Simple IPU assembler skeleton");
    options.add_options()("i,input", "Input assembly file(s)", cxxopts::value<std::vector<std::string>>())("I,include", "Include paths for assembler", cxxopts::value<std::vector<std::string>>())("L,libpath", "Library search path", cxxopts::value<std::vector<std::string>>())("l,lib", "Link with library (name)", cxxopts::value<std::vector<std::string>>())("o,output", "Output binary file", cxxopts::value<std::string>()->default_value("out.bin"))("v,verbose", "Verbose logging")("h,help", "Print help");

    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (result.count("verbose"))
    {
        logger__set_level(LOG_LEVEL_DEBUG);
    }

    std::vector<std::string> inputs;
    if (result.count("input"))
        inputs = result["input"].as<std::vector<std::string>>();
    if (inputs.empty())
    {
        LOG_ERROR("No input files specified");
        std::cerr << "No input file specified. Use -i <file> (can be repeated)\n";
        return 2;
    }

    std::string output = result["output"].as<std::string>();

    std::vector<std::string> include_paths;
    if (result.count("include"))
        include_paths = result["include"].as<std::vector<std::string>>();
    std::vector<std::string> lib_paths;
    if (result.count("libpath"))
        lib_paths = result["libpath"].as<std::vector<std::string>>();
    std::vector<std::string> libs;
    if (result.count("lib"))
        libs = result["lib"].as<std::vector<std::string>>();

    LOG_INFO("ipu-as starting");
    LOG_INFO("Output: %s", output.c_str());

    for (auto &f : inputs)
        LOG_INFO("Input: %s", f.c_str());

    // Initialize IPU for assembler-time simulation
    ipu__obj_t *ipu = ipu__init_ipu(); // create default ipu object
    if (!ipu)
    {
        LOG_ERROR("Failed to init IPU");
        return 3;
    }

    // TODO: actual assembler implementation
    // - parse input files
    // - resolve include paths
    // - assemble instructions to binary
    // - resolve external libraries if specified (lib_paths/libs)
    // - write output

    LOG_INFO("ipu-as completed (skeleton)");
    return 0;
}
