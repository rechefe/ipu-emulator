/* ETISS plugin entry points for the IPU architecture. */

#define ETISS_LIBNAME IPU
#include "etiss/helper/CPUArchLibrary.h"

#include "IPUArch.h"

#include <string>

extern "C"
{

ETISS_LIBRARYIF_VERSION_FUNC_IMPL

ETISS_PLUGIN_EXPORT unsigned IPU_countCPUArch()
{
    return 1;
}

ETISS_PLUGIN_EXPORT const char *IPU_nameCPUArch(unsigned index)
{
    return index == 0 ? "IPU" : "";
}

ETISS_PLUGIN_EXPORT etiss::CPUArch *IPU_createCPUArch(unsigned index, std::map<std::string, std::string> options)
{
    if (index != 0)
        return nullptr;
    unsigned coreno = 0;
    auto it = options.find("coreno");
    if (it != options.end())
        coreno = (unsigned)std::stoul(it->second);
    return new IPUArch(coreno);
}

ETISS_PLUGIN_EXPORT void IPU_deleteCPUArch(etiss::CPUArch *arch)
{
    delete arch;
}
}
