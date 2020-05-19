/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if !defined(AF_CPU)

#include <common/kernel_cache.hpp>

#include <common/compile_kernel.hpp>
#include <device_manager.hpp>
#include <platform.hpp>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using detail::Kernel;
using std::back_inserter;
using std::map;
using std::string;
using std::transform;
using std::vector;

namespace common {

using KernelMap = map<string, Kernel>;

KernelMap& getCache(const int device) {
    thread_local KernelMap caches[detail::DeviceManager::MAX_DEVICES];
    return caches[device];
}

void cacheKernel(const int device, const string& nameExpr, const Kernel entry) {
    getCache(device).emplace(nameExpr, entry);
}

Kernel lookupKernel(const int device, const string& nameExpr,
                    const vector<string>& sources) {
    auto& cache = getCache(device);
    auto iter   = cache.find(nameExpr);

    if (iter != cache.end()) return iter->second;

#if defined(AF_CUDA) && defined(AF_CACHE_KERNELS_TO_DISK)
    Kernel kernel = loadKernel(device, nameExpr, sources);
    if (kernel.getModule() != nullptr && kernel.getKernel() != nullptr) {
        cacheKernel(device, nameExpr, kernel);
        return kernel;
    }
#endif

    return Kernel{nullptr, nullptr};
}

Kernel findKernel(const string& kernelName, const vector<string>& sources,
                  const vector<TemplateArg>& targs,
                  const vector<string>& compileOpts) {
    vector<string> args;
    args.reserve(targs.size());

    transform(targs.begin(), targs.end(), back_inserter(args),
              [](const TemplateArg& arg) -> string { return arg._tparam; });

    string tInstance = kernelName;
    if (args.size() > 0) {
        tInstance = kernelName + "<" + args[0];
        for (size_t i = 1; i < args.size(); ++i) {
            tInstance += ("," + args[i]);
        }
        tInstance += ">";
    }

    int device    = detail::getActiveDeviceId();
    Kernel kernel = lookupKernel(device, tInstance, sources);

    if (kernel.getModule() == nullptr || kernel.getKernel() == nullptr) {
        kernel = compileKernel(kernelName, tInstance, sources, compileOpts);
        cacheKernel(device, tInstance, kernel);
    }

    return kernel;
}

}  // namespace common

#endif
