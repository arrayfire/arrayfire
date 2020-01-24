/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if !defined(AF_CPU)

#include <common/kernel_cache.hpp>

#include <device_manager.hpp>
#include <platform.hpp>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

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

common::Kernel lookupKernel(const int device, const string& nameExpr) {
    auto& cache = getCache(device);
    auto iter   = cache.find(nameExpr);
    return (iter == cache.end() ? common::Kernel{0, 0} : iter->second);
}

void cacheKernel(const int device, const string& nameExpr,
                 const common::Kernel entry) {
    getCache(device).emplace(nameExpr, entry);
}

Kernel findKernel(const string& kernelName, const vector<string>& sources,
                  const vector<TemplateArg>& targs,
                  const vector<string>& compileOpts) {
    vector<string> args;
    args.reserve(targs.size());

    transform(targs.begin(), targs.end(), back_inserter(args),
              [](const TemplateArg& arg) -> string { return arg._tparam; });

    string tInstance = kernelName + "<" + args[0];
    for (size_t i = 1; i < args.size(); ++i) { tInstance += ("," + args[i]); }
    tInstance += ">";

    int device    = detail::getActiveDeviceId();
    Kernel kernel = lookupKernel(device, tInstance);

    if (kernel.prog == 0 || kernel.kern == 0) {
        compileKernel(kernel, kernelName, tInstance, sources, compileOpts);
        cacheKernel(device, tInstance, kernel);
    }

    return kernel;
}

}  // namespace common

#endif
