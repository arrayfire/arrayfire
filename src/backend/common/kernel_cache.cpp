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

#include <common/compile_module.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <platform.hpp>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

using detail::Kernel;
using detail::Module;

using std::back_inserter;
using std::string;
using std::transform;
using std::unordered_map;
using std::vector;

namespace common {

using ModuleMap = unordered_map<string, Module>;

ModuleMap& getCache(const int device) {
    thread_local ModuleMap caches[detail::DeviceManager::MAX_DEVICES];
    return caches[device];
}

Module findModule(const int device, const string& key) {
    auto& cache = getCache(device);
    auto iter   = cache.find(key);
    if (iter != cache.end()) { return iter->second; }
    return Module{nullptr};
}

Kernel getKernel(const string& kernelName, const vector<string>& sources,
                 const vector<TemplateArg>& targs,
                 const vector<string>& options, const bool sourceIsJIT) {
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

    const bool notJIT = !sourceIsJIT;

    vector<string> hashingVals;
    hashingVals.reserve(1 + (notJIT * (sources.size() + options.size())));
    hashingVals.push_back(tInstance);
    if (notJIT) {
        // This code path is only used for regular kernel compilation
        // since, jit funcName(kernelName) is unique to use it's hash
        // for caching the relevant compiled/linked module
        hashingVals.insert(hashingVals.end(), sources.begin(), sources.end());
        hashingVals.insert(hashingVals.end(), options.begin(), options.end());
    }

    const string moduleKey = std::to_string(deterministicHash(hashingVals));
    const int device       = detail::getActiveDeviceId();
    Module currModule      = findModule(device, moduleKey);

    if (currModule.get() == nullptr) {
        currModule = loadModuleFromDisk(device, moduleKey);
        if (currModule.get() == nullptr) {
            currModule = compileModule(moduleKey, sources, options, {tInstance},
                                       sourceIsJIT);
        }
        getCache(device).emplace(moduleKey, currModule);
    }
#if defined(AF_CUDA)
    return getKernel(currModule, tInstance, sourceIsJIT);
#elif defined(AF_OPENCL)
    return getKernel(currModule, kernelName, sourceIsJIT);
#endif
}

}  // namespace common

#endif
