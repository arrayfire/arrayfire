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
#include <shared_mutex>
#include <string>
#include <vector>

using detail::Kernel;

using std::back_inserter;
using std::map;
using std::shared_lock;
using std::shared_timed_mutex;
using std::string;
using std::transform;
using std::unique_lock;
using std::vector;

namespace common {

using KernelMap = map<string, Kernel>;

shared_timed_mutex& getCacheMutex(const int device) {
    static shared_timed_mutex jitMutexes[detail::DeviceManager::MAX_DEVICES];
    return jitMutexes[device];
}

KernelMap& getCache(const int device) {
    static KernelMap caches[detail::DeviceManager::MAX_DEVICES];
    return caches[device];
}

Kernel findKernel(const int device, const string& key) {
    shared_lock<shared_timed_mutex> readLock(getCacheMutex(device));
    auto& cache = getCache(device);
    auto iter   = cache.find(key);
    if (iter != cache.end()) { return iter->second; }
    return Kernel{nullptr, nullptr};
}

Kernel getKernel(const string& kernelName, const vector<string>& sources,
                 const vector<TemplateArg>& targs,
                 const vector<string>& compileOpts, const bool isKernelJIT) {
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
    Kernel kernel = findKernel(device, tInstance);

    if (kernel.getModule() == nullptr || kernel.getKernel() == nullptr) {
#if defined(AF_CUDA) && defined(AF_CACHE_KERNELS_TO_DISK)
        kernel = loadKernelFromDisk(device, tInstance, sources);
        if (kernel.getModule() == nullptr || kernel.getKernel() == nullptr) {
            kernel = compileKernel(kernelName, tInstance, sources, compileOpts,
                                   isKernelJIT);
        }
#else
        kernel = compileKernel(kernelName, tInstance, sources, compileOpts,
                               isKernelJIT);
#endif
        unique_lock<shared_timed_mutex> writeLock(getCacheMutex(device));

        auto& cache = getCache(device);

        // Lookup in case another thread finished adding the same kernel
        // to this device's cache during it's life time with write access
        auto iter = cache.find(tInstance);
        if (iter != cache.end()) {
            kernel = iter->second;
        } else {
            cache.emplace(tInstance, kernel);
        }
    }
    return kernel;
}

}  // namespace common

#endif
