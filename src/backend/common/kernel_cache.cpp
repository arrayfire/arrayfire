/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if !defined(AF_CPU)

#include <common/compile_module.hpp>
#include <common/kernel_cache.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <platform.hpp>

#include <algorithm>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

using detail::Kernel;
using detail::Module;

using std::back_inserter;
using std::shared_timed_mutex;
using std::string;
using std::to_string;
using std::transform;
using std::unordered_map;
using std::vector;

namespace common {

using ModuleMap = unordered_map<size_t, Module>;

shared_timed_mutex& getCacheMutex(const int device) {
    static shared_timed_mutex mutexes[detail::DeviceManager::MAX_DEVICES];
    return mutexes[device];
}

ModuleMap& getCache(const int device) {
    static ModuleMap* caches =
        new ModuleMap[detail::DeviceManager::MAX_DEVICES];
    return caches[device];
}

Module findModule(const int device, const size_t& key) {
    std::shared_lock<shared_timed_mutex> readLock(getCacheMutex(device));
    auto& cache = getCache(device);
    auto iter   = cache.find(key);
    if (iter != cache.end()) { return iter->second; }
    return Module{};
}

Kernel getKernel(const string& kernelName,
                 const vector<common::Source>& sources,
                 const vector<TemplateArg>& targs,
                 const vector<string>& options, const bool sourceIsJIT) {
    string tInstance = kernelName;

#if defined(AF_CUDA)
    auto targsIt  = targs.begin();
    auto targsEnd = targs.end();
    if (targsIt != targsEnd) {
        tInstance += '<' + targsIt->_tparam;
        while (++targsIt != targsEnd) { tInstance += ',' + targsIt->_tparam; }
        tInstance += '>';
    }
#else
    UNUSED(targs);
#endif

    size_t moduleKey = 0;
    if (sourceIsJIT) {
        moduleKey = deterministicHash(tInstance);
    } else {
        moduleKey = (sources.size() == 1 && sources[0].hash)
                        ? sources[0].hash
                        : deterministicHash(sources);
        moduleKey = deterministicHash(options, moduleKey);
#if defined(AF_CUDA)
        moduleKey = deterministicHash(tInstance, moduleKey);
#endif
    }
    const int device  = detail::getActiveDeviceId();
    Module currModule = findModule(device, moduleKey);

    if (!currModule) {
        currModule =
            loadModuleFromDisk(device, to_string(moduleKey), sourceIsJIT);
        if (!currModule) {
            vector<string> sources_str;
            for (auto s : sources) { sources_str.push_back({s.ptr, s.length}); }
            currModule = compileModule(to_string(moduleKey), sources_str,
                                       options, {tInstance}, sourceIsJIT);
        }

        std::unique_lock<shared_timed_mutex> writeLock(getCacheMutex(device));
        auto& cache = getCache(device);
        auto iter   = cache.find(moduleKey);
        if (iter == cache.end()) {
            // If not found, this thread is the first one to compile
            // this kernel. Keep the generated module.
            Module mod = currModule;
            getCache(device).emplace(moduleKey, mod);
        } else {
            currModule.unload();  // dump the current threads extra
                                  // compilation
            currModule = iter->second;
        }
    }
    return getKernel(currModule, tInstance, sourceIsJIT);
}

}  // namespace common

#endif
