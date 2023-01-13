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
#include <common/deterministicHash.hpp>
#include <common/kernel_cache.hpp>
#include <device_manager.hpp>
#include <platform.hpp>

#include <nonstd/span.hpp>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

using detail::Kernel;
using detail::Module;

using nonstd::span;
using std::back_inserter;
using std::shared_timed_mutex;
using std::string;
using std::to_string;
using std::transform;
using std::unordered_map;
using std::vector;

namespace arrayfire {
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
                 const span<const common::Source> sources,
                 const span<const TemplateArg> targs,
                 const span<const string> options, const bool sourceIsJIT) {
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

    // The JIT kernel uses the hashing of the kernelName (tInstance) only to
    // speed up to search for its cached kernel.  All the other kernels have the
    // full source code linked in, and will hash the full code + options
    // instead.
    size_t moduleKeyCache = 0;
    if (sourceIsJIT) {
        moduleKeyCache = deterministicHash(tInstance);
    } else {
        moduleKeyCache = (sources.size() == 1 && sources[0].hash)
                             ? sources[0].hash
                             : deterministicHash(sources);
        moduleKeyCache = deterministicHash(options, moduleKeyCache);
#if defined(AF_CUDA)
        moduleKeyCache = deterministicHash(tInstance, moduleKeyCache);
#endif
    }
    const int device  = detail::getActiveDeviceId();
    Module currModule = findModule(device, moduleKeyCache);

    if (!currModule) {
        // When saving on disk, the moduleKeyDisk has to correspond with the
        // full code + optinos (in all circumstances). A recalculation for JIT
        // is necessary, while for the others we can reuse the moduleKeyCache.
        size_t moduleKeyDisk = 0;
        if (sourceIsJIT) {
            moduleKeyDisk = (sources.size() == 1 && sources[0].hash)
                                ? sources[0].hash
                                : deterministicHash(sources);
            moduleKeyDisk = deterministicHash(options, moduleKeyDisk);
#if defined(AF_CUDA)
            moduleKeyDisk = deterministicHash(tInstance, moduleKeyDisk);
#endif
        } else {
            moduleKeyDisk = moduleKeyCache;
        }
        currModule =
            loadModuleFromDisk(device, to_string(moduleKeyDisk), sourceIsJIT);
        if (!currModule) {
            vector<string> sources_str;
            for (const auto& s : sources) {
                sources_str.push_back({s.ptr, s.length});
            }
            currModule = compileModule(to_string(moduleKeyDisk), sources_str,
                                       options, {{tInstance}}, sourceIsJIT);
        }

        std::unique_lock<shared_timed_mutex> writeLock(getCacheMutex(device));
        auto& cache = getCache(device);
        auto iter   = cache.find(moduleKeyCache);
        if (iter == cache.end()) {
            // If not found, this thread is the first one to compile
            // this kernel. Keep the generated module.
            Module mod = currModule;
            getCache(device).emplace(moduleKeyCache, mod);
        } else {
            currModule.unload();  // dump the current threads extra
                                  // compilation
            currModule = iter->second;
        }
    }
    return getKernel(currModule, tInstance, sourceIsJIT);
}

}  // namespace common
}  // namespace arrayfire

#endif
