/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/DependencyModule.hpp>
#include <common/Logger.hpp>
#include <common/module_loading.hpp>
#include <algorithm>
#include <string>

#ifdef OS_WIN
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef OS_WIN
#include <Windows.h>
static const char* librarySuffix = ".dll";
static const char* libraryPrefix = "";
#elif defined(OS_MAC)
static const char* librarySuffix = ".dylib";
static const char* libraryPrefix = "lib";
#elif defined(OS_LNX)
static const char* librarySuffix = ".so";
static const char* libraryPrefix = "lib";
#else
#error "Unsupported platform"
#endif

using std::string;
using std::vector;

namespace {

std::string libName(std::string name) {
    return libraryPrefix + name + librarySuffix;
}
}  // namespace

namespace common {

DependencyModule::DependencyModule(const char* plugin_file_name,
                                   const char** paths)
    : handle(nullptr), logger(common::loggerFactory("platform")) {
    // TODO(umar): Implement handling of non-standard paths
    UNUSED(paths);
    if (plugin_file_name) {
        string filename = libName(plugin_file_name);
        AF_TRACE("Attempting to load: {}", filename);
        handle = loadLibrary(filename.c_str());
        if (handle) {
            AF_TRACE("Found: {}", filename);
        } else {
            AF_TRACE("Unable to open {}", plugin_file_name);
        }
    }
}

DependencyModule::DependencyModule(const vector<string> plugin_base_file_name,
                                   const vector<string> suffixes,
                                   const vector<string> paths)
    : handle(nullptr), logger(common::loggerFactory("platform")) {
    for (const string& base_name : plugin_base_file_name) {
        for (const string& path : paths) {
            for (const string& suffix : suffixes) {
                string filename = libName(base_name + suffix);
                AF_TRACE("Attempting to load: {}", filename);
                handle = loadLibrary(filename.c_str());
                if (handle) {
                    AF_TRACE("Found: {}", filename);
                    return;
                }
            }
        }
    }
    AF_TRACE("Unable to open {}", plugin_base_file_name[0]);
}

DependencyModule::~DependencyModule() noexcept {
    if (handle) { unloadLibrary(handle); }
}

bool DependencyModule::isLoaded() const noexcept { return (bool)handle; }

bool DependencyModule::symbolsLoaded() const noexcept {
    return all_of(begin(functions), end(functions),
                  [](void* ptr) { return ptr != nullptr; });
}

string DependencyModule::getErrorMessage() const noexcept {
    return common::getErrorMessage();
}

spdlog::logger* DependencyModule::getLogger() const noexcept {
    return logger.get();
}

}  // namespace common
