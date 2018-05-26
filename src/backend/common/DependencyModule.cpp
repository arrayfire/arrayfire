/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/DependencyModule.hpp>
#include <common/module_loading.hpp>
#include <algorithm>
#include <string>

#ifdef OS_WIN
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

using std::string;

namespace {

    std::string libName(std::string name) {
        return libraryPrefix + name + librarySuffix;
    }
}

namespace common {

#ifdef OS_WIN
void* DependencyModule::getFunctionPointer(LibHandle handle, const char* symbolName) {
    return GetProcAddress(handle, symbolName);
}
#else
void* DependencyModule::getFunctionPointer(LibHandle handle, const char* symbolName) {
    return dlsym(handle, symbolName);
}
#endif

DependencyModule::DependencyModule(const char* plugin_file_name, const char** paths)
    : handle(nullptr) {
    // TODO(umar): Implement handling of non-standard paths
    if(plugin_file_name) {
        handle = loadLibrary(libName(plugin_file_name).c_str());
    }
}

DependencyModule::~DependencyModule() {
    if(handle) {
        unloadLibrary(handle);
    }
}

bool DependencyModule::isLoaded() {
    return (bool)handle;
}

bool DependencyModule::symbolsLoaded() {
    return all_of(begin(functions), end(functions), [](void* ptr){ return ptr != nullptr; });
}

string DependencyModule::getErrorMessage() {
    return common::getErrorMessage();
}
}
