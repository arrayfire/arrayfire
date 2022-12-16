/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/defines.hpp>
#include <common/module_loading.hpp>

#include <dlfcn.h>

#include <string>
using std::string;

namespace arrayfire {
namespace common {

void* getFunctionPointer(LibHandle handle, const char* symbolName) {
    return dlsym(handle, symbolName);
}

LibHandle loadLibrary(const char* library_name) {
    return dlopen(library_name, RTLD_LAZY);
}
void unloadLibrary(LibHandle handle) { dlclose(handle); }

string getErrorMessage() {
    char* errMsg = dlerror();
    if (errMsg) { return string(errMsg); }
    // constructing std::basic_string from NULL/0 address is
    // invalid and has undefined behavior
    return string("No Error");
}

}  // namespace common
}  // namespace arrayfire
