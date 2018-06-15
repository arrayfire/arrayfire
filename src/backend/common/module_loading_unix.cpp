/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/module_loading.hpp>
#include <common/defines.hpp>

#include <dlfcn.h>

#include <string>
using std::string;

namespace common {


LibHandle loadLibrary(const char* library_name) {
    return dlopen(library_name, RTLD_LAZY);
}
void unloadLibrary(LibHandle handle) {
    dlclose(handle);
}

string getErrorMessage() {
    string error_message(dlerror());
    return error_message;
}

}
