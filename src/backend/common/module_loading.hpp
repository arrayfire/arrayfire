/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/defines.hpp>

namespace arrayfire {
namespace common {

void* getFunctionPointer(LibHandle handle, const char* symbolName);

LibHandle loadLibrary(const char* library_name);

void unloadLibrary(LibHandle handle);

std::string getErrorMessage();

}  // namespace common
}  // namespace arrayfire
