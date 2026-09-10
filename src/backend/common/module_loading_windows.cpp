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

#include <Windows.h>
#include <string>

using std::string;

namespace arrayfire {
namespace common {

void* getFunctionPointer(LibHandle handle, const char* symbolName) {
    return GetProcAddress(handle, symbolName);
}

LibHandle loadLibrary(const char* library_name) {
    return LoadLibrary(library_name);
}

void unloadLibrary(LibHandle handle) { FreeLibrary(handle); }

string getErrorMessage() {
    char* lpMsgBuf = nullptr;
    DWORD dw       = GetLastError();

    // FormatMessage only writes lpMsgBuf when it succeeds. On failure it
    // returns 0 and leaves the pointer untouched, so it must be initialized
    // and the result checked before it is dereferenced.
    DWORD chars = FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPTSTR>(&lpMsgBuf), 0, NULL);

    if (chars == 0 || lpMsgBuf == nullptr) {
        return "Unknown error (system error code " + std::to_string(dw) + ")";
    }

    string error_message(lpMsgBuf);
    // FORMAT_MESSAGE_ALLOCATE_BUFFER allocates with LocalAlloc; the caller
    // owns the buffer and must release it.
    LocalFree(lpMsgBuf);
    return error_message;
}

}  // namespace common
}  // namespace arrayfire
