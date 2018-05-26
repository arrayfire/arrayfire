/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <module_loading.hpp>
#include <common/defines.hpp>

#include <string>
#include <Windows.h>

using std::string;

namespace common {

LibHandle loadLibrary(const char* library_name) {
    return LoadLibrary(library_name);
}

void unloadLibrary(LibHandle handle) {
    FreeLibrary(handle);
}

string getErrorMessage() {
    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError();

    size_t characters_in_message;
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                  FORMAT_MESSAGE_FROM_SYSTEM |
                  FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL,
                  dw,
                  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  (LPTSTR) &lpMsgBuf,
                  0, NULL );
    string error_message(lpMsgBuf);
    return error_message;
}

}
