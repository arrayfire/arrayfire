/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/exception.h>
#include <algorithm>
#include <cstdio>
#include <cstring>  // strncpy

#ifdef OS_WIN
#define snprintf _snprintf
#endif

namespace af {

exception::exception() : m_msg{}, m_err(AF_ERR_UNKNOWN) {
    strncpy(m_msg, "unknown exception", sizeof(m_msg));
}

exception::exception(const char *msg) : m_msg{}, m_err(AF_ERR_UNKNOWN) {
    strncpy(m_msg, msg, sizeof(m_msg) - 1);
    m_msg[sizeof(m_msg) - 1] = '\0';
}

exception::exception(const char *file, unsigned line, af_err err)
    : m_msg{}, m_err(err) {
    snprintf(m_msg, sizeof(m_msg) - 1, "ArrayFire Exception (%s:%d):\nIn %s:%u",
             af_err_to_string(err), static_cast<int>(err), file, line);

    m_msg[sizeof(m_msg) - 1] = '\0';
}

exception::exception(const char *msg, const char *file, unsigned line,
                     af_err err)
    : m_msg{}, m_err(err) {
    snprintf(m_msg, sizeof(m_msg) - 1,
             "ArrayFire Exception (%s:%d):\n%s\nIn %s:%u",
             af_err_to_string(err), static_cast<int>(err), msg, file, line);

    m_msg[sizeof(m_msg) - 1] = '\0';
}

exception::exception(const char *msg, const char *func, const char *file,
                     unsigned line, af_err err)
    : m_msg{}, m_err(err) {
    snprintf(m_msg, sizeof(m_msg) - 1,
             "ArrayFire Exception (%s:%d):\n%s\nIn function %s\nIn file %s:%u",
             af_err_to_string(err), static_cast<int>(err), msg, func, file,
             line);

    m_msg[sizeof(m_msg) - 1] = '\0';
}

}  // namespace af
