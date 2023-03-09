/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/err_common.hpp>

#define ONEAPI_NOT_SUPPORTED(message)                                       \
    do {                                                                    \
        throw SupportError(__AF_FUNC__, __AF_FILENAME__, __LINE__, message, \
                           boost::stacktrace::stacktrace());                \
    } while (0)

#define CL_CHECK(call)                                                      \
    do {                                                                    \
        if (cl_int err = (call)) {                                          \
            char cl_err_msg[2048];                                          \
            const char* cl_err_call = #call;                                \
            snprintf(cl_err_msg, sizeof(cl_err_msg),                        \
                     "CL Error %s(%d): %d = %s\n", __FILE__, __LINE__, err, \
                     cl_err_call);                                          \
            AF_ERROR(cl_err_msg, AF_ERR_INTERNAL);                          \
        }                                                                   \
    } while (0)

#define CL_CHECK_BUILD(call)                                                  \
    do {                                                                      \
        if (cl_int err = (call)) {                                            \
            char log[8192];                                                   \
            char cl_err_msg[8192];                                            \
            const char* cl_err_call = #call;                                  \
            size_t log_ret;                                                   \
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 8192, log, \
                                  &log_ret);                                  \
            snprintf(cl_err_msg, sizeof(cl_err_msg),                          \
                     "OpenCL Error building %s(%d): %d = %s\nLog:\n%s",       \
                     __FILE__, __LINE__, err, cl_err_call, log);              \
            AF_ERROR(cl_err_msg, AF_ERR_INTERNAL);                            \
        }                                                                     \
    } while (0)
