/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <common/util.hpp>
#include <type_util.hpp>
#include <af/device.h>
#include <af/exception.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>

#ifdef AF_OPENCL
#include <errorcodes.hpp>
#include <platform.hpp>
#endif

using boost::stacktrace::stacktrace;
using std::move;
using std::string;
using std::stringstream;

using arrayfire::common::getEnvVar;
using arrayfire::common::getName;
using arrayfire::common::is_stacktrace_enabled;

AfError::AfError(const char *const func, const char *const file, const int line,
                 const char *const message, af_err err, stacktrace st)
    : logic_error(message)
    , functionName(func)
    , fileName(file)
    , st_(move(st))
    , lineNumber(line)
    , error(err) {}

AfError::AfError(string func, string file, const int line,
                 const string &message, af_err err, stacktrace st)
    : logic_error(message)
    , functionName(move(func))
    , fileName(move(file))
    , st_(move(st))
    , lineNumber(line)
    , error(err) {}

const string &AfError::getFunctionName() const noexcept { return functionName; }

const string &AfError::getFileName() const noexcept { return fileName; }

int AfError::getLine() const noexcept { return lineNumber; }

af_err AfError::getError() const noexcept { return error; }

AfError::~AfError() noexcept = default;

TypeError::TypeError(const char *const func, const char *const file,
                     const int line, const int index, const af_dtype type,
                     stacktrace st)
    : AfError(func, file, line, "Invalid data type", AF_ERR_TYPE, move(st))
    , errTypeName(getName(type))
    , argIndex(index) {}

const string &TypeError::getTypeName() const noexcept { return errTypeName; }

int TypeError::getArgIndex() const noexcept { return argIndex; }

ArgumentError::ArgumentError(const char *const func, const char *const file,
                             const int line, const int index,
                             const char *const expectString, stacktrace st)
    : AfError(func, file, line, "Invalid argument", AF_ERR_ARG, move(st))
    , expected(expectString)
    , argIndex(index) {}

const string &ArgumentError::getExpectedCondition() const noexcept {
    return expected;
}

int ArgumentError::getArgIndex() const noexcept { return argIndex; }

SupportError::SupportError(const char *const func, const char *const file,
                           const int line, const char *const back,
                           stacktrace st)
    : AfError(func, file, line, "Unsupported Error", AF_ERR_NOT_SUPPORTED,
              move(st))
    , backend(back) {}

const string &SupportError::getBackendName() const noexcept { return backend; }

DimensionError::DimensionError(const char *const func, const char *const file,
                               const int line, const int index,
                               const char *const expectString,
                               const stacktrace &st)
    : AfError(func, file, line, "Invalid size", AF_ERR_SIZE, st)
    , expected(expectString)
    , argIndex(index) {}

const string &DimensionError::getExpectedCondition() const noexcept {
    return expected;
}

int DimensionError::getArgIndex() const noexcept { return argIndex; }

af_err set_global_error_string(const string &msg, af_err err) {
    string perr = getEnvVar("AF_PRINT_ERRORS");
    if (!perr.empty()) {
        if (perr != "0") { fprintf(stderr, "%s\n", msg.c_str()); }
    }
    get_global_error_string() = msg;
    return err;
}

af_err processException() {
    stringstream ss;
    af_err err = AF_ERR_INTERNAL;

    try {
        throw;
    } catch (const DimensionError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid dimension for argument " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";
        if (is_stacktrace_enabled()) { ss << ex.getStacktrace(); }

        err = set_global_error_string(ss.str(), AF_ERR_SIZE);
    } catch (const ArgumentError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid argument at index " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        if (is_stacktrace_enabled()) { ss << ex.getStacktrace(); }
        err = set_global_error_string(ss.str(), AF_ERR_ARG);
    } catch (const SupportError &ex) {
        ss << ex.getFunctionName() << " not supported for "
           << ex.getBackendName() << " backend\n";

        if (is_stacktrace_enabled()) { ss << ex.getStacktrace(); }
        err = set_global_error_string(ss.str(), AF_ERR_NOT_SUPPORTED);
    } catch (const TypeError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid type for argument " << ex.getArgIndex() << "\n";

        if (is_stacktrace_enabled()) { ss << ex.getStacktrace(); }
        err = set_global_error_string(ss.str(), AF_ERR_TYPE);
    } catch (const AfError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << ex.what() << "\n";
        if (is_stacktrace_enabled()) { ss << ex.getStacktrace(); }

        err = set_global_error_string(ss.str(), ex.getError());
#ifdef AF_OPENCL
    } catch (const cl::Error &ex) {
        char opencl_err_msg[1024];
        snprintf(opencl_err_msg, sizeof(opencl_err_msg),
                 "OpenCL Error (%d): %s when calling %s", ex.err(),
                 getErrorMessage(ex.err()).c_str(), ex.what());

        if (ex.err() == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
            err = set_global_error_string(opencl_err_msg, AF_ERR_NO_MEM);
        } else {
            err = set_global_error_string(opencl_err_msg, AF_ERR_INTERNAL);
        }
#endif
    } catch (...) { err = set_global_error_string(ss.str(), AF_ERR_UNKNOWN); }

    return err;
}

std::string &get_global_error_string() noexcept {
    thread_local auto *global_error_string = new std::string("");
    return *global_error_string;
}

const char *af_err_to_string(const af_err err) {
    switch (err) {
        case AF_SUCCESS: return "Success";
        case AF_ERR_NO_MEM: return "Device out of memory";
        case AF_ERR_DRIVER: return "Driver not available or incompatible";
        case AF_ERR_RUNTIME: return "Runtime error ";
        case AF_ERR_INVALID_ARRAY: return "Invalid array";
        case AF_ERR_ARG: return "Invalid input argument";
        case AF_ERR_SIZE: return "Invalid input size";
        case AF_ERR_TYPE: return "Function does not support this data type";
        case AF_ERR_DIFF_TYPE: return "Input types are not the same";
        case AF_ERR_BATCH: return "Invalid batch configuration";
        case AF_ERR_DEVICE:
            return "Input does not belong to the current device.";
        case AF_ERR_NOT_SUPPORTED: return "Function not supported";
        case AF_ERR_NOT_CONFIGURED: return "Function not configured to build";
        case AF_ERR_NONFREE:
            return "Function unavailable. "
                   "ArrayFire compiled without Non-Free algorithms support";
        case AF_ERR_NO_DBL:
            return "Double precision not supported for this device";
        case AF_ERR_NO_GFX:
            return "Graphics functionality unavailable. "
                   "ArrayFire compiled without Graphics support";
        case AF_ERR_NO_HALF:
            return "Half precision floats not supported for this device";
        case AF_ERR_LOAD_LIB: return "Failed to load dynamic library. ";
        case AF_ERR_LOAD_SYM: return "Failed to load symbol";
        case AF_ERR_ARR_BKND_MISMATCH:
            return "There was a mismatch between an array and the current "
                   "backend";
        case AF_ERR_INTERNAL: return "Internal error";
        case AF_ERR_UNKNOWN: return "Unknown error";
    }
    return "Unknown error. Please open an issue and add this error code to the "
           "case in af_err_to_string.";
}

namespace arrayfire {
namespace common {

bool &is_stacktrace_enabled() noexcept {
    static bool stacktrace_enabled = true;
    return stacktrace_enabled;
}

}  // namespace common
}  // namespace arrayfire
