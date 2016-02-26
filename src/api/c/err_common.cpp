/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/exception.h>
#include <af/device.h>
#include <err_common.hpp>
#include <type_util.hpp>
#include <util.hpp>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <algorithm>

#if defined(WITH_GRAPHICS) && !defined(AF_UNIFIED)
#include <graphics_common.hpp>
#endif

using std::string;
using std::stringstream;

AfError::AfError(const char * const func,
                 const char * const file,
                 const int line,
                 const char * const message, af_err err)
    : logic_error   (message),
      functionName  (func),
      fileName      (file),
      lineNumber(line),
      error(err)
{}

AfError::AfError(string func,
                 string file,
                 const int line,
                 string message, af_err err)
    : logic_error   (message),
      functionName  (func),
      fileName      (file),
      lineNumber(line),
      error(err)
{}

const string&
AfError::getFunctionName() const
{
    return functionName;
}

const string&
AfError::getFileName() const
{
    return fileName;
}

int
AfError::getLine() const
{
    return lineNumber;
}

af_err
AfError::getError() const
{
    return error;
}

AfError::~AfError() throw() {}

TypeError::TypeError(const char * const func,
                     const char * const file,
                     const int line,
                     const int index, const af_dtype type)
    : AfError (func, file, line, "Invalid data type", AF_ERR_TYPE),
      argIndex(index),
      errTypeName(getName(type))
{}

const string& TypeError::getTypeName() const
{
    return errTypeName;
}

int TypeError::getArgIndex() const
{
    return argIndex;
}

ArgumentError::ArgumentError(const char * const func,
                             const char * const file,
                             const int line,
                             const int index,
                             const char * const  expectString)
    : AfError(func, file, line, "Invalid argument", AF_ERR_ARG),
      argIndex(index),
      expected(expectString)
{

}

const string& ArgumentError::getExpectedCondition() const
{
    return expected;
}

int ArgumentError::getArgIndex() const
{
    return argIndex;
}


SupportError::SupportError(const char * const func,
                           const char * const file,
                           const int line,
                           const char * const back)
    : AfError(func, file, line, "Unsupported Error", AF_ERR_NOT_SUPPORTED),
      backend(back)
{}

const string& SupportError::getBackendName() const
{
    return backend;
}

DimensionError::DimensionError(const char * const  func,
                               const char * const file,
                               const int line,
                               const int index,
                               const char * const  expectString)
    : AfError(func, file, line, "Invalid size", AF_ERR_SIZE),
      argIndex(index),
      expected(expectString)
{

}

const string& DimensionError::getExpectedCondition() const
{
    return expected;
}

int DimensionError::getArgIndex() const
{
    return argIndex;
}

void
print_error(const string &msg)
{
    std::string perr = getEnvVar("AF_PRINT_ERRORS");
    if(!perr.empty()) {
        if(perr != "0")
            fprintf(stderr, "%s\n", msg.c_str());
    }
    get_global_error_string() = msg;
}

af_err processException()
{
    stringstream    ss;
    af_err          err= AF_ERR_INTERNAL;

    try {
        throw;
    } catch (const DimensionError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid dimension for argument " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        print_error(ss.str());
        err = AF_ERR_SIZE;
    } catch (const ArgumentError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid argument at index " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        print_error(ss.str());
        err = AF_ERR_ARG;
    } catch (const SupportError &ex) {
        ss << ex.getFunctionName()
           << " not supported for " << ex.getBackendName()
           << " backend\n";

        print_error(ss.str());
        err = AF_ERR_NOT_SUPPORTED;
    } catch (const TypeError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid type for argument " << ex.getArgIndex() << "\n";

        print_error(ss.str());
        err = AF_ERR_TYPE;
    } catch (const AfError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << ex.what() << "\n";

        print_error(ss.str());
        err = ex.getError();
#if defined(WITH_GRAPHICS) && !defined(AF_UNIFIED)
    } catch (const fg::Error &ex) {
        ss << ex << "\n";
        print_error(ss.str());
        err = AF_ERR_INTERNAL;
#endif
    } catch (...) {
        print_error(ss.str());
        err = AF_ERR_UNKNOWN;
    }

    return err;
}

std::string& get_global_error_string()
{
    static std::string global_error_string = std::string("");
    return global_error_string;
}
