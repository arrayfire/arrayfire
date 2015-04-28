/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_common.hpp>
#include <type_util.hpp>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdio>

using std::string;
using std::stringstream;

AfError::AfError(const char * const funcName,
                 const int line,
                 const char * const message, af_err err)
    : logic_error   (message),
      functionName  (funcName),
      lineNumber(line),
      error(err)
{}

AfError::AfError(string funcName,
                 const int line,
                 string message, af_err err)
    : logic_error   (message),
      functionName  (funcName),
      lineNumber(line),
      error(err)
{}

const string&
AfError::getFunctionName() const
{
    return functionName;
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

TypeError::TypeError(const char * const  funcName,
                     const int line,
                     const int index, const af_dtype type)
    : AfError (funcName, line, "Invalid data type", AF_ERR_INVALID_TYPE),
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

ArgumentError::ArgumentError(const char * const  funcName,
                             const int line,
                             const int index,
                             const char * const  expectString)
    : AfError(funcName, line, "Invalid argument", AF_ERR_INVALID_ARG),
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


SupportError::SupportError(const char * const funcName,
                           const int line,
                           const char * const back)
    : AfError(funcName, line, "Unsupported Error", AF_ERR_NOT_SUPPORTED),
      backend(back)
{}

const string& SupportError::getBackendName() const
{
    return backend;
}

DimensionError::DimensionError(const char * const  funcName,
                             const int line,
                             const int index,
                             const char * const  expectString)
    : AfError(funcName, line, "Invalid dimension", AF_ERR_SIZE),
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
print_error(const stringstream &msg)
{
    const char* perr = getenv("AF_PRINT_ERRORS");
    if(perr != nullptr) {
        if(std::strncmp(perr, "0", 1) != 0)
            fprintf(stderr, "%s\n", msg.str().c_str());
    }
}

af_err processException()
{
    stringstream    ss;
    af_err          err= AF_ERR_INTERNAL;

    try {
        throw;
    } catch (const DimensionError &ex) {
        ss << "In function " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << "Invalid dimension for argument " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        print_error(ss);
        err = AF_ERR_SIZE;
    } catch (const ArgumentError &ex) {
        ss << "In function " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << "Invalid argument at index " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        print_error(ss);
        err = AF_ERR_ARG;
    } catch (const SupportError &ex) {
        ss << ex.getFunctionName()
           << " not supported for " << ex.getBackendName()
           << " backend\n";

        print_error(ss);
        err = AF_ERR_NOT_SUPPORTED;
    } catch (const TypeError &ex) {
        ss << "In function " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << "Invalid type for argument " << ex.getArgIndex() << "\n";

        print_error(ss);
        err = AF_ERR_INVALID_TYPE;
    } catch (const AfError &ex) {
        ss << "Error in " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << ex.what() << "\n";

        print_error(ss);
        err = ex.getError();
    } catch (...) {
        print_error(ss);
        err = AF_ERR_UNKNOWN;
    }

    return err;
}
