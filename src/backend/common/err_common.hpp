/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>
#include <common/defines.hpp>

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class AfError   : public std::logic_error
{
    std::string functionName;
    std::string fileName;
    int lineNumber;
    af_err error;
    AfError();

public:

    AfError(const char * const func,
            const char * const file,
            const int line,
            const char * const message, af_err err);

    AfError(std::string func,
            std::string file,
            const int line,
            std::string message, af_err err);

    const std::string&
    getFunctionName() const;

    const std::string&
    getFileName() const;

    int getLine() const;

    af_err getError() const;

    virtual ~AfError() throw();
};

// TODO: Perhaps add a way to return supported types
class TypeError : public AfError
{
    int argIndex;
    std::string errTypeName;
    TypeError();

public:

    TypeError(const char * const func,
              const char * const file,
              const int line,
              const int index,
              const af_dtype type);

    const std::string&
    getTypeName() const;

    int getArgIndex() const;

    ~TypeError() throw() {}
};

class ArgumentError : public AfError
{
    int argIndex;
    std::string expected;
    ArgumentError();

public:

    ArgumentError(const char * const func,
                  const char * const file,
                  const int line,
                  const int index,
                  const char * const expectString);

    const std::string&
    getExpectedCondition() const;

    int getArgIndex() const;

    ~ArgumentError() throw(){}
};

class SupportError  :   public AfError
{
    std::string backend;
    SupportError();

public:

    SupportError(const char * const func,
                 const char * const file,
                 const int line,
                 const char * const back);

    ~SupportError()throw() {}

    const std::string&
    getBackendName() const;
};

class DimensionError : public AfError
{
    int argIndex;
    std::string expected;
    DimensionError();

public:

    DimensionError(const char * const func,
                   const char * const file,
                   const int line,
                   const int index,
                   const char * const expectString);

    const std::string&
    getExpectedCondition() const;

    int getArgIndex() const;

    ~DimensionError() throw(){}
};

af_err processException();

void print_error(const std::string &msg);

#define DIM_ASSERT(INDEX, COND) do {                        \
        if((COND) == false) {                               \
            throw DimensionError(__PRETTY_FUNCTION__,       \
                                 __AF_FILENAME__, __LINE__, \
                                 INDEX, #COND);             \
        }                                                   \
    } while(0)

#define ARG_ASSERT(INDEX, COND) do {                        \
        if((COND) == false) {                               \
            throw ArgumentError(__PRETTY_FUNCTION__,        \
                                __AF_FILENAME__, __LINE__,  \
                                INDEX, #COND);              \
        }                                                   \
    } while(0)

#define TYPE_ERROR(INDEX, type) do {                        \
        throw TypeError(__PRETTY_FUNCTION__,                \
                        __AF_FILENAME__, __LINE__,          \
                        INDEX, type);                       \
    } while(0)                                              \


#define AF_ERROR(MSG, ERR_TYPE) do {                        \
        throw AfError(__PRETTY_FUNCTION__,                  \
                      __AF_FILENAME__, __LINE__,            \
                      MSG, ERR_TYPE);                       \
    } while(0)

#define AF_RETURN_ERROR(MSG, ERR_TYPE) do {                 \
        AfError err(__PRETTY_FUNCTION__,                    \
                    __AF_FILENAME__, __LINE__,              \
                    MSG, ERR_TYPE);                         \
        std::stringstream s;                                \
        s << "Error in " << err.getFunctionName() << "\n"   \
          << "In file " << err.getFileName()                \
          << ":" << err.getLine()  << "\n"                  \
          << err.what() << "\n";                            \
        print_error(s.str());                               \
        return ERR_TYPE;                                    \
    } while(0)

#define TYPE_ASSERT(COND) do {                              \
        if ((COND) == false) {                              \
            AF_ERROR("Type mismatch inputs",                \
                     AF_ERR_DIFF_TYPE);                     \
        }                                                   \
    } while(0)

#define AF_ASSERT(COND, MESSAGE)                            \
    assert(MESSAGE && COND)

#define CATCHALL                                            \
    catch(...) {                                            \
        return processException();                          \
    }

#define AF_CHECK(fn) do {                                   \
        af_err __err = fn;                                  \
        if (__err == AF_SUCCESS) break;                     \
        throw AfError(__PRETTY_FUNCTION__,                  \
                      __AF_FILENAME__, __LINE__,            \
                      "\n", __err);                         \
    } while(0)

static const int MAX_ERR_SIZE = 1024;
std::string& get_global_error_string();
