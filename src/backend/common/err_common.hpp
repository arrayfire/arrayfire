/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#pragma GCC diagnostic ignored "-Wparentheses"
#include <boost/stacktrace.hpp>
#pragma GCC diagnostic pop
#include <common/defines.hpp>
#include <af/defines.h>

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

class AfError : public std::logic_error {
    std::string functionName;
    std::string fileName;
    boost::stacktrace::stacktrace st_;
    int lineNumber;
    af_err error;
    AfError();

   public:
    AfError(const char* const func, const char* const file, const int line,
            const char* const message, af_err err,
            boost::stacktrace::stacktrace st);

    AfError(std::string func, std::string file, const int line,
            const std::string& message, af_err err,
            boost::stacktrace::stacktrace st);

    AfError(const AfError& other) noexcept = delete;

    /// This is the same as default but gcc 6.1 fails when noexcept is used
    /// along with the default specifier. Expanded the default definition
    /// to avoid this error
    AfError(AfError&& other) noexcept
        : std::logic_error(std::forward<std::logic_error>(other))
        , functionName(std::forward<std::string>(other.functionName))
        , fileName(std::forward<std::string>(other.fileName))
        , st_(std::forward<boost::stacktrace::stacktrace>(other.st_))
        , lineNumber(std::forward<int>(other.lineNumber))
        , error(std::forward<af_err>(other.error)) {}

    const std::string& getFunctionName() const noexcept;

    const std::string& getFileName() const noexcept;

    const boost::stacktrace::stacktrace& getStacktrace() const noexcept {
        return st_;
    };

    int getLine() const noexcept;

    af_err getError() const noexcept;

    virtual ~AfError() noexcept;
};

// TODO: Perhaps add a way to return supported types
class TypeError : public AfError {
    std::string errTypeName;
    int argIndex;
    TypeError();

   public:
    TypeError(const char* const func, const char* const file, const int line,
              const int index, const af_dtype type,
              const boost::stacktrace::stacktrace st);

    TypeError(TypeError&& other) noexcept = default;

    const std::string& getTypeName() const noexcept;

    int getArgIndex() const noexcept;

    ~TypeError() noexcept {}
};

class ArgumentError : public AfError {
    std::string expected;
    int argIndex;
    ArgumentError();

   public:
    ArgumentError(const char* const func, const char* const file,
                  const int line, const int index,
                  const char* const expectString,
                  const boost::stacktrace::stacktrace st);
    ArgumentError(ArgumentError&& other) noexcept = default;

    const std::string& getExpectedCondition() const noexcept;

    int getArgIndex() const noexcept;

    ~ArgumentError() noexcept {}
};

class SupportError : public AfError {
    std::string backend;
    SupportError();

   public:
    SupportError(const char* const func, const char* const file, const int line,
                 const char* const back,
                 const boost::stacktrace::stacktrace st);
    SupportError(SupportError&& other) noexcept = default;

    ~SupportError() noexcept {}

    const std::string& getBackendName() const noexcept;
};

class DimensionError : public AfError {
    std::string expected;
    int argIndex;
    DimensionError();

   public:
    DimensionError(const char* const func, const char* const file,
                   const int line, const int index,
                   const char* const expectString,
                   const boost::stacktrace::stacktrace& st);
    DimensionError(DimensionError&& other) noexcept = default;

    const std::string& getExpectedCondition() const noexcept;

    int getArgIndex() const noexcept;

    ~DimensionError() noexcept {}
};

af_err processException();

af_err set_global_error_string(const std::string& msg,
                               af_err err = AF_ERR_UNKNOWN);

#define DIM_ASSERT(INDEX, COND)                                          \
    do {                                                                 \
        if ((COND) == false) {                                           \
            throw DimensionError(__AF_FUNC__, __AF_FILENAME__, __LINE__, \
                                 INDEX, #COND,                           \
                                 boost::stacktrace::stacktrace());       \
        }                                                                \
    } while (0)

#define ARG_ASSERT(INDEX, COND)                                                \
    do {                                                                       \
        if ((COND) == false) {                                                 \
            throw ArgumentError(__AF_FUNC__, __AF_FILENAME__, __LINE__, INDEX, \
                                #COND, boost::stacktrace::stacktrace());       \
        }                                                                      \
    } while (0)

#define TYPE_ERROR(INDEX, type)                                              \
    do {                                                                     \
        throw TypeError(__AF_FUNC__, __AF_FILENAME__, __LINE__, INDEX, type, \
                        boost::stacktrace::stacktrace());                    \
    } while (0)

#define AF_ERROR(MSG, ERR_TYPE)                                              \
    do {                                                                     \
        throw AfError(__AF_FUNC__, __AF_FILENAME__, __LINE__, MSG, ERR_TYPE, \
                      boost::stacktrace::stacktrace());                      \
    } while (0)

#define AF_RETURN_ERROR(MSG, ERR_TYPE)                                       \
    do {                                                                     \
        std::stringstream s;                                                 \
        s << "Error in " << __AF_FUNC__ << "\n"                              \
          << "In file " << __AF_FILENAME__ << ":" << __LINE__ << ": " << MSG \
          << "\n"                                                            \
          << boost::stacktrace::stacktrace();                                \
        return set_global_error_string(s.str(), ERR_TYPE);                   \
    } while (0)

#define TYPE_ASSERT(COND)                                       \
    do {                                                        \
        if ((COND) == false) {                                  \
            AF_ERROR("Type mismatch inputs", AF_ERR_DIFF_TYPE); \
        }                                                       \
    } while (0)

#define AF_ASSERT(COND, MESSAGE) assert(MESSAGE&& COND)

#define CATCHALL                   \
    catch (...) {                  \
        return processException(); \
    }

#define AF_CHECK(fn)                                                       \
    do {                                                                   \
        af_err __err = fn;                                                 \
        if (__err == AF_SUCCESS) break;                                    \
        throw AfError(__AF_FUNC__, __AF_FILENAME__, __LINE__, "\n", __err, \
                      boost::stacktrace::stacktrace());                    \
    } while (0)

static const int MAX_ERR_SIZE = 1024;
std::string& get_global_error_string() noexcept;

namespace arrayfire {
namespace common {

bool& is_stacktrace_enabled() noexcept;

}  // namespace common
}  // namespace arrayfire
