/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifdef __cplusplus

#include <ostream>
#include <af/defines.h>

namespace af {

/// An ArrayFire exception class
/// \ingroup arrayfire_class
class AFAPI exception : public std::exception
{
private:
    char m_msg[1024];
    af_err m_err;
public:
    af_err err() { return m_err; }
    exception();
    /// Creates a new af::exception given a message. The error code is AF_ERR_UNKNOWN
    exception(const char *msg);

    /// Creates a new exception with a formatted error message for a given file
    /// and line number in the source code.
    exception(const char *file, unsigned line, af_err err);

    /// Creates a new af::exception with a formatted error message for a given
    /// an error code, file and line number in the source code.
    exception(const char *msg, const char *file, unsigned line, af_err err);
#if AF_API_VERSION >= 33
    /// Creates a new exception given a message, function name, file name, line number and
    /// error code.
    exception(const char *msg, const char *func, const char *file, unsigned line, af_err err);
#endif
    virtual ~exception() throw() {}
    /// Returns an error message for the exception in a string format
    virtual const char *what() const throw() { return m_msg; }

    /// Writes the exception to a stream
    friend inline std::ostream& operator<<(std::ostream &s, const exception &e)
    { return s << e.what(); }
};

}  // namespace af

#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Returns the last error message that occurred and its error message
///
/// \param[out] msg The message of the previous error
/// \param[out] len The number of characters in the msg object
AFAPI void af_get_last_error(char **msg, dim_t *len);

/// Converts the af_err error code to its string representation
///
/// \param[in] err The ArrayFire error code
AFAPI const char *af_err_to_string(const af_err err);

#ifdef __cplusplus
}
#endif
