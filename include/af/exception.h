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

#include <iostream>
#include <af/defines.h>

namespace af {

class AFAPI exception : public std::exception
{
private:
    char m_msg[1024];
    af_err m_err;
public:
    af_err err() { return m_err; }
    exception();
    exception(const char *msg);
    exception(const char *file, unsigned line, af_err err);
    exception(const char *msg, const char *file, unsigned line, af_err err);
    virtual ~exception() throw() {}
    virtual const char *what() const throw() { return m_msg; }
    friend inline std::ostream& operator<<(std::ostream &s, const exception &e)
    { return s << e.what(); }
};

}

#endif

#ifdef __cplusplus
extern "C" {
#endif

AFAPI void af_get_last_error(char **msg, dim_t *len);
AFAPI const char *af_err_to_string(const af_err err);

#ifdef __cplusplus
}
#endif
