/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/// This file contains platform independent utility functions
#include <string>
#include <cstdlib>

#if defined(OS_WIN)
#include <Windows.h>
#endif

#include <af/defines.h>

using std::string;

string getEnvVar(const std::string &key)
{
#if defined(OS_WIN)
    DWORD bufSize = 32767; // limit according to GetEnvironment Variable documentation
    string retVal;
    retVal.resize(bufSize);
    bufSize = GetEnvironmentVariable(key.c_str(), &retVal[0], bufSize);
    if (!bufSize) {
        return string("");
    } else {
        retVal.resize(bufSize);
        return retVal;
    }
#else
    char * str = getenv(key.c_str());
    return str==NULL ? string("") : string(str);
#endif
}

const char *getName(af_dtype type)
{
  switch(type) {
  case f32: return "float";
  case f64: return "double";
  case c32: return "complex float";
  case c64: return "complex double";
  case u32: return "unsigned int";
  case s32: return "int";
  case u16: return "unsigned short";
  case s16: return "short";
  case u64: return "unsigned long long";
  case s64: return "long long";
  case u8 : return "unsigned char";
  case b8 : return "bool";
  default : return "unknown type";
  }
}
