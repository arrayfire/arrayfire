/*******************************************************
* Copyright (c) 2014, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include "types.hpp"

namespace cuda
{
	template<typename T > const char *shortname(bool caps) { return caps ? "X" : "x"; }
	template<> const char *shortname<float   >(bool caps) { return caps ? "S" : "s"; }
	template<> const char *shortname<double  >(bool caps) { return caps ? "D" : "d"; }
	template<> const char *shortname<cfloat  >(bool caps) { return caps ? "C" : "c"; }
	template<> const char *shortname<cdouble >(bool caps) { return caps ? "Z" : "z"; }
	template<> const char *shortname<int     >(bool caps) { return caps ? "I" : "i"; }
	template<> const char *shortname<uint    >(bool caps) { return caps ? "U" : "u"; }
	template<> const char *shortname<char    >(bool caps) { return caps ? "J" : "j"; }
	template<> const char *shortname<uchar   >(bool caps) { return caps ? "V" : "v"; }

	template<typename T > const char *irname() { return  "i32"; }
	template<> const char *irname<float   >() { return  "float"; }
	template<> const char *irname<double  >() { return  "double"; }
	template<> const char *irname<cfloat  >() { return  "<2 x float>"; }
	template<> const char *irname<cdouble >() { return  "<2 x double>"; }
	template<> const char *irname<int     >() { return  "i32"; }
	template<> const char *irname<uint    >() { return  "i32"; }
	template<> const char *irname<char    >() { return  "i8"; }
	template<> const char *irname<uchar   >() { return  "i8"; }
}