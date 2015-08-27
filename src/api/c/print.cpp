/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <af/array.h>
#include <af/data.h>
#include <copy.hpp>
#include <print.hpp>
#include <ArrayInfo.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <type_util.hpp>

#include <af/index.h>

using namespace detail;
using std::ostream;
using std::cout;
using std::endl;
using std::vector;

template<typename T>
static void printer(ostream &out, const T* ptr, const ArrayInfo &info, unsigned dim, const int precision)
{

    dim_t stride =   info.strides()[dim];
    dim_t d      =   info.dims()[dim];
    ToNum<T> toNum;

    if(dim == 0) {
        for(dim_t i = 0, j = 0; i < d; i++, j+=stride) {
            out<<   std::fixed <<
                    std::setw(precision + 6) <<
                    std::setprecision(precision) << toNum(ptr[j]) << " ";
        }
        out << endl;
    }
    else {
        for(dim_t i = 0; i < d; i++) {
            printer(out, ptr, info, dim - 1, precision);
            ptr += stride;
        }
        out << endl;
    }
}

template<typename T>
static void print(const char *exp, af_array arr, const int precision, std::ostream &os = std::cout, bool transpose = true)
{
    if(exp == NULL) {
        os << "No Name Array" << std::endl;
    } else {
        os << exp << std::endl;
    }

    const ArrayInfo info = getInfo(arr);
    vector<T> data(info.elements());

    af_array arrT;
    if(transpose) {
        AF_CHECK(af_reorder(&arrT, arr, 1, 0, 2, 3));
    } else {
        arrT = arr;
    }

    //FIXME: Use alternative function to avoid copies if possible
    AF_CHECK(af_get_data_ptr(&data.front(), arrT));
    const ArrayInfo infoT = getInfo(arrT);

    if(transpose) {
        AF_CHECK(af_release_array(arrT));
    }

    std::ios_base::fmtflags backup = os.flags();

    os << "[" << info.dims() << "]\n";
#ifndef NDEBUG
    os <<"   Offsets: [" << info.offsets() << "]" << std::endl;
    os <<"   Strides: [" << info.strides() << "]" << std::endl;
#endif

    printer(os, &data.front(), infoT, infoT.ndims() - 1, precision);

    os.flags(backup);
}

af_err af_print_array(af_array arr)
{
    try {
        ArrayInfo info = getInfo(arr);
        af_dtype type = info.getType();
        switch(type)
        {
        case f32:   print<float>   (NULL, arr, 4);   break;
        case c32:   print<cfloat>  (NULL, arr, 4);   break;
        case f64:   print<double>  (NULL, arr, 4);   break;
        case c64:   print<cdouble> (NULL, arr, 4);   break;
        case b8:    print<char>    (NULL, arr, 4);   break;
        case s32:   print<int>     (NULL, arr, 4);   break;
        case u32:   print<unsigned>(NULL, arr, 4);   break;
        case u8:    print<uchar>   (NULL, arr, 4);   break;
        case s64:   print<intl>    (NULL, arr, 4);   break;
        case u64:   print<uintl>   (NULL, arr, 4);   break;
        default:    TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_print_array_gen(const char *exp, const af_array arr, const int precision)
{
    try {
        ARG_ASSERT(0, exp != NULL);
        ArrayInfo info = getInfo(arr);
        af_dtype type = info.getType();
        switch(type)
        {
        case f32:   print<float   >(exp, arr, precision);   break;
        case c32:   print<cfloat  >(exp, arr, precision);   break;
        case f64:   print<double  >(exp, arr, precision);   break;
        case c64:   print<cdouble >(exp, arr, precision);   break;
        case b8:    print<char    >(exp, arr, precision);   break;
        case s32:   print<int     >(exp, arr, precision);   break;
        case u32:   print<unsigned>(exp, arr, precision);   break;
        case u8:    print<uchar   >(exp, arr, precision);   break;
        case s64:   print<intl    >(exp, arr, precision);   break;
        case u64:   print<uintl   >(exp, arr, precision);   break;
        default:    TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_array_to_string(char **output, const char *exp, const af_array arr,
                          const int precision, bool transpose)
{
    try {
        ARG_ASSERT(0, exp != NULL);
        ArrayInfo info = getInfo(arr);
        af_dtype type = info.getType();
        std::stringstream ss;
        switch(type)
        {
        case f32:   print<float   >(exp, arr, precision, ss, transpose);   break;
        case c32:   print<cfloat  >(exp, arr, precision, ss, transpose);   break;
        case f64:   print<double  >(exp, arr, precision, ss, transpose);   break;
        case c64:   print<cdouble >(exp, arr, precision, ss, transpose);   break;
        case b8:    print<char    >(exp, arr, precision, ss, transpose);   break;
        case s32:   print<int     >(exp, arr, precision, ss, transpose);   break;
        case u32:   print<unsigned>(exp, arr, precision, ss, transpose);   break;
        case u8:    print<uchar   >(exp, arr, precision, ss, transpose);   break;
        case s64:   print<intl    >(exp, arr, precision, ss, transpose);   break;
        case u64:   print<uintl   >(exp, arr, precision, ss, transpose);   break;
        default:    TYPE_ERROR(1, type);
        }
        std::string str = ss.str();
        *output = new char[str.size() + 1];
        std::copy(str.begin(), str.end(), *output);
        (*output)[str.size()] = '\0'; // don't forget the terminating 0
    }
    CATCHALL;
    return AF_SUCCESS;
}
