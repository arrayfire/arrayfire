/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <iostream>
#include <af/array.h>
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

template<typename T>
static void printer(ostream &out, const T* ptr, const ArrayInfo &info, unsigned dim)
{

    dim_type stride =   info.strides()[dim];
    dim_type d      =   info.dims()[dim];
    ToNum<T> toNum;

    if(dim == 0) {
        for(dim_type i = 0, j = 0; i < d; i++, j+=stride) {
            out << toNum(ptr[j]) << "\t";
        }
        out << endl;
    }
    else {
        for(dim_type i = 0; i < d; i++) {
            printer(out, ptr, info, dim - 1);
            ptr += stride;
        }
        out << endl;
    }
}

template<typename T>
static void print(af_array arr)
{
    const ArrayInfo info = getInfo(arr);
    T *data = new T[info.elements()];

    af_array arrT;
    af_reorder(&arrT, arr, 1, 0, 2, 3);

    //FIXME: Use alternative function to avoid copies if possible
    af_get_data_ptr(data, arrT);
    const ArrayInfo infoT = getInfo(arrT);

    //std::cout << "TRANSPOSED\n";
    std::cout << "Dim:" << info.dims();
    std::cout << "Offset: " << info.offsets();
    std::cout << "Stride: " << info.strides();

    printer(std::cout, data, infoT, infoT.ndims() - 1);

    delete[] data;
}

af_err af_print_array(af_array arr)
{
    try {
        ArrayInfo info = getInfo(arr);
        af_dtype type = info.getType();
        switch(type)
        {
        case f32:   print<float>(arr);    break;
        case c32:   print<cfloat>(arr);   break;
        case f64:   print<double>(arr);   break;
        case c64:   print<cdouble>(arr);  break;
        case b8:    print<char>(arr);     break;
        case s32:   print<int>(arr);      break;
        case u32:   print<unsigned>(arr); break;
        case u8:    print<uchar>(arr);    break;
        case s8:    print<char>(arr);     break;
        default:    TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
