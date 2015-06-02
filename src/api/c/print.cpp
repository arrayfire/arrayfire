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
static void printer(ostream &out, const T* ptr, const ArrayInfo &info, unsigned dim)
{

    dim_t stride =   info.strides()[dim];
    dim_t d      =   info.dims()[dim];
    ToNum<T> toNum;

    if(dim == 0) {
        for(dim_t i = 0, j = 0; i < d; i++, j+=stride) {
            out<<   std::fixed <<
                    std::setw(10) <<
                    std::setprecision(4) << toNum(ptr[j]) << " ";
        }
        out << endl;
    }
    else {
        for(dim_t i = 0; i < d; i++) {
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
    vector<T> data(info.elements());

    af_array arrT;
    AF_CHECK(af_reorder(&arrT, arr, 1, 0, 2, 3));

    //FIXME: Use alternative function to avoid copies if possible
    AF_CHECK(af_get_data_ptr(&data.front(), arrT));
    const ArrayInfo infoT = getInfo(arrT);
    AF_CHECK(af_release_array(arrT));

    std::ios_base::fmtflags backup = std::cout.flags();

    std::cout << "[" << info.dims() << "]\n";
#ifndef NDEBUG
    std::cout <<"   Offsets: ["<<info.offsets()<<"]"<<std::endl;
    std::cout <<"   Strides: ["<<info.strides()<<"]"<<std::endl;
#endif

    printer(std::cout, &data.front(), infoT, infoT.ndims() - 1);

    std::cout.flags(backup);
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
        case s64:   print<intl>(arr);     break;
        case u64:   print<uintl>(arr);    break;
        default:    TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
