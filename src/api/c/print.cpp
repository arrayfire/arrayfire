/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <print.hpp>

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <sparse_handle.hpp>
#include <type_util.hpp>

#include <af/array.h>
#include <af/data.h>
#include <af/internal.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <af/index.h>

using arrayfire::getSparseArray;
using arrayfire::common::half;
using arrayfire::common::SparseArray;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::cout;
using std::endl;
using std::ostream;
using std::vector;

template<typename T>
static void printer(ostream &out, const T *ptr, const ArrayInfo &info,
                    unsigned dim, const int precision) {
    dim_t stride = info.strides()[dim];
    dim_t d      = info.dims()[dim];
    ToNum<T> toNum;
    using namespace detail;  // NOLINT

    if (dim == 0) {
        for (dim_t i = 0, j = 0; i < d; i++, j += stride) {
            out << std::fixed << std::setw(precision + 6)
                << std::setprecision(precision) << toNum(ptr[j]) << " ";
        }
        out << endl;
    } else {
        for (dim_t i = 0; i < d; i++) {
            printer(out, ptr, info, dim - 1, precision);
            ptr += stride;
        }
        out << endl;
    }
}

template<typename T>
static void print(const char *exp, af_array arr, const int precision,
                  std::ostream &os = std::cout, bool transpose = true) {
    if (exp == NULL) {
        os << "No Name Array" << std::endl;
    } else {
        os << exp << std::endl;
    }

    const ArrayInfo &info = getInfo(arr);

    std::ios_base::fmtflags backup = os.flags();

    os << "[" << info.dims() << "]\n";
#ifndef NDEBUG
    os << "   Offset: " << info.getOffset() << std::endl;
    os << "   Strides: [" << info.strides() << "]" << std::endl;
#endif

    // Handle empty array
    if (info.elements() == 0) {
        os << "<empty>" << std::endl;
        os.flags(backup);
        return;
    }

    vector<T> data(info.elements());

    af_array arrT;
    if (transpose) {
        AF_CHECK(af_reorder(&arrT, arr, 1, 0, 2, 3));
    } else {
        arrT = arr;
    }

    // FIXME: Use alternative function to avoid copies if possible
    AF_CHECK(af_get_data_ptr(&data.front(), arrT));
    const ArrayInfo &infoT = getInfo(arrT);

    printer(os, &data.front(), infoT, infoT.ndims() - 1, precision);

    if (transpose) { AF_CHECK(af_release_array(arrT)); }

    os.flags(backup);
}

template<typename T>
static void printSparse(const char *exp, af_array arr, const int precision,
                        std::ostream &os = std::cout, bool transpose = true) {
    SparseArray<T> sparse = getSparseArray<T>(arr);
    std::string name("No Name Sparse Array");

    if (exp != NULL) { name = std::string(exp); }
    os << name << std::endl;
    os << "Storage Format : ";
    switch (sparse.getStorage()) {
        case AF_STORAGE_DENSE: os << "AF_STORAGE_DENSE\n"; break;
        case AF_STORAGE_CSR: os << "AF_STORAGE_CSR\n"; break;
        case AF_STORAGE_CSC: os << "AF_STORAGE_CSC\n"; break;
        case AF_STORAGE_COO: os << "AF_STORAGE_COO\n"; break;
    }
    os << "[" << sparse.dims() << "]\n";

    print<T>(std::string(name + ": Values").c_str(),
             getHandle(sparse.getValues()), precision, os, transpose);
    print<int>(std::string(name + ": RowIdx").c_str(),
               getHandle(sparse.getRowIdx()), precision, os, transpose);
    print<int>(std::string(name + ": ColIdx").c_str(),
               getHandle(sparse.getColIdx()), precision, os, transpose);
}

af_err af_print_array(af_array arr) {
    try {
        const ArrayInfo &info =
            getInfo(arr, false);  // Don't assert sparse/dense
        af_dtype type = info.getType();

        if (info.isSparse()) {
            switch (type) {
                case f32: printSparse<float>(NULL, arr, 4); break;
                case f64: printSparse<double>(NULL, arr, 4); break;
                case c32: printSparse<cfloat>(NULL, arr, 4); break;
                case c64: printSparse<cdouble>(NULL, arr, 4); break;
                default: TYPE_ERROR(0, type);
            }
        } else {
            switch (type) {
                case f32: print<float>(NULL, arr, 4); break;
                case c32: print<cfloat>(NULL, arr, 4); break;
                case f64: print<double>(NULL, arr, 4); break;
                case c64: print<cdouble>(NULL, arr, 4); break;
                case b8: print<char>(NULL, arr, 4); break;
                case s32: print<int>(NULL, arr, 4); break;
                case u32: print<unsigned>(NULL, arr, 4); break;
                case u8: print<uchar>(NULL, arr, 4); break;
                case s64: print<intl>(NULL, arr, 4); break;
                case u64: print<uintl>(NULL, arr, 4); break;
                case s16: print<short>(NULL, arr, 4); break;
                case u16: print<ushort>(NULL, arr, 4); break;
                case f16: print<half>(NULL, arr, 4); break;
                default: TYPE_ERROR(1, type);
            }
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_print_array_gen(const char *exp, const af_array arr,
                          const int precision) {
    try {
        ARG_ASSERT(0, exp != NULL);
        const ArrayInfo &info =
            getInfo(arr, false);  // Don't assert sparse/dense
        af_dtype type = info.getType();

        if (info.isSparse()) {
            switch (type) {
                case f32: printSparse<float>(exp, arr, precision); break;
                case f64: printSparse<double>(exp, arr, precision); break;
                case c32: printSparse<cfloat>(exp, arr, precision); break;
                case c64: printSparse<cdouble>(exp, arr, precision); break;
                default: TYPE_ERROR(0, type);
            }
        } else {
            switch (type) {
                case f32: print<float>(exp, arr, precision); break;
                case c32: print<cfloat>(exp, arr, precision); break;
                case f64: print<double>(exp, arr, precision); break;
                case c64: print<cdouble>(exp, arr, precision); break;
                case b8: print<char>(exp, arr, precision); break;
                case s32: print<int>(exp, arr, precision); break;
                case u32: print<unsigned>(exp, arr, precision); break;
                case u8: print<uchar>(exp, arr, precision); break;
                case s64: print<intl>(exp, arr, precision); break;
                case u64: print<uintl>(exp, arr, precision); break;
                case s16: print<short>(exp, arr, precision); break;
                case u16: print<ushort>(exp, arr, precision); break;
                case f16: print<half>(exp, arr, precision); break;
                default: TYPE_ERROR(1, type);
            }
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_array_to_string(char **output, const char *exp, const af_array arr,
                          const int precision, bool transpose) {
    try {
        ARG_ASSERT(0, exp != NULL);
        const ArrayInfo &info =
            getInfo(arr, false);  // Don't assert sparse/dense
        af_dtype type = info.getType();
        std::stringstream ss;

        if (info.isSparse()) {
            switch (type) {
                case f32:
                    printSparse<float>(exp, arr, precision, ss, transpose);
                    break;
                case f64:
                    printSparse<double>(exp, arr, precision, ss, transpose);
                    break;
                case c32:
                    printSparse<cfloat>(exp, arr, precision, ss, transpose);
                    break;
                case c64:
                    printSparse<cdouble>(exp, arr, precision, ss, transpose);
                    break;
                default: TYPE_ERROR(0, type);
            }
        } else {
            switch (type) {
                case f32:
                    print<float>(exp, arr, precision, ss, transpose);
                    break;
                case c32:
                    print<cfloat>(exp, arr, precision, ss, transpose);
                    break;
                case f64:
                    print<double>(exp, arr, precision, ss, transpose);
                    break;
                case c64:
                    print<cdouble>(exp, arr, precision, ss, transpose);
                    break;
                case b8: print<char>(exp, arr, precision, ss, transpose); break;
                case s32: print<int>(exp, arr, precision, ss, transpose); break;
                case u32:
                    print<unsigned>(exp, arr, precision, ss, transpose);
                    break;
                case u8:
                    print<uchar>(exp, arr, precision, ss, transpose);
                    break;
                case s64:
                    print<intl>(exp, arr, precision, ss, transpose);
                    break;
                case u64:
                    print<uintl>(exp, arr, precision, ss, transpose);
                    break;
                case s16:
                    print<short>(exp, arr, precision, ss, transpose);
                    break;
                case u16:
                    print<ushort>(exp, arr, precision, ss, transpose);
                    break;
                case f16:
                    print<half>(exp, arr, precision, ss, transpose);
                    break;
                default: TYPE_ERROR(1, type);
            }
        }
        std::string str  = ss.str();
        void *halloc_ptr = nullptr;
        af_alloc_host(&halloc_ptr, sizeof(char) * (str.size() + 1));
        memcpy(output, &halloc_ptr, sizeof(void *));
        str.copy(*output, str.size());
        (*output)[str.size()] = '\0';  // don't forget the terminating 0
    }
    CATCHALL;
    return AF_SUCCESS;
}
