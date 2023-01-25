/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <backend.hpp>

#include <arith.hpp>
#include <blas.hpp>
#include <common/ArrayInfo.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/moddims.hpp>
#include <diagonal.hpp>
#include <handle.hpp>
#include <logic.hpp>
#include <math.hpp>
#include <reduce.hpp>
#include <select.hpp>
#include <svd.hpp>
#include <tile.hpp>
#include <transpose.hpp>
#include <af/array.h>
#include <af/complex.h>
#include <af/defines.h>
#include <af/lapack.h>

using af::dim4;
using af::dtype_traits;
using arrayfire::common::cast;
using arrayfire::common::modDims;
using detail::arithOp;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::createSelectNode;
using detail::createSubArray;
using detail::createValueArray;
using detail::diagCreate;
using detail::gemm;
using detail::logicOp;
using detail::max;
using detail::min;
using detail::reduce;
using detail::scalar;
using detail::svd;
using detail::tile;
using detail::uint;
using std::swap;
using std::vector;

template<typename T>
Array<T> getSubArray(const Array<T> &in, const bool copy, uint dim0begin = 0,
                     uint dim0end = 0, uint dim1begin = 0, uint dim1end = 0,
                     uint dim2begin = 0, uint dim2end = 0, uint dim3begin = 0,
                     uint dim3end = 0) {
    vector<af_seq> seqs = {
        {static_cast<double>(dim0begin), static_cast<double>(dim0end), 1.},
        {static_cast<double>(dim1begin), static_cast<double>(dim1end), 1.},
        {static_cast<double>(dim2begin), static_cast<double>(dim2end), 1.},
        {static_cast<double>(dim3begin), static_cast<double>(dim3end), 1.}};
    return createSubArray<T>(in, seqs, copy);
}

// Moore-Penrose Pseudoinverse
template<typename T>
Array<T> pinverseSvd(const Array<T> &in, const double tol) {
    in.eval();
    dim_t M = in.dims()[0];
    dim_t N = in.dims()[1];
    dim_t P = in.dims()[2];
    dim_t Q = in.dims()[3];

    // Compute SVD
    using Tr = typename dtype_traits<T>::base_type;
    // Ideally, these initializations should use createEmptyArray(), but for
    // some reason, linux-opencl-k80 will produce wrong results for large arrays
    Array<T> u  = createValueArray<T>(dim4(M, M, P, Q), scalar<T>(0));
    Array<T> vT = createValueArray<T>(dim4(N, N, P, Q), scalar<T>(0));
    Array<Tr> sVec =
        createValueArray<Tr>(dim4(min(M, N), 1, P, Q), scalar<Tr>(0));
    for (dim_t j = 0; j < Q; ++j) {
        for (dim_t i = 0; i < P; ++i) {
            Array<T> inSlice =
                getSubArray(in, false, 0, M - 1, 0, N - 1, i, i, j, j);
            Array<Tr> sVecSlice = getSubArray(
                sVec, false, 0, sVec.dims()[0] - 1, 0, 0, i, i, j, j);
            Array<T> uSlice  = getSubArray(u, false, 0, u.dims()[0] - 1, 0,
                                           u.dims()[1] - 1, i, i, j, j);
            Array<T> vTSlice = getSubArray(vT, false, 0, vT.dims()[0] - 1, 0,
                                           vT.dims()[1] - 1, i, i, j, j);
            svd<T, Tr>(sVecSlice, uSlice, vTSlice, inSlice);
        }
    }

    // Cast s back to original data type for matmul later
    // (since svd() makes s' type the base type of T)
    Array<T> sVecCast = cast<T, Tr>(sVec);

    Array<T> v = transpose(vT, true);

    // Build relative tolerance array
    Array<Tr> sVecMax    = reduce<af_max_t, Tr, Tr>(sVec, 0);
    Array<T> sVecMaxCast = cast<T, Tr>(sVecMax);
    double tolMulShape   = tol * static_cast<double>(max(M, N));
    Array<T> tolMulShapeArr =
        createValueArray<T>(sVecMaxCast.dims(), scalar<T>(tolMulShape));
    Array<T> relTol =
        arithOp<T, af_mul_t>(tolMulShapeArr, sVecMaxCast, sVecMaxCast.dims());
    Array<T> relTolArr = tile<T>(relTol, dim4(sVecCast.dims()[0]));

    // Get reciprocal of sVec's non-zero values for s pinverse, except for
    // very small non-zero values though (< relTol), in order to avoid very
    // large reciprocals
    Array<T> ones      = createValueArray<T>(sVecCast.dims(), scalar<T>(1.));
    Array<T> sVecRecip = arithOp<T, af_div_t>(ones, sVecCast, sVecCast.dims());
    Array<char> cond =
        logicOp<T, af_ge_t>(sVecCast, relTolArr, sVecCast.dims());
    Array<T> zeros = createValueArray<T>(sVecCast.dims(), scalar<T>(0.));
    sVecRecip = createSelectNode<T>(cond, sVecRecip, zeros, sVecRecip.dims());

    // Make s vector into s pinverse array
    Array<T> sVecRecipMod = modDims<T>(
        sVecRecip,
        dim4(sVecRecip.dims()[0], (sVecRecip.dims()[2] * sVecRecip.dims()[3])));
    Array<T> sPinv = diagCreate<T>(sVecRecipMod, 0);
    sPinv          = modDims<T>(sPinv, dim4(sPinv.dims()[0], sPinv.dims()[1],
                                            sVecRecip.dims()[2], sVecRecip.dims()[3]));

    Array<T> uT = transpose(u, true);

    // Crop v and u* for final matmul later based on s+'s size, because
    // sVec produced by svd() has minimal dim length (no extra zeroes).
    // Thus s+ produced by diagCreate() will have minimal dims as well,
    // and v could have an extra dim0 or u* could have an extra dim1
    if (v.dims()[1] > sPinv.dims()[0]) {
        v = getSubArray(v, false, 0, v.dims()[0] - 1, 0, sPinv.dims()[0] - 1, 0,
                        v.dims()[2] - 1, 0, v.dims()[3] - 1);
    }
    if (uT.dims()[0] > sPinv.dims()[1]) {
        uT = getSubArray(uT, false, 0, sPinv.dims()[1] - 1, 0, uT.dims()[1] - 1,
                         0, uT.dims()[2] - 1, 0, uT.dims()[3] - 1);
    }

    Array<T> vsPinv =
        createEmptyArray<T>(dim4(v.dims()[0], sPinv.dims()[1], P, Q));
    Array<T> out =
        createEmptyArray<T>(dim4(vsPinv.dims()[0], uT.dims()[1], P, Q));

    T alpha = scalar<T>(1.0);
    T beta  = scalar<T>(0.0);

    gemm<T>(vsPinv, AF_MAT_NONE, AF_MAT_NONE, &alpha, v, sPinv, &beta);
    gemm<T>(out, AF_MAT_NONE, AF_MAT_NONE, &alpha, vsPinv, uT, &beta);

    return out;
}

template<typename T>
static inline af_array pinverse(const af_array in, const double tol) {
    return getHandle(pinverseSvd<T>(getArray<T>(in), tol));
}

af_err af_pinverse(af_array *out, const af_array in, const double tol,
                   const af_mat_prop options) {
    try {
        const ArrayInfo &i_info = getInfo(in);

        af_dtype type = i_info.getType();

        if (options != AF_MAT_NONE) {
            AF_ERROR("Using this property is not yet supported in inverse",
                     AF_ERR_NOT_SUPPORTED);
        }

        ARG_ASSERT(1, i_info.isFloating());  // Only floating and complex types
        ARG_ASSERT(2, tol >= 0.);            // Ensure tolerance is not negative

        af_array output;

        if (i_info.ndims() == 0) { return af_retain_array(out, in); }

        switch (type) {
            case f32: output = pinverse<float>(in, tol); break;
            case f64: output = pinverse<double>(in, tol); break;
            case c32: output = pinverse<cfloat>(in, tol); break;
            case c64: output = pinverse<cdouble>(in, tol); break;
            default: TYPE_ERROR(1, type);
        }
        swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
