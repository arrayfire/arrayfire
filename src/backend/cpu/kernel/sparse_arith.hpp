/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <math.hpp>

#include <cmath>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T, af_op_t op>
struct arith_op {
    T operator()(T v1, T v2) {
        UNUSED(v1);
        UNUSED(v2);
        return scalar<T>(0);
    }
};

template<typename T>
struct arith_op<T, af_add_t> {
    T operator()(T v1, T v2) { return v1 + v2; }
};

template<typename T>
struct arith_op<T, af_sub_t> {
    T operator()(T v1, T v2) { return v1 - v2; }
};

template<typename T>
struct arith_op<T, af_mul_t> {
    T operator()(T v1, T v2) { return v1 * v2; }
};

template<typename T>
struct arith_op<T, af_div_t> {
    T operator()(T v1, T v2) { return v1 / v2; }
};

template<typename T, af_op_t op, af_storage type>
void sparseArithOpD(Param<T> output, CParam<T> values, CParam<int> rowIdx,
                    CParam<int> colIdx, CParam<T> rhs,
                    const bool reverse = false) {
    T *oPtr       = output.get();
    const T *hPtr = rhs.get();

    const T *vPtr   = values.get();
    const int *rPtr = rowIdx.get();
    const int *cPtr = colIdx.get();

    dim4 odims    = output.dims();
    dim4 ostrides = output.strides();
    ;
    dim4 hstrides = rhs.strides();
    ;

    std::vector<int> temp;
    if (type == AF_STORAGE_CSR) {
        temp.resize(values.dims().elements());
        for (int i = 0; i < rowIdx.dims(0) - 1; i++) {
            for (int ii = rPtr[i]; ii < rPtr[i + 1]; ii++) { temp[ii] = i; }
        }
        //} else if(type == AF_STORAGE_CSC) {   // For future
    }

    const int *xx = (type == AF_STORAGE_CSR) ? temp.data() : rPtr;
    const int *yy = (type == AF_STORAGE_CSC) ? temp.data() : cPtr;

    for (int i = 0; i < (int)values.dims().elements(); i++) {
        // Bad index data
        if (xx[i] >= odims[0] || yy[i] >= odims[1]) continue;

        int offset = xx[i] + yy[i] * ostrides[1];
        int hoff   = xx[i] + yy[i] * hstrides[1];

        if (reverse)
            oPtr[offset] = arith_op<T, op>()(hPtr[hoff], vPtr[i]);
        else
            oPtr[offset] = arith_op<T, op>()(vPtr[i], hPtr[hoff]);
    }
}

template<typename T, af_op_t op, af_storage type>
void sparseArithOpS(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
                    CParam<T> rhs, const bool reverse = false) {
    T *vPtr         = values.get();
    const int *rPtr = rowIdx.get();
    const int *cPtr = colIdx.get();

    const T *hPtr = rhs.get();

    dim4 dims     = rhs.dims();
    dim4 hstrides = rhs.strides();

    std::vector<int> temp;
    if (type == AF_STORAGE_CSR) {
        temp.resize(values.dims().elements());
        for (int i = 0; i < rowIdx.dims(0) - 1; i++) {
            for (int ii = rPtr[i]; ii < rPtr[i + 1]; ii++) { temp[ii] = i; }
        }
        //} else if(type == AF_STORAGE_CSC) {   // For future
    }

    const int *xx = (type == AF_STORAGE_CSR) ? temp.data() : rPtr;
    const int *yy = (type == AF_STORAGE_CSC) ? temp.data() : cPtr;

    for (int i = 0; i < (int)values.dims().elements(); i++) {
        // Bad index data
        if (xx[i] >= dims[0] || yy[i] >= dims[1]) continue;

        int hoff = xx[i] + yy[i] * hstrides[1];

        if (reverse)
            vPtr[i] = arith_op<T, op>()(hPtr[hoff], vPtr[i]);
        else
            vPtr[i] = arith_op<T, op>()(vPtr[i], hPtr[hoff]);
    }
}

// The following functions can handle CSR
// storage format only as of now.
static void calcOutNNZ(Param<int> outRowIdx, const uint M, const uint N,
                       CParam<int> lRowIdx, CParam<int> lColIdx,
                       CParam<int> rRowIdx, CParam<int> rColIdx) {
    UNUSED(N);
    int *orPtr       = outRowIdx.get();
    const int *lrPtr = lRowIdx.get();
    const int *lcPtr = lColIdx.get();
    const int *rrPtr = rRowIdx.get();
    const int *rcPtr = rColIdx.get();

    unsigned csrOutCount = 0;
    for (uint row = 0; row < M; ++row) {
        const int lEnd = lrPtr[row + 1];
        const int rEnd = rrPtr[row + 1];

        uint rowNNZ = 0;
        int l       = lrPtr[row];
        int r       = rrPtr[row];
        while (l < lEnd && r < rEnd) {
            int lci = lcPtr[l];
            int rci = rcPtr[r];

            l += (lci <= rci);
            r += (lci >= rci);
            rowNNZ++;
        }
        // Elements from lhs or rhs are exhausted.
        // Just count left over elements
        rowNNZ += (lEnd - l);
        rowNNZ += (rEnd - r);

        orPtr[row] = csrOutCount;
        csrOutCount += rowNNZ;
    }
    // Write out the Rows+1 entry
    orPtr[M] = csrOutCount;
}

template<typename T, af_op_t op>
void sparseArithOp(Param<T> oVals, Param<int> oColIdx, CParam<int> oRowIdx,
                   const uint Rows, CParam<T> lvals, CParam<int> lRowIdx,
                   CParam<int> lColIdx, CParam<T> rvals, CParam<int> rRowIdx,
                   CParam<int> rColIdx) {
    const int *orPtr = oRowIdx.get();
    const T *lvPtr   = lvals.get();
    const int *lrPtr = lRowIdx.get();
    const int *lcPtr = lColIdx.get();
    const T *rvPtr   = rvals.get();
    const int *rrPtr = rRowIdx.get();
    const int *rcPtr = rColIdx.get();

    arith_op<T, op> binOp;

    auto ZERO = scalar<T>(0);

    for (uint row = 0; row < Rows; ++row) {
        const int lEnd = lrPtr[row + 1];
        const int rEnd = rrPtr[row + 1];
        const int offs = orPtr[row];

        T *ovPtr   = oVals.get() + offs;
        int *ocPtr = oColIdx.get() + offs;

        uint rowNNZ = 0;
        int l       = lrPtr[row];
        int r       = rrPtr[row];
        while (l < lEnd && r < rEnd) {
            int lci = lcPtr[l];
            int rci = rcPtr[r];

            T lhs = (lci <= rci ? lvPtr[l] : ZERO);
            T rhs = (lci >= rci ? rvPtr[r] : ZERO);

            ovPtr[rowNNZ] = binOp(lhs, rhs);
            ocPtr[rowNNZ] = (lci <= rci) ? lci : rci;

            l += (lci <= rci);
            r += (lci >= rci);
            rowNNZ++;
        }
        while (l < lEnd) {
            ovPtr[rowNNZ] = binOp(lvPtr[l], ZERO);
            ocPtr[rowNNZ] = lcPtr[l];
            l++;
            rowNNZ++;
        }
        while (r < rEnd) {
            ovPtr[rowNNZ] = binOp(ZERO, rvPtr[r]);
            ocPtr[rowNNZ] = rcPtr[r];
            r++;
            rowNNZ++;
        }
    }
}
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
