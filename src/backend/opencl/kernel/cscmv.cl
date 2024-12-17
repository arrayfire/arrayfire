/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if IS_DBL || IS_LONG
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#if IS_CPLX
T __cmul(T lhs, T rhs) {
    T out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}

T __ccmul(T lhs, T rhs) {
    T out;
    out.x = lhs.x * rhs.x + lhs.y * rhs.y;
    out.y = lhs.x * rhs.y - lhs.y * rhs.x;
    return out;
}

#define MUL(a, b) __cmul(a, b)

#if IS_CONJ
#define CMUL(a, b) __ccmul(a, b)
#else
#define CMUL(a, b) __cmul(a, b)
#endif

#else
#define MUL(a, b) (a) * (b)
#define CMUL(a, b) (a) * (b)
#endif

#if IS_DBL || IS_LONG
#define U ulong
#define ATOMIC_FN atom_cmpxchg
#else
#define U unsigned
#define ATOMIC_FN atomic_cmpxchg
#endif

#if IS_CPLX
inline void atomicAdd(volatile __global T *ptr, T val) {
    union {
        U u[2];
        T t;
    } next, expected, current;
    current.t = *ptr;

    do {
        expected.t.x = current.t.x;
        next.t.x = expected.t.x + val.x;
        current.u[0] = ATOMIC_FN((volatile __global U *) ptr, expected.u[0], next.u[0]);
    } while(current.u[0] != expected.u[0]);
    do {
        expected.t.y = current.t.y;
        next.t.y = expected.t.y + val.y;
        current.u[1] = ATOMIC_FN(((volatile __global U *) ptr) + 1, expected.u[1], next.u[1]);
    } while(current.u[1] != expected.u[1]);
}
#else
inline void atomicAdd(volatile __global T *ptr, T val) {
    union {
        U u;
        T t;
    } next, expected, current;
    current.t = *ptr;

    do {
        expected.t = current.t;
        next.t = expected.t + val;
        current.u = ATOMIC_FN((volatile __global U *) ptr, expected.u, next.u);
    } while(current.u != expected.u);
}
#endif

kernel void cscmv_beta(global T *output, const int M, const T beta) {
    for(unsigned j = get_global_id(0); j < M; j += THREADS * get_num_groups(0))
        output[j] *= beta;
}

kernel void cscmv_atomic(
    global T *output, __global T *values,
    global int *colidx,  // rowidx from csr is colidx in csc
    global int *rowidx,  // colidx from csr is rowidx in csc
    const int K,                 // M from csr is K in csc
    global const T *rhs, const KParam rinfo, const T alpha) {

    rhs += rinfo.offset;

    for(unsigned j = get_group_id(0); j < K; j += get_num_groups(0)) {
        for(unsigned i = get_local_id(0) + colidx[j]; i < colidx[j + 1]; i += THREADS) {
            T outval = CMUL(values[i], rhs[j]);
#if USE_ALPHA
            outval = MUL(alpha, outval);
#endif
            atomicAdd(output + rowidx[i], outval);
        }
    }
}
