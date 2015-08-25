/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/lapack.h>
#include "error.hpp"

namespace af
{
    void svd(array &u, array &s, array &vt, const array &in)
    {
        af_array sl = 0, ul = 0, vtl = 0;
        AF_THROW(af_svd(&ul, &sl, &vtl, in.get()));
        s = array(sl);
        u = array(ul);
        vt = array(vtl);
    }

    void svdInPlace(array &u, array &s, array &vt, array &in)
    {
        af_array sl = 0, ul = 0, vtl = 0;
        AF_THROW(af_svd_inplace(&ul, &sl, &vtl, in.get()));
        s = array(sl);
        u = array(ul);
        vt = array(vtl);
    }

    void lu(array &out, array &pivot, const array &in, const bool is_lapack_piv)
    {
        out = in.copy();
        af_array p = 0;
        AF_THROW(af_lu_inplace(&p, out.get(), is_lapack_piv));
        pivot = array(p);
    }

    void lu(array &lower, array &upper, array &pivot, const array &in)
    {
        af_array l = 0, u = 0, p = 0;
        AF_THROW(af_lu(&l, &u, &p, in.get()));
        lower = array(l);
        upper = array(u);
        pivot = array(p);
    }

    void luInPlace(array &pivot, array &in, const bool is_lapack_piv)
    {
        af_array p = 0;
        AF_THROW(af_lu_inplace(&p, in.get(), is_lapack_piv));
        pivot = array(p);
    }

    void qr(array &out, array &tau, const array &in)
    {
        out = in.copy();
        af_array t = 0;
        AF_THROW(af_qr_inplace(&t, out.get()));
        tau = array(t);
    }

    void qr(array &q, array &r, array &tau, const array &in)
    {
        af_array q_ = 0, r_ = 0, t_ = 0;
        AF_THROW(af_qr(&q_, &r_, &t_, in.get()));
        q = array(q_);
        r = array(r_);
        tau = array(t_);
    }

    void qrInPlace(array &tau, array &in)
    {
        af_array t = 0;
        AF_THROW(af_qr_inplace(&t, in.get()));
        tau = array(t);
    }

    int cholesky(array &out, const array &in, const bool is_upper)
    {
        int info = 0;
        af_array res;
        AF_THROW(af_cholesky(&res, &info, in.get(), is_upper));
        out = array(res);
        return info;
    }

    int choleskyInPlace(array &in, const bool is_upper)
    {
        int info = 0;
        AF_THROW(af_cholesky_inplace(&info, in.get(), is_upper));
        return info;
    }

    array solve(const array &a, const array &b, const matProp options)
    {
        af_array out;
        AF_THROW(af_solve(&out, a.get(), b.get(), options));
        return array(out);
    }

    array solveLU(const array &a, const array &piv,
                  const array &b, const matProp options)
    {
        af_array out;
        AF_THROW(af_solve_lu(&out, a.get(), piv.get(), b.get(), options));
        return array(out);
    }

    array inverse(const array &in, const matProp options)
    {
        af_array out;
        AF_THROW(af_inverse(&out, in.get(), options));
        return array(out);
    }

    unsigned rank(const array &in, const double tol)
    {
        unsigned r = 0;
        AF_THROW(af_rank(&r, in.get(), tol));
        return r;
    }

#define INSTANTIATE_DET(TR, TC)                     \
    template<> AFAPI                                \
    TR det(const array &in)                         \
    {                                               \
        double real;                                \
        double imag;                                \
        AF_THROW(af_det(&real, &imag, in.get()));   \
        return real;                                \
    }                                               \
    template<> AFAPI                                \
    TC det(const array &in)                         \
    {                                               \
        double real;                                \
        double imag;                                \
        AF_THROW(af_det(&real, &imag, in.get()));   \
        TC out((TR)real, (TR)imag);                 \
        return out;                                 \
    }                                               \

    INSTANTIATE_DET(float, af_cfloat)
    INSTANTIATE_DET(double, af_cdouble)

    double norm(const array &in, const normType type,
                const double p, const double q)
    {
        double out;
        AF_THROW(af_norm(&out, in.get(), type, p, q));
        return out;
    }
}
