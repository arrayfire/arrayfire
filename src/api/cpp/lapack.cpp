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
    void lu(array &out, array &pivot, const array &in)
    {
        out = in.copy();
        af_array p = 0;
        AF_THROW(af_lu_inplace(&p, out.get()));
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

    array luInplace(array &in)
    {
        af_array pivot = 0;
        AF_THROW(af_lu_inplace(&pivot, in.get()));
        return array(pivot);
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

    array qrInplace(array &in)
    {
        af_array tau = 0;
        AF_THROW(af_qr_inplace(&tau, in.get()));
        return array(tau);
    }

    array cholesky(const array &in, int *info, const bool is_upper)
    {
        af_array out;
        AF_THROW(af_cholesky(&out, info, in.get(), is_upper));
        return array(out);
    }

    void choleskyInplace(array &in, int *info, const bool is_upper)
    {
        AF_THROW(af_cholesky_inplace(info, in.get(), is_upper));
    }

    array solve(const array &a, const array &b, const af_solve_t options)
    {
        af_array out;
        AF_THROW(af_solve(&out, a.get(), b.get(), options));
        return array(out);
    }

    array inverse(const array &in)
    {
        af_array out;
        AF_THROW(af_inverse(&out, in.get()));
        return array(out);
    }

}
