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
    AFAPI void lu(array &out, array &pivot, const array &in)
    {
        out = in.copy();
        af_array p = pivot.get();
        AF_THROW(af_lu_inplace(&p, out.get()));
    }

    AFAPI void lu(array& lower, array& upper, array& pivot, const array& in)
    {
        af_array l = 0, u = 0, p = 0;
        AF_THROW(af_lu(&l, &u, &p, in.get()));
        lower = array(l);
        upper = array(u);
        pivot = array(p);
    }

    AFAPI array luInplace(array &in)
    {
        af_array pivot = 0;
        AF_THROW(af_lu_inplace(&pivot, in.get()));
        return array(pivot);
    }

    AFAPI void qr(array& out, array& tau, const array& in)
    {
        out = in.copy();
        af_array t = tau.get();
        AF_THROW(af_qr_inplace(&t, out.get()));
    }

    AFAPI void qr(array& q, array& r, array& tau, const array& in)
    {
        af_array q_ = 0, r_ = 0, t_ = 0;
        AF_THROW(af_qr(&q_, &r_, &t_, in.get()));
        q = array(q_);
        r = array(r_);
        tau = array(t_);
    }

    AFAPI array qrInplace(array& in)
    {
        af_array tau = 0;
        AF_THROW(af_qr_inplace(&tau, in.get()));
        return array(tau);
    }

    AFAPI array cholesky(const array& in, int *info, const bool is_upper)
    {
        array out = in.copy();
        AF_THROW(af_cholesky_inplace(info, out.get(), is_upper));
        return out;
    }

    AFAPI void choleskyInplace(array& in, int *info, const bool is_upper)
    {
        AF_THROW(af_cholesky_inplace(info, in.get(), is_upper));
    }
}
