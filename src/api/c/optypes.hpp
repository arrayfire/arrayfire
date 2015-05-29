/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

typedef enum {
    af_add_t = 0,
    af_sub_t,
    af_mul_t,
    af_div_t,

    af_and_t,
    af_or_t,
    af_eq_t,
    af_neq_t,
    af_lt_t,
    af_le_t,
    af_gt_t,
    af_ge_t,

    af_bitor_t,
    af_bitand_t,
    af_bitxor_t,
    af_bitshiftl_t,
    af_bitshiftr_t,

    af_min_t,
    af_max_t,
    af_cplx2_t,
    af_atan2_t,
    af_pow_t,
    af_hypot_t,

    af_sin_t,
    af_cos_t,
    af_tan_t,
    af_asin_t,
    af_acos_t,
    af_atan_t,

    af_sinh_t,
    af_cosh_t,
    af_tanh_t,
    af_asinh_t,
    af_acosh_t,
    af_atanh_t,

    af_exp_t,
    af_expm1_t,
    af_erf_t,
    af_erfc_t,

    af_log_t,
    af_log10_t,
    af_log1p_t,
    af_log2_t,

    af_sqrt_t,
    af_cbrt_t,

    af_abs_t,
    af_cast_t,
    af_cplx_t,
    af_real_t,
    af_imag_t,
    af_conj_t,

    af_floor_t,
    af_ceil_t,
    af_round_t,
    af_trunc_t,
    af_sign_t,

    af_rem_t,
    af_mod_t,

    af_tgamma_t,
    af_lgamma_t,

    af_notzero_t,

    af_iszero_t,
    af_isinf_t,
    af_isnan_t,

    af_noop_t
} af_op_t;
