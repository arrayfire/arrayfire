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

    af_min_t,
    af_max_t,
    af_cplx2_t,
    af_tan2_t,
    af_pow_t,

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
    af_log_t,
    af_log10_t,
    af_expm1_t,
    af_log1p_t,

    af_abs_t,
    af_cast_t,
    af_cplx_t,
    af_real_t,
    af_imag_t,
    af_sqrt_t,

    af_notzero_t,

    af_noop_t
} af_op_t;
