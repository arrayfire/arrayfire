/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/TemplateArg.hpp>

#include <common/internal_enums.hpp>
#include <optypes.hpp>
#include <af/defines.h>

#include <string>

using std::string;

template<typename T>
string toString(T value) {
    return std::to_string(value);
}

template string toString<int>(int);
template string toString<long>(long);
template string toString<long long>(long long);
template string toString<unsigned>(unsigned);
template string toString<unsigned long>(unsigned long);
template string toString<unsigned long long>(unsigned long long);
template string toString<float>(float);
template string toString<double>(double);
template string toString<long double>(long double);

template<>
string toString(bool val) {
    return string(val ? "true" : "false");
}

template<>
string toString(const char* str) {
    return string(str);
}

template<>
string toString(const string str) {
    return str;
}

template<>
string toString(unsigned short val) {
    return std::to_string((unsigned int)(val));
}

template<>
string toString(short val) {
    return std::to_string(int(val));
}

template<>
string toString(unsigned char val) {
    return std::to_string((unsigned int)(val));
}

template<>
string toString(char val) {
    return std::to_string(int(val));
}

string getOpEnumStr(af_op_t val) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
        CASE_STMT(af_add_t);
        CASE_STMT(af_sub_t);
        CASE_STMT(af_mul_t);
        CASE_STMT(af_div_t);

        CASE_STMT(af_and_t);
        CASE_STMT(af_or_t);
        CASE_STMT(af_eq_t);
        CASE_STMT(af_neq_t);
        CASE_STMT(af_lt_t);
        CASE_STMT(af_le_t);
        CASE_STMT(af_gt_t);
        CASE_STMT(af_ge_t);

        CASE_STMT(af_bitor_t);
        CASE_STMT(af_bitand_t);
        CASE_STMT(af_bitxor_t);
        CASE_STMT(af_bitshiftl_t);
        CASE_STMT(af_bitshiftr_t);

        CASE_STMT(af_min_t);
        CASE_STMT(af_max_t);
        CASE_STMT(af_cplx2_t);
        CASE_STMT(af_atan2_t);
        CASE_STMT(af_pow_t);
        CASE_STMT(af_hypot_t);

        CASE_STMT(af_sin_t);
        CASE_STMT(af_cos_t);
        CASE_STMT(af_tan_t);
        CASE_STMT(af_asin_t);
        CASE_STMT(af_acos_t);
        CASE_STMT(af_atan_t);

        CASE_STMT(af_sinh_t);
        CASE_STMT(af_cosh_t);
        CASE_STMT(af_tanh_t);
        CASE_STMT(af_asinh_t);
        CASE_STMT(af_acosh_t);
        CASE_STMT(af_atanh_t);

        CASE_STMT(af_exp_t);
        CASE_STMT(af_expm1_t);
        CASE_STMT(af_erf_t);
        CASE_STMT(af_erfc_t);

        CASE_STMT(af_log_t);
        CASE_STMT(af_log10_t);
        CASE_STMT(af_log1p_t);
        CASE_STMT(af_log2_t);

        CASE_STMT(af_sqrt_t);
        CASE_STMT(af_cbrt_t);

        CASE_STMT(af_abs_t);
        CASE_STMT(af_cast_t);
        CASE_STMT(af_cplx_t);
        CASE_STMT(af_real_t);
        CASE_STMT(af_imag_t);
        CASE_STMT(af_conj_t);

        CASE_STMT(af_floor_t);
        CASE_STMT(af_ceil_t);
        CASE_STMT(af_round_t);
        CASE_STMT(af_trunc_t);
        CASE_STMT(af_signbit_t);

        CASE_STMT(af_rem_t);
        CASE_STMT(af_mod_t);

        CASE_STMT(af_tgamma_t);
        CASE_STMT(af_lgamma_t);

        CASE_STMT(af_notzero_t);

        CASE_STMT(af_iszero_t);
        CASE_STMT(af_isinf_t);
        CASE_STMT(af_isnan_t);

        CASE_STMT(af_sigmoid_t);

        CASE_STMT(af_noop_t);

        CASE_STMT(af_select_t);
        CASE_STMT(af_not_select_t);
        CASE_STMT(af_rsqrt_t);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_op_t val) {
    return getOpEnumStr(val);
}

template<>
string toString(af_interp_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(AF_INTERP_NEAREST);
        CASE_STMT(AF_INTERP_LINEAR);
        CASE_STMT(AF_INTERP_BILINEAR);
        CASE_STMT(AF_INTERP_CUBIC);
        CASE_STMT(AF_INTERP_LOWER);
        CASE_STMT(AF_INTERP_LINEAR_COSINE);
        CASE_STMT(AF_INTERP_BILINEAR_COSINE);
        CASE_STMT(AF_INTERP_BICUBIC);
        CASE_STMT(AF_INTERP_CUBIC_SPLINE);
        CASE_STMT(AF_INTERP_BICUBIC_SPLINE);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_border_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(AF_PAD_ZERO);
        CASE_STMT(AF_PAD_SYM);
        CASE_STMT(AF_PAD_CLAMP_TO_EDGE);
        CASE_STMT(AF_PAD_PERIODIC);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_moment_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(AF_MOMENT_M00);
        CASE_STMT(AF_MOMENT_M01);
        CASE_STMT(AF_MOMENT_M10);
        CASE_STMT(AF_MOMENT_M11);
        CASE_STMT(AF_MOMENT_FIRST_ORDER);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_match_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(AF_SAD);
        CASE_STMT(AF_ZSAD);
        CASE_STMT(AF_LSAD);
        CASE_STMT(AF_SSD);
        CASE_STMT(AF_ZSSD);
        CASE_STMT(AF_LSSD);
        CASE_STMT(AF_NCC);
        CASE_STMT(AF_ZNCC);
        CASE_STMT(AF_SHD);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_flux_function p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(AF_FLUX_QUADRATIC);
        CASE_STMT(AF_FLUX_EXPONENTIAL);
        CASE_STMT(AF_FLUX_DEFAULT);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(AF_BATCH_KIND val) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
        CASE_STMT(AF_BATCH_NONE);
        CASE_STMT(AF_BATCH_LHS);
        CASE_STMT(AF_BATCH_RHS);
        CASE_STMT(AF_BATCH_SAME);
        CASE_STMT(AF_BATCH_DIFF);
        CASE_STMT(AF_BATCH_UNSUPPORTED);
    }
#undef CASE_STMT
    return retVal;
}
