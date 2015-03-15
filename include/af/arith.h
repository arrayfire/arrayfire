/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include "array.h"

#ifdef __cplusplus
namespace af
{
    class array;
    /**
       C++ API for min of two arrays

       \param[in] lhs first input
       \param[in] rhs second input
       \return minimum of \p lhs and \p rhs

       \ingroup numeric_func_min
    */
    AFAPI array min    (const array &lhs, const array &rhs);

    /**
       C++ API for min of array and scalar

       \param[in] lhs first input
       \param[in] rhs second input
       \return minimum of \p lhs and \p rhs

       \ingroup numeric_func_min
    */
    AFAPI array min    (const array &lhs, const double rhs);

    /**
       C++ API for min of scalar and array

       \param[in] lhs first input
       \param[in] rhs second input
       \return minimum of \p lhs and \p rhs

       \ingroup numeric_func_min
    */
    AFAPI array min    (const double lhs, const array &rhs);

    /**
       C++ API for max of two arrays

       \param[in] lhs first input
       \param[in] rhs second input
       \return maximum of \p lhs and \p rhs

       \ingroup numeric_func_max
    */
    AFAPI array max    (const array &lhs, const array &rhs);

    /**
       C++ API for max of array and scalar

       \param[in] lhs first input
       \param[in] rhs second input
       \return maximum of \p lhs and \p rhs

       \ingroup numeric_func_max
    */
    AFAPI array max    (const array &lhs, const double rhs);

    /**
       C++ API for max of array and scalar

       \param[in] lhs first input
       \param[in] rhs second input
       \return maximum of \p lhs and \p rhs

       \ingroup numeric_func_max
    */
    AFAPI array max    (const double lhs, const array &rhs);

    /**
       C++ API for remainder when array divides array

       \param[in] lhs is numerator
       \param[in] rhs is denominator
       \return remainder when \p rhs divides \p lhs

       \ingroup numeric_func_rem
    */
    AFAPI array rem    (const array &lhs, const array &rhs);

    /**
       C++ API for remainder when scalar divides array

       \param[in] lhs is numerator
       \param[in] rhs is denominator
       \return remainder when \p rhs divides \p lhs

       \ingroup numeric_func_rem
    */
    AFAPI array rem    (const array &lhs, const double rhs);

    /**
       C++ API for remainder when array divides scalar

       \param[in] lhs is numerator
       \param[in] rhs is denominator
       \return remainder when \p rhs divides \p lhs

       \ingroup numeric_func_rem
    */
    AFAPI array rem    (const double lhs, const array &rhs);

    /**
       C++ API for modulus when dividend and divisor are arrays

       \param[in] lhs is dividend
       \param[in] rhs is divisor
       \return \p lhs modulo \p rhs

       \ingroup numeric_func_mod
    */
    AFAPI array mod    (const array &lhs, const array &rhs);

    /**
       C++ API for modulus when dividend is array, divisor is scalar

       \param[in] lhs is dividend
       \param[in] rhs is divisor
       \return \p lhs modulo \p rhs

       \ingroup numeric_func_mod
    */
    AFAPI array mod    (const array &lhs, const double rhs);

    /**
       C++ API for modulus when dividend is scalar, divisor is array

       \param[in] lhs is dividend
       \param[in] rhs is divisor
       \return \p lhs modulo \p rhs

       \ingroup numeric_func_mod
    */
    AFAPI array mod    (const double lhs, const array &rhs);

    /**
       C++ API for power when base and exponent are arrays

       \param[in] lhs is base
       \param[in] rhs is exponent
       \return \p lhs raised to power \p rhs

       \ingroup numeric_func_pow
    */
    AFAPI array pow    (const array &lhs, const array &rhs);

    /**
       C++ API for power when base is array, exponent is scalar

       \param[in] lhs is base
       \param[in] rhs is exponent
       \return \p lhs raised to power \p rhs

       \ingroup numeric_func_pow
    */
    AFAPI array pow    (const array &lhs, const double rhs);

    /**
       C++ API for power when base is scalar, exponent is array

       \param[in] lhs is base
       \param[in] rhs is exponent
       \return \p lhs raised to power \p rhs

       \ingroup numeric_func_pow
    */
    AFAPI array pow    (const double lhs, const array &rhs);

    AFAPI array round  (const array &in);
    AFAPI array floor  (const array &in);
    AFAPI array ceil   (const array &in);

    AFAPI array complex(const array &lhs, const array &rhs);
    AFAPI array complex(const array &lhs, const double rhs);
    AFAPI array complex(const double lhs, const array &rhs);

    AFAPI array atan2  (const array &lhs, const array &rhs);
    AFAPI array atan2  (const array &lhs, const double rhs);
    AFAPI array atan2  (const double lhs, const array &rhs);

    AFAPI array hypot  (const array &lhs, const array &rhs);
    AFAPI array hypot  (const array &lhs, const double rhs);
    AFAPI array hypot  (const double lhs, const array &rhs);

    AFAPI array complex(const array &in);
    AFAPI array real   (const array &in);
    AFAPI array imag   (const array &in);
    AFAPI array conjg  (const array &in);
    AFAPI array abs    (const array &in);

    AFAPI array sin    (const array &in);
    AFAPI array cos    (const array &in);
    AFAPI array tan    (const array &in);

    AFAPI array asin   (const array &in);
    AFAPI array acos   (const array &in);
    AFAPI array atan   (const array &in);

    AFAPI array sinh   (const array &in);
    AFAPI array cosh   (const array &in);
    AFAPI array tanh   (const array &in);

    AFAPI array asinh  (const array &in);
    AFAPI array acosh  (const array &in);
    AFAPI array atanh  (const array &in);

    AFAPI array exp    (const array &in);
    AFAPI array expm1  (const array &in);
    AFAPI array erf    (const array &in);
    AFAPI array erfc   (const array &in);

    AFAPI array log    (const array &in);
    AFAPI array log1p  (const array &in);
    AFAPI array log10  (const array &in);

    AFAPI array sqrt   (const array &in);
    AFAPI array cbrt   (const array &in);

    AFAPI array iszero (const array &in);
    AFAPI array isInf  (const array &in);
    AFAPI array isNaN  (const array &in);

    AFAPI array tgamma (const array &in);
    AFAPI array lgamma (const array &in);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_add   (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_sub   (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_mul   (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_div   (af_array *result, const af_array lhs, const af_array rhs, bool batch);

    AFAPI af_err af_lt    (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_gt    (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_le    (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_ge    (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_eq    (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_neq   (af_array *result, const af_array lhs, const af_array rhs, bool batch);

    AFAPI af_err af_and   (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_or    (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_not   (af_array *result, const af_array in);

    /**
       C API for min of two arrays

       \param[out] out will contain minimum of \p lhs and \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup numeric_func_min
    */
    AFAPI af_err af_minof (af_array *result, const af_array lhs, const af_array rhs, bool batch);

    /**
       C API for max of two arrays

       \param[out] out will contain maximum of \p lhs and \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup numeric_func_max
    */
    AFAPI af_err af_maxof (af_array *result, const af_array lhs, const af_array rhs, bool batch);

    /**
       C API for remainder

       \param[out] out will contain the remainder of \p lhs divided by \p rhs
       \param[in] lhs is numerator
       \param[in] rhs is denominator
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup numeric_func_rem
    */
    AFAPI af_err af_rem   (af_array *result, const af_array lhs, const af_array rhs, bool batch);

    /**
       C API for modulus

       \param[out] out will contain the output of \p lhs modulo \p rhs
       \param[in] lhs is dividend
       \param[in] rhs is divisor
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup numeric_func_mod
    */
    AFAPI af_err af_mod   (af_array *result, const af_array lhs, const af_array rhs, bool batch);

    /**
       C API for power

       \param[out] out will contain \p lhs raised to power \p rhs
       \param[in] lhs is base
       \param[in] rhs is exponent
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup numeric_func_pow
    */
    AFAPI af_err af_pow   (af_array *result, const af_array lhs, const af_array rhs, bool batch);

    AFAPI af_err af_cplx2 (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_atan2 (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_hypot (af_array *result, const af_array lhs, const af_array rhs, bool batch);

    AFAPI af_err af_bitand   (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_bitor    (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_bitxor   (af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_bitshiftl(af_array *result, const af_array lhs, const af_array rhs, bool batch);
    AFAPI af_err af_bitshiftr(af_array *result, const af_array lhs, const af_array rhs, bool batch);

    AFAPI af_err af_cast    (af_array *out, const af_array in, af_dtype type);
    AFAPI af_err af_cplx    (af_array *out, const af_array in);
    AFAPI af_err af_real    (af_array *out, const af_array in);
    AFAPI af_err af_imag    (af_array *out, const af_array in);
    AFAPI af_err af_conjg   (af_array *out, const af_array in);
    AFAPI af_err af_abs     (af_array *out, const af_array in);

    AFAPI af_err af_round   (af_array *out, const af_array in);
    AFAPI af_err af_floor   (af_array *out, const af_array in);
    AFAPI af_err af_ceil    (af_array *out, const af_array in);

    AFAPI af_err af_sin     (af_array *out, const af_array in);
    AFAPI af_err af_cos     (af_array *out, const af_array in);
    AFAPI af_err af_tan     (af_array *out, const af_array in);

    AFAPI af_err af_asin    (af_array *out, const af_array in);
    AFAPI af_err af_acos    (af_array *out, const af_array in);
    AFAPI af_err af_atan    (af_array *out, const af_array in);

    AFAPI af_err af_sinh    (af_array *out, const af_array in);
    AFAPI af_err af_cosh    (af_array *out, const af_array in);
    AFAPI af_err af_tanh    (af_array *out, const af_array in);

    AFAPI af_err af_asinh   (af_array *out, const af_array in);
    AFAPI af_err af_acosh   (af_array *out, const af_array in);
    AFAPI af_err af_atanh   (af_array *out, const af_array in);

    AFAPI af_err af_exp     (af_array *out, const af_array in);
    AFAPI af_err af_expm1   (af_array *out, const af_array in);
    AFAPI af_err af_erf     (af_array *out, const af_array in);
    AFAPI af_err af_erfc    (af_array *out, const af_array in);

    AFAPI af_err af_log     (af_array *out, const af_array in);
    AFAPI af_err af_log1p   (af_array *out, const af_array in);
    AFAPI af_err af_log10   (af_array *out, const af_array in);

    AFAPI af_err af_sqrt    (af_array *out, const af_array in);
    AFAPI af_err af_cbrt    (af_array *out, const af_array in);

    AFAPI af_err af_iszero  (af_array *out, const af_array in);
    AFAPI af_err af_isinf   (af_array *out, const af_array in);
    AFAPI af_err af_isnan   (af_array *out, const af_array in);

    AFAPI af_err af_tgamma   (af_array *out, const af_array in);
    AFAPI af_err af_lgamma   (af_array *out, const af_array in);

#ifdef __cplusplus
}
#endif
