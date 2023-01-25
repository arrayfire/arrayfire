/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>

#ifdef __cplusplus
namespace af
{
    class array;

    /// C++ Interface to find the elementwise minimum between two arrays.
    ///
    /// \param[in] lhs input array
    /// \param[in] rhs input array
    /// \return minimum of \p lhs and \p rhs
    ///
    /// \ingroup arith_func_min
    AFAPI array min    (const array &lhs, const array &rhs);

    /// C++ Interface to find the elementwise minimum between an array and a scalar value.
    ///
    /// \param[in] lhs input array
    /// \param[in] rhs scalar value
    /// \return minimum of \p lhs and \p rhs
    ///
    /// \ingroup arith_func_min
    AFAPI array min    (const array &lhs, const double rhs);

    /// C++ Interface to find the elementwise minimum between an array and a scalar value.
    ///
    /// \param[in] lhs scalar value
    /// \param[in] rhs input array
    /// \return minimum of \p lhs and \p rhs
    ///
    /// \ingroup arith_func_min
    AFAPI array min    (const double lhs, const array &rhs);

    /// C++ Interface to find the elementwise maximum between two arrays.
    ///
    /// \param[in] lhs input array
    /// \param[in] rhs input array
    /// \return maximum of \p lhs and \p rhs
    ///
    /// \ingroup arith_func_max
    AFAPI array max    (const array &lhs, const array &rhs);

    /// C++ Interface to find the elementwise maximum between an array and a scalar value.
    ///
    /// \param[in] lhs input array
    /// \param[in] rhs scalar value
    /// \return maximum of \p lhs and \p rhs
    ///
    /// \ingroup arith_func_max
    AFAPI array max    (const array &lhs, const double rhs);

    /// C++ Interface to find the elementwise maximum between an array and a scalar value.
    ///
    /// \param[in] lhs input array
    /// \param[in] rhs scalar value
    /// \return maximum of \p lhs and \p rhs
    ///
    /// \ingroup arith_func_max
    AFAPI array max    (const double lhs, const array &rhs);

#if AF_API_VERSION >= 34
    /// @{
    /// C++ Interface to clamp an array between an upper and a lower limit.
    ///
    /// \param[in] in input array
    /// \param[in] lo lower limit; can be an array or a scalar
    /// \param[in] hi upper limit; can be an array or a scalar
    /// \return array containing values from \p in clamped between \p lo and \p hi
    /// 
    /// \ingroup arith_func_clamp
    AFAPI array clamp(const array &in, const array &lo, const array &hi);
#endif

#if AF_API_VERSION >= 34
    /// \copydoc clamp(const array&, const array&, const array&)
    AFAPI array clamp(const array &in, const array &lo, const double hi);
#endif

#if AF_API_VERSION >= 34
    /// \copydoc clamp(const array&, const array&, const array&)
    AFAPI array clamp(const array &in, const double lo, const array &hi);
#endif

#if AF_API_VERSION >= 34
    /// \copydoc clamp(const array&, const array&, const array&)
    AFAPI array clamp(const array &in, const double lo, const double hi);
#endif
    /// @}

    /// @{
    /// C++ Interface to calculate the remainder.
    ///
    /// \param[in] lhs numerator; can be an array or a scalar
    /// \param[in] rhs denominator; can be an array or a scalar
    /// \return remainder of \p lhs divided by \p rhs
    /// 
    /// \ingroup arith_func_rem
    AFAPI array rem    (const array &lhs, const array &rhs);

    /// \copydoc rem(const array&, const array&)
    AFAPI array rem    (const array &lhs, const double rhs);

    /// \copydoc rem(const array&, const array&)
    AFAPI array rem    (const double lhs, const array &rhs);
    /// @}

    /// @{
    /// C++ Interface to calculate the modulus.
    ///
    /// \param[in] lhs dividend; can be an array or a scalar
    /// \param[in] rhs divisor; can be an array or a scalar
    /// \return \p lhs modulo \p rhs
    /// 
    /// \ingroup arith_func_mod
    AFAPI array mod    (const array &lhs, const array &rhs);

    /// \copydoc mod(const array&, const array&)
    AFAPI array mod    (const array &lhs, const double rhs);

    /// \copydoc mod(const array&, const array&)
    AFAPI array mod    (const double lhs, const array &rhs);
    /// @}

    /// C++ Interface to calculate the absolute value.
    ///
    /// \param[in] in input array
    /// \return absolute value
    ///
    /// \ingroup arith_func_abs
    AFAPI array abs    (const array &in);

    /// C++ Interface to calculate the phase angle (in radians) of a complex array.
    ///
    /// \param[in] in input array, typically complex
    /// \return phase angle (in radians)
    /// 
    /// \ingroup arith_func_arg
    AFAPI array arg    (const array &in);

    /// C++ Interface to return the sign of elements in an array.
    ///
    /// \param[in] in input array
    /// \return array containing 1's for negative values; 0's otherwise
    /// 
    /// \ingroup arith_func_sign
    AFAPI array sign  (const array &in);

    /// C++ Interface to round numbers.
    ///
    /// \param[in] in input array
    /// \return numbers rounded to nearest integer
    ///
    /// \ingroup arith_func_round
    AFAPI array round  (const array &in);

    /// C++ Interface to truncate numbers.
    ///
    /// \param[in] in input array
    /// \return nearest integer not greater in magnitude than \p in
    /// 
    /// \ingroup arith_func_trunc
    AFAPI array trunc  (const array &in);

    /// C++ Interface to floor numbers.
    ///
    /// \param[in] in input array
    /// \return values rounded to nearest integer less than or equal to current value
    ///
    /// \ingroup arith_func_floor
    AFAPI array floor  (const array &in);

    /// C++ Interface to ceil numbers.
    ///
    /// \param[in] in input array
    /// \return values rounded to nearest integer greater than or equal to current value
    ///
    /// \ingroup arith_func_ceil
    AFAPI array ceil   (const array &in);

    /// \ingroup arith_func_hypot
    /// @{
    /// C++ Interface to calculate the length of the hypotenuse of two inputs.
    ///
    /// Calculates the hypotenuse of two inputs. The inputs can be both arrays
    /// or an array and a scalar.
    ///
    /// \param[in] lhs length of first side
    /// \param[in] rhs length of second side
    /// \return length of the hypotenuse
    AFAPI array hypot  (const array &lhs, const array &rhs);

    /// \copydoc hypot(const array&, const array&)
    AFAPI array hypot  (const array &lhs, const double rhs);

    /// \copydoc hypot(const array&, const array&)
    AFAPI array hypot  (const double lhs, const array &rhs);
    /// @}

    /// C++ Interface to evaluate the sine function.
    ///
    /// \param[in] in input array
    /// \return sine
    ///
    /// \ingroup arith_func_sin
    AFAPI array sin    (const array &in);

    /// C++ Interface to evaluate the cosine function.
    ///
    /// \param[in] in input array
    /// \return cosine
    ///
    /// \ingroup arith_func_cos
    AFAPI array cos    (const array &in);

    /// C++ Interface to evaluate the tangent function.
    ///
    /// \param[in] in input array
    /// \return tangent
    ///
    /// \ingroup arith_func_tan
    AFAPI array tan    (const array &in);

    /// C++ Interface to evaluate the inverse sine function.
    ///
    /// \param[in] in input array
    /// \return inverse sine
    ///
    /// \ingroup arith_func_asin
    AFAPI array asin   (const array &in);

    /// C++ Interface to evaluate the inverse cosine function.
    ///
    /// \param[in] in input array
    /// \return inverse cosine
    ///
    /// \ingroup arith_func_acos
    AFAPI array acos   (const array &in);

    /// C++ Interface to evaluate the inverse tangent function.
    ///
    /// \param[in] in input array
    /// \return inverse tangent
    ///
    /// \ingroup arith_func_atan
    AFAPI array atan   (const array &in);

    /// \ingroup arith_func_atan
    /// @{
    /// C++ Interface to evaluate the inverse tangent of two arrays.
    ///
    /// \param[in] lhs value of numerator
    /// \param[in] rhs value of denominator
    /// \return inverse tangent of the inputs
    AFAPI array atan2  (const array &lhs, const array &rhs);

    /// \copydoc atan2(const array&, const array&)
    AFAPI array atan2  (const array &lhs, const double rhs);

    /// \copydoc atan2(const array&, const array&)
    AFAPI array atan2  (const double lhs, const array &rhs);
    /// @}

    /// C++ Interface to evaluate the hyperbolic sine function.
    ///
    /// \param[in] in input array
    /// \return hyperbolic sine
    ///
    /// \ingroup arith_func_sinh
    AFAPI array sinh(const array& in);

    /// C++ Interface to evaluate the hyperbolic cosine function.
    ///
    /// \param[in] in input array
    /// \return hyperbolic cosine
    ///
    /// \ingroup arith_func_cosh
    AFAPI array cosh(const array& in);

    /// C++ Interface to evaluate the hyperbolic tangent function.
    ///
    /// \param[in] in input array
    /// \return hyperbolic tangent
    ///
    /// \ingroup arith_func_tanh
    AFAPI array tanh(const array& in);

    /// C++ Interface to evaluate the inverse hyperbolic sine function.
    ///
    /// \param[in] in input array
    /// \return inverse hyperbolic sine
    ///
    /// \ingroup arith_func_asinh
    AFAPI array asinh(const array& in);

    /// C++ Interface to evaluate the inverse hyperbolic cosine function.
    ///
    /// \param[in] in input array
    /// \return inverse hyperbolic cosine
    ///
    /// \ingroup arith_func_acosh
    AFAPI array acosh(const array& in);

    /// C++ Interface to evaluate the inverse hyperbolic tangent function.
    ///
    /// \param[in] in input array
    /// \return inverse hyperbolic tangent
    ///
    /// \ingroup arith_func_atanh
    AFAPI array atanh(const array& in);

    /// \ingroup arith_func_cplx
    /// @{
    /// C++ Interface to create a complex array from a single real array.
    ///
    /// \param[in] in a real array
    /// \return the returned complex array
    AFAPI array complex(const array& in);
 
    /// C++ Interface to create a complex array from two real arrays.
    ///
    /// \param[in] real_ a real array to be assigned as the real component of the returned complex array
    /// \param[in] imag_ a real array to be assigned as the imaginary component of the returned complex array
    /// \return the returned complex array
    AFAPI array complex(const array &real_, const array &imag_);

    /// C++ Interface to create a complex array from a single real array for the real component and a single scalar for each imaginary component.
    ///
    /// \param[in] real_ a real array to be assigned as the real component of the returned complex array
    /// \param[in] imag_ a single scalar to be assigned as the imaginary component of each value of the returned complex array
    /// \return the returned complex array
    AFAPI array complex(const array &real_, const double imag_);

    /// C++ Interface to create a complex array from a single scalar for each real component and a single real array for the imaginary component.
    ///
    /// \param[in] real_ a single scalar to be assigned as the real component of each value of the returned complex array
    /// \param[in] imag_ a real array to be assigned as the imaginary component of the returned complex array
    /// \return the returned complex array
    AFAPI array complex(const double real_, const array &imag_);
    /// @}

    /// C++ Interface to return the real part of a complex array.
    ///
    /// \param[in] in input complex array
    /// \return real part
    ///
    /// \ingroup arith_func_real
    AFAPI array real   (const array &in);

    /// C++ Interface to return the imaginary part of a complex array.
    ///
    /// \param[in] in input complex array
    /// \return imaginary part
    ///
    /// \ingroup arith_func_imag
    AFAPI array imag   (const array &in);

    /// C++ Interface to calculate the complex conjugate of an input array.
    ///
    /// \param[in] in input complex array
    /// \return complex conjugate
    ///
    /// \ingroup arith_func_conjg
    AFAPI array conjg  (const array &in);

    /// C++ Interface to evaluate the nth root.
    ///
    /// \param[in] nth_root nth root
    /// \param[in] value value
    /// \return \p nth_root th root of \p value
    ///
    /// \ingroup arith_func_root
    AFAPI array root    (const array &nth_root, const array &value);

    /// C++ Interface to evaluate the nth root.
    ///
    /// \param[in] nth_root nth root
    /// \param[in] value value
    /// \return \p nth_root th root of \p value
    ///
    /// \ingroup arith_func_root
    AFAPI array root    (const array &nth_root, const double value);

    /// C++ Interface to evaluate the nth root.
    ///
    /// \param[in] nth_root nth root
    /// \param[in] value value
    /// \return \p nth_root th root of \p value
    ///
    /// \ingroup arith_func_root
    AFAPI array root    (const double nth_root, const array &value);


    /// \ingroup arith_func_pow
    /// @{
    /// C++ Interface to raise a base to a power (or exponent).
    ///
    /// Computes the value of \p base raised to the power of \p exponent. The inputs can be two arrays or an array and a scalar.
    ///
    /// \param[in] base base
    /// \param[in] exponent exponent
    /// \return \p base raised to the power of \p exponent
    AFAPI array pow    (const array &base, const array &exponent);

    /// \copydoc pow(const array&, const array&)
    AFAPI array pow    (const array &base, const double exponent);

    /// \copydoc pow(const array&, const array&)
    AFAPI array pow    (const double base, const array &exponent);

    /// C++ Interface to raise 2 to a power (or exponent).
    ///
    /// \param[in] in exponent
    /// \return 2 raised to the power
    ///
    AFAPI array pow2    (const array &in);
    /// @}

#if AF_API_VERSION >= 31
    /// C++ Interface to evaluate the logistical sigmoid function.
    ///
    /// \param[in] in input
    /// \return sigmoid
    /// 
    /// \note Computes `1/(1+e^-x)`.
    ///
    /// \ingroup arith_func_sigmoid
    AFAPI array sigmoid (const array &in);
#endif

    /// C++ Interface to evaluate the exponential.
    ///
    /// \param[in] in exponent
    /// \return exponential
    ///
    /// \ingroup arith_func_exp
    AFAPI array exp    (const array &in);

    /// C++ Interface to evaluate the exponential of an array minus 1, `exp(in) - 1`.
    ///
    /// \param[in] in exponent
    /// \return the exponential minus 1
    ///
    /// \note This function is useful when \p in is small
    /// \ingroup arith_func_expm1
    AFAPI array expm1  (const array &in);

    /// C++ Interface to evaluate the error function.
    ///
    /// \param[in] in input
    /// \return error function
    ///
    /// \ingroup arith_func_erf
    AFAPI array erf    (const array &in);

    /// C++ Interface to evaluate the complementary error function.
    ///
    /// \param[in] in input
    /// \return complementary error function
    ///
    /// \ingroup arith_func_erfc
    AFAPI array erfc   (const array &in);

    /// C++ Interface to evaluate the natural logarithm.
    ///
    /// \param[in] in input
    /// \return natural logarithm
    ///
    /// \ingroup arith_func_log
    AFAPI array log    (const array &in);

    /// C++ Interface to evaluate the natural logarithm of 1 + input, `ln(1+in)`.
    ///
    /// \param[in] in input
    /// \return natural logarithm of `1 + input`
    ///
    /// \note This function is useful when \p in is small
    /// \ingroup arith_func_log1p
    AFAPI array log1p  (const array &in);

    /// C++ Interface to evaluate the base 10 logarithm.
    ///
    /// \param[in] in input
    /// \return base 10 logarithm
    ///
    /// \ingroup arith_func_log10
    AFAPI array log10  (const array &in);

    /// C++ Interface to evaluate the base 2 logarithm.
    ///
    /// \param[in] in input
    /// \return base 2 logarithm
    ///
    /// \ingroup explog_func_log2
    AFAPI array log2   (const array &in);

    /// C++ Interface to evaluate the square root.
    ///
    /// \param[in] in input
    /// \return square root
    ///
    /// \ingroup arith_func_sqrt
    AFAPI array sqrt   (const array &in);

#if AF_API_VERSION >= 37
    /// C++ Interface to evaluate the reciprocal square root.
    ///
    /// \param[in] in input
    /// \return reciprocal square root
    ///
    /// \ingroup arith_func_rsqrt
    AFAPI array rsqrt   (const array &in);
#endif

    /// C++ Interface to evaluate the cube root.
    ///
    /// \param[in] in input
    /// \return cube root
    ///
    /// \ingroup arith_func_cbrt
    AFAPI array cbrt   (const array &in);

    /// C++ Interface to calculate the factorial.
    ///
    /// \param[in] in input
    /// \return the factorial function
    ///
    /// \ingroup arith_func_factorial
    AFAPI array factorial (const array &in);

    /// C++ Interface to evaluate the gamma function.
    ///
    /// \param[in] in input
    /// \return gamma function
    ///
    /// \ingroup arith_func_tgamma
    AFAPI array tgamma (const array &in);

    /// C++ Interface to evaluate the logarithm of the absolute value of the gamma function.
    ///
    /// \param[in] in input
    /// \return logarithm of the absolute value of the gamma function
    ///
    /// \ingroup arith_func_lgamma
    AFAPI array lgamma (const array &in);

    /// C++ Interface to check which values are zero.
    ///
    /// \param[in] in input
    /// \return array containing 1's where input is 0; 0's otherwise
    ///
    /// \ingroup arith_func_iszero
    AFAPI array iszero (const array &in);

    /// C++ Interface to check if values are infinite.
    ///
    /// \param[in] in input
    /// \return array containing 1's where input is Inf or -Inf; 0's otherwise
    ///
    /// \ingroup arith_func_isinf
    AFAPI array isInf  (const array &in);

    /// C++ Interface to check if values are NaN.
    ///
    /// \param[in] in input
    /// \return array containing 1's where input is NaN; 0's otherwise
    ///
    /// \ingroup arith_func_isnan
    AFAPI array isNaN  (const array &in);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface to add two arrays.

       \param[out] out sum of \p lhs and \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_add
    */
    AFAPI af_err af_add   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to subtract one array from another array.

       \param[out] out subtraction of \p lhs - \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_sub
    */
    AFAPI af_err af_sub   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to multiply two arrays.

       \param[out] out product of \p lhs and \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_mul
    */
    AFAPI af_err af_mul   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to divide one array by another array.

       \param[out] out result of \p lhs / \p rhs.
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_div
    */
    AFAPI af_err af_div   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to perform a less-than comparison between corresponding elements of two arrays.

       \param[out] out result of \p lhs < \p rhs; type is b8
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup logic_func_lt
    */
    AFAPI af_err af_lt    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to perform a greater-than comparison between corresponding elements of two arrays.

       \param[out] out result of \p lhs > \p rhs; type is b8
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_gt
    */
    AFAPI af_err af_gt    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to perform a less-than-or-equal comparison between corresponding elements of two arrays.

       \param[out] out result of \p lhs <= \p rhs; type is b8
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_le
    */
    AFAPI af_err af_le    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to perform a greater-than-or-equal comparison between corresponding elements of two arrays.

       \param[out] out result of \p lhs >= \p rhs; type is b8
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_ge
    */
    AFAPI af_err af_ge    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to check if corresponding elements of two arrays are equal

       \param[out] out result of `lhs == rhs`; type is b8
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_eq
    */
    AFAPI af_err af_eq    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to check if corresponding elements of two arrays are not equal

       \param[out] out result of `lhs != rhs`; type is b8
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_neq
    */
    AFAPI af_err af_neq   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to evaluate the logical AND of two arrays.

       \param[out] out result of \p lhs && \p rhs; type is b8
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_and
    */
    AFAPI af_err af_and   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface the evaluate the logical OR of two arrays.

       \param[out] out result of \p lhs || \p rhs; type is b8
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_or
    */
    AFAPI af_err af_or    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to evaluate the logical NOT of an array.

       \param[out] out result of logical NOT; type is b8
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_not
    */
    AFAPI af_err af_not   (af_array *out, const af_array in);

#if AF_API_VERSION >= 38
    /**
       C Interface to evaluate the bitwise NOT of an array.

       \param[out] out result of bitwise NOT
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_bitnot
    */
    AFAPI af_err af_bitnot   (af_array *out, const af_array in);
#endif

    /**
       C Interface to evaluate the bitwise AND of two arrays.

       \param[out] out result of \p lhs & \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_bitand
    */
    AFAPI af_err af_bitand   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to evaluate the bitwise OR of two arrays.

       \param[out] out result of \p lhs | \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_bitor
    */
    AFAPI af_err af_bitor    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to evaluate the bitwise XOR of two arrays.

       \param[out] out result of \p lhs ^ \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_bitxor
    */
    AFAPI af_err af_bitxor   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to shift the bits of integer arrays left.

       \param[out] out result of the left shift
       \param[in] lhs values to shift
       \param[in] rhs n bits to shift
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_shiftl
    */
    AFAPI af_err af_bitshiftl(af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to shift the bits of integer arrays right.

       \param[out] out result of the right shift
       \param[in] lhs values to shift
       \param[in] rhs n bits to shift
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_shiftr
    */
    AFAPI af_err af_bitshiftr(af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to cast an array from one type to another.

       This function casts an af_array object from one type to another. If the
       type of the original array is the same as \p type then the same array is
       returned.

       \note Consecitive casting operations may be may be optimized out if the
       original type of the af_array is the same as the final type. For example
       if the original type is f64 which is then cast to f32 and then back to
       f64, then the cast to f32 will be skipped and that operation will *NOT*
       be performed by ArrayFire. The following table shows which casts will
       be optimized out. outer -> inner -> outer
       | inner-> | f32 | f64 | c32 | c64 | s32 | u32 | u8 | b8 | s64 | u64 | s16 | u16 | f16 |
       |---------|-----|-----|-----|-----|-----|-----|----|----|-----|-----|-----|-----|-----|
       | f32     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
       | f64     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
       | c32     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
       | c64     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
       | s32     | x   | x   | x   | x   | x   | x   |    |    | x   | x   |     |     | x   |
       | u32     | x   | x   | x   | x   | x   | x   |    |    | x   | x   |     |     | x   |
       | u8      | x   | x   | x   | x   | x   | x   | x  | x  | x   | x   | x   | x   | x   |
       | b8      | x   | x   | x   | x   | x   | x   | x  | x  | x   | x   | x   | x   | x   |
       | s64     | x   | x   | x   | x   |     |     |    |    | x   | x   |     |     | x   |
       | u64     | x   | x   | x   | x   |     |     |    |    | x   | x   |     |     | x   |
       | s16     | x   | x   | x   | x   | x   | x   |    |    | x   | x   | x   | x   | x   |
       | u16     | x   | x   | x   | x   | x   | x   |    |    | x   | x   | x   | x   | x   |
       | f16     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
       If you want to avoid this behavior use af_eval after the first cast
       operation. This will ensure that the cast operation is performed on the
       af_array.

       \param[out] out values in the specified type
       \param[in] in input
       \param[in] type target data type \ref af_dtype
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cast
    */
    AFAPI af_err af_cast    (af_array *out, const af_array in, const af_dtype type);

    /**
       C Interface to find the elementwise minimum between two arrays.

       \param[out] out minimum of \p lhs and \p rhs
       \param[in] lhs input array
       \param[in] rhs input array
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_min
    */
    AFAPI af_err af_minof (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to find the elementwise minimum between an array and a scalar value.

       \param[out] out maximum of \p lhs and \p rhs
       \param[in] lhs input array
       \param[in] rhs input array
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_max
    */
    AFAPI af_err af_maxof (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

#if AF_API_VERSION >= 34
    /**
       C Interface to clamp an array between an upper and a lower limit.

       \param[out] out array containing values from \p in clamped between \p lo and \p hi
       \param[in] in input array
       \param[in] lo lower limit array
       \param[in] hi upper limit array
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_clamp
    */
    AFAPI af_err af_clamp(af_array *out, const af_array in,
                          const af_array lo, const af_array hi, const bool batch);
#endif

    /**
       C Interface to calculate the remainder.

       \param[out] out remainder of \p lhs divided by \p rhs
       \param[in] lhs numerator
       \param[in] rhs denominator
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_rem
    */
    AFAPI af_err af_rem   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to calculate the modulus.

       \param[out] out \p lhs modulo \p rhs
       \param[in] lhs dividend
       \param[in] rhs divisor
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_mod
    */
    AFAPI af_err af_mod   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to calculate the absolute value.

       \param[out] out absolute value
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_abs
    */
    AFAPI af_err af_abs     (af_array *out, const af_array in);

    /**
       C Interface to calculate the phase angle (in radians) of a complex array.

       \param[out] out phase angle (in radians)
       \param[in] in input array, typically complex
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_arg
    */
    AFAPI af_err af_arg     (af_array *out, const af_array in);

    /**
       C Interface to calculate the sign of elements in an array.

       \param[out] out array containing 1's for negative values; 0's otherwise
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_sign
    */
    AFAPI af_err af_sign   (af_array *out, const af_array in);

    /**
       C Interface to round numbers.

       \param[out] out values rounded to nearest integer
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_round
    */
    AFAPI af_err af_round   (af_array *out, const af_array in);

    /**
       C Interface to truncate numbers.

       \param[out] out nearest integer not greater in magnitude than \p in
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_trunc
    */
    AFAPI af_err af_trunc   (af_array *out, const af_array in);

    /**
       C Interface to floor numbers.

       \param[out] out values rounded to nearest integer less than or equal to \p in
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_floor
    */
    AFAPI af_err af_floor   (af_array *out, const af_array in);

    /**
       C Interface to ceil numbers.

       \param[out] out values rounded to nearest integer greater than or equal to \p in
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_ceil
    */
    AFAPI af_err af_ceil    (af_array *out, const af_array in);

    /**
       C Interface to calculate the length of the hypotenuse of two inputs.

       \param[out] out length of the hypotenuse
       \param[in] lhs length of first side
       \param[in] rhs length of second side
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_floor
    */
    AFAPI af_err af_hypot (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to evaluate the sine function.

       \param[out] out sine
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_sin
    */
    AFAPI af_err af_sin     (af_array *out, const af_array in);

    /**
       C Interface to evaluate the cosine function.

       \param[out] out cosine
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cos
    */
    AFAPI af_err af_cos     (af_array *out, const af_array in);

    /**
       C Interface to evaluate the tangent function.

       \param[out] out tangent
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_tan
    */
    AFAPI af_err af_tan     (af_array *out, const af_array in);

    /**
       C Interface to evaluate the inverse sine function.

       \param[out] out inverse sine
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_asin
    */
    AFAPI af_err af_asin    (af_array *out, const af_array in);

    /**
       C Interface to evaluate the inverse cosine function.

       \param[out] out inverse cos
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_acos
    */
    AFAPI af_err af_acos    (af_array *out, const af_array in);

    /**
       C Interface to evaluate the inverse tangent function.

       \param[out] out inverse tangent
       \param[in] in input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_atan
    */
    AFAPI af_err af_atan    (af_array *out, const af_array in);

    /**
       C Interface to evaluate the inverse tangent of two arrays.

       \param[out] out inverse tangent of two arrays
       \param[in] lhs numerator
       \param[in] rhs denominator
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_atan
    */
    AFAPI af_err af_atan2 (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to evaluate the hyperbolic sine function.

       \param[out] out hyperbolic sine
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_sinh
    */
    AFAPI af_err af_sinh    (af_array *out, const af_array in);

    /**
       C Interface to evaluate the hyperbolic cosine function.

       \param[out] out hyperbolic cosine
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cosh
    */
    AFAPI af_err af_cosh    (af_array *out, const af_array in);

    /**
       C Interface to evaluate the hyperbolic tangent function.

       \param[out] out hyperbolic tangent
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_tanh
    */
    AFAPI af_err af_tanh    (af_array *out, const af_array in);

    /**
       C Interface to evaluate the inverse hyperbolic sine function.

       \param[out] out inverse hyperbolic sine
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_asinh
    */
    AFAPI af_err af_asinh   (af_array *out, const af_array in);

    /**
       C Interface to evaluate the inverse hyperbolic cosine function.

       \param[out] out inverse hyperbolic cosine
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_acosh
    */
    AFAPI af_err af_acosh   (af_array *out, const af_array in);

    /**
       C Interface to evaluate the inverse hyperbolic tangent function.

       \param[out] out inverse hyperbolic tangent
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_atanh
    */
    AFAPI af_err af_atanh   (af_array *out, const af_array in);

    /**
       C Interface to create a complex array from a single real array.

       \param[out] out complex array
       \param[in] in real array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cplx
    */
    AFAPI af_err af_cplx(af_array* out, const af_array in);

    /**
       C Interface to create a complex array from two real arrays.

       \param[out] out complex array
       \param[in] real real array to be assigned as the real component of the returned complex array
       \param[in] imag real array to be assigned as the imaginary component of the returned complex array
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cplx
    */
    AFAPI af_err af_cplx2(af_array* out, const af_array real, const af_array imag, const bool batch);

    /**
       C Interface to return the real part of a complex array.

       \param[out] out real part
       \param[in] in complex array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_real
    */
    AFAPI af_err af_real(af_array* out, const af_array in);

    /**
       C Interface to return the imaginary part of a complex array.

       \param[out] out imaginary part
       \param[in] in complex array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_imag
    */
    AFAPI af_err af_imag(af_array* out, const af_array in);

    /**
       C Interface to evaluate the complex conjugate of an input array.

       \param[out] out complex conjugate
       \param[in] in complex array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_conjg
    */
    AFAPI af_err af_conjg(af_array* out, const af_array in);

    /**
       C Interface to evaluate the nth root.

       \param[out] out \p lhs th root of \p rhs
       \param[in] lhs nth root
       \param[in] rhs value
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_root
    */
    AFAPI af_err af_root   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);


    /**
       C Interface to raise a base to a power (or exponent).

       \param[out] out \p lhs raised to the power of \p rhs
       \param[in] lhs base
       \param[in] rhs exponent
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_pow
    */
    AFAPI af_err af_pow   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface to raise 2 to a power (or exponent).

       \param[out] out 2 raised to the power of \p in
       \param[in] in exponent
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_pow2
    */
    AFAPI af_err af_pow2     (af_array *out, const af_array in);

#if AF_API_VERSION >= 31
    /**
       C Interface to evaluate the logistical sigmoid function.

       Computes `1/(1+e^-x)`.

       \param[out] out output of the logistic sigmoid function
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_sigmoid
    */
    AFAPI af_err af_sigmoid(af_array* out, const af_array in);
#endif

    /**
       C Interface to evaluate the exponential.

       \param[out] out e raised to the power of \p in
       \param[in] in exponent
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_exp
    */
    AFAPI af_err af_exp     (af_array *out, const af_array in);

    /**
       C Interface to evaluate the exponential of an array minus 1, `exp(in) - 1`.

       \param[out] out exponential of `in - 1`
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_expm1
    */
    AFAPI af_err af_expm1   (af_array *out, const af_array in);

    /**
       C Interface to evaluate the error function.

       \param[out] out error function value
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_erf
    */
    AFAPI af_err af_erf     (af_array *out, const af_array in);

    /**
       C Interface to evaluate the complementary error function.

       \param[out] out complementary error function
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_erfc
    */
    AFAPI af_err af_erfc    (af_array *out, const af_array in);

    /**
       C Interface to evaluate the natural logarithm.

       \param[out] out natural logarithm
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_log
    */
    AFAPI af_err af_log     (af_array *out, const af_array in);

    /**
       C Interface to evaluate the natural logarithm of 1 + input, `ln(1+in)`.

       \param[out] out logarithm of `in + 1`
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_log1p
    */
    AFAPI af_err af_log1p   (af_array *out, const af_array in);

    /**
       C Interface to evaluate the base 10 logarithm.

       \param[out] out base 10 logarithm
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_log10
    */
    AFAPI af_err af_log10   (af_array *out, const af_array in);

    /**
       C Interface to evaluate the base 2 logarithm.

       \param[out] out base 2 logarithm
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup explog_func_log2
    */
    AFAPI af_err af_log2   (af_array *out, const af_array in);

    /**
       C Interface to evaluate the square root.

       \param[out] out square root
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_sqrt
    */
    AFAPI af_err af_sqrt    (af_array *out, const af_array in);

#if AF_API_VERSION >= 37
    /**
      C Interface to evaluate the reciprocal square root.

      \param[out] out reciprocal square root
      \param[in] in input
      \return \ref AF_SUCCESS if the execution completes properly

      \ingroup arith_func_rsqrt
    */
    AFAPI af_err af_rsqrt    (af_array *out, const af_array in);
#endif
    /**
       C Interface to evaluate the cube root.

       \param[out] out cube root
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cbrt
    */
    AFAPI af_err af_cbrt    (af_array *out, const af_array in);

    /**
       C Interface to calculate the factorial.

       \param[out] out factorial
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_factorial
    */
    AFAPI af_err af_factorial   (af_array *out, const af_array in);

    /**
       C Interface to evaluate the gamma function.

       \param[out] out gamma function
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_tgamma
    */
    AFAPI af_err af_tgamma   (af_array *out, const af_array in);

    /**
       C Interface to evaluate the logarithm of the absolute value of the gamma function.

       \param[out] out logarithm of the absolute value of the gamma function
       \param[in] in input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_lgamma
    */
    AFAPI af_err af_lgamma   (af_array *out, const af_array in);

    /**
        C Interface to check if values are zero.

        \param[out] out array containing 1's where input is 0; 0's otherwise
        \param[in] in input
        \return \ref AF_SUCCESS if the execution completes properly

        \ingroup arith_func_iszero
    */
    AFAPI af_err af_iszero  (af_array *out, const af_array in);

    /**
        C Interface to check if values are infinite.

        \param[out] out array containing 1's where input is Inf or -Inf; 0's otherwise
        \param[in] in input
        \return \ref AF_SUCCESS if the execution completes properly

        \ingroup arith_func_isinf
    */
    AFAPI af_err af_isinf   (af_array *out, const af_array in);

    /**
        C Interface to check if values are NaN.

        \param[out] out array containing 1's where input is NaN; 0's otherwise
        \param[in] in input
        \return \ref AF_SUCCESS if the execution completes properly

        \ingroup arith_func_isnan
    */
    AFAPI af_err af_isnan   (af_array *out, const af_array in);

#ifdef __cplusplus
}
#endif
