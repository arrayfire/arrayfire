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

    /// \ingroup arith_func_min
    /// @{
    /// \brief C++ interface for min of two arrays
    ///
    /// \param[in] lhs first input
    /// \param[in] rhs second input
    /// \return minimum of \p lhs and \p rhs
    ///
    AFAPI array min    (const array &lhs, const array &rhs);

    /// \copydoc min(const array&, const array &)
    AFAPI array min    (const array &lhs, const double rhs);

    /// \copydoc min(const array&, const array &)
    AFAPI array min    (const double lhs, const array &rhs);
    /// @}

    /// \ingroup arith_func_max
    /// @{
    /// C++ Interface for max of two arrays or an array and a scalar
    ///
    /// \param[in] lhs first input
    /// \param[in] rhs second input
    /// \return maximum of \p lhs and \p rhs
    AFAPI array max    (const array &lhs, const array &rhs);

    /// \copydoc max(const array&, const array&)
    AFAPI array max    (const array &lhs, const double rhs);

    /// \copydoc max(const array&, const array&)
    AFAPI array max    (const double lhs, const array &rhs);
    /// @}

    /// \ingroup arith_func_rem
    /// @{
    /// C++ Interface for remainder when array divides array,
    /// scalar divides array or array divides scalar
    ///
    /// \param[in] lhs is numerator
    /// \param[in] rhs is denominator
    /// \return remainder when \p rhs divides \p lhs
    AFAPI array rem    (const array &lhs, const array &rhs);

    /// \copydoc rem(const array&, const array&)
    AFAPI array rem    (const array &lhs, const double rhs);

    /// \copydoc rem(const array&, const array&)
    AFAPI array rem    (const double lhs, const array &rhs);
    /// @}

    /// \ingroup arith_func_mod
    /// @{
    /// C++ Interface for modulus when dividend and divisor are arrays
    /// or one of them is scalar
    ///
    /// \param[in] lhs is dividend
    /// \param[in] rhs is divisor
    /// \return \p lhs modulo \p rhs
    AFAPI array mod    (const array &lhs, const array &rhs);

    /// \copydoc mod(const array&, const array&)
    AFAPI array mod    (const array &lhs, const double rhs);

    /// \copydoc mod(const array&, const array&)
    AFAPI array mod    (const double lhs, const array &rhs);
    /// @}

    /// C++ Interface for absolute value
    ///
    /// \param[in] in is input array
    /// \return absolute value of \p in
    ///
    /// \ingroup arith_func_abs
    AFAPI array abs    (const array &in);

    /**
       C++ Interface for arg

       \param[in] in is input array
       \return phase of \p in

       \ingroup arith_func_arg
    */
    AFAPI array arg    (const array &in);

    /**
       C++ Interface for getting the sign of input

       \param[in] in is input array
       \return the sign of each element of input

       \note output is 1 for negative numbers and 0 for positive numbers

       \ingroup arith_func_sign
    */
    AFAPI array sign  (const array &in);

    ///C++ Interface for rounding an array of numbers
    ///
    ///\param[in] in is input array
    ///\return values rounded to nearest integer
    ///
    ///\note The values are rounded to nearest integer
    ///
    ///\ingroup arith_func_round
    AFAPI array round  (const array &in);

    /**
       C++ Interface for truncating an array of numbers

       \param[in] in is input array
       \return values truncated to nearest integer not greater than input values

       \ingroup arith_func_trunc
    */
    AFAPI array trunc  (const array &in);


    /// C++ Interface for flooring an array of numbers
    ///
    /// \param[in] in is input array
    /// \return values rounded to nearest integer less than or equal to current value
    ///
    /// \ingroup arith_func_floor
    AFAPI array floor  (const array &in);

    /// C++ Interface for ceiling an array of numbers
    ///
    /// \param[in] in is input array
    /// \return values rounded to nearest integer greater than or equal to current value
    ///
    /// \ingroup arith_func_ceil
    AFAPI array ceil   (const array &in);

    /// \ingroup arith_func_hypot
    /// @{
    /// \brief C++ Interface for getting length of hypotenuse of two inputs
    ///
    /// Calculates the hypotenuse of two inputs. The inputs can be both arrays
    /// or an array and a scalar.
    ///
    /// \param[in] lhs is the length of first side
    /// \param[in] rhs is the length of second side
    /// \return the length of the hypotenuse
    AFAPI array hypot  (const array &lhs, const array &rhs);

    /// \copydoc hypot(const array&, const array&)
    AFAPI array hypot  (const array &lhs, const double rhs);

    /// \copydoc hypot(const array&, const array&)
    AFAPI array hypot  (const double lhs, const array &rhs);
    /// @}

    /// C++ Interface for sin
    ///
    /// \param[in] in is input array
    /// \return sin of input
    ///
    /// \ingroup arith_func_sin
    AFAPI array sin    (const array &in);

    /// C++ Interface for cos
    ///
    /// \param[in] in is input array
    /// \return cos of input
    ///
    /// \ingroup arith_func_cos
    AFAPI array cos    (const array &in);

    /// C++ Interface for tan
    ///
    /// \param[in] in is input array
    /// \return tan of input
    ///
    /// \ingroup arith_func_tan
    AFAPI array tan    (const array &in);

    /// C++ Interface for arc sin (sin inverse)
    ///
    /// \param[in] in is input array
    /// \return arc sin of input
    ///
    /// \ingroup arith_func_asin
    AFAPI array asin   (const array &in);

    /// C++ Interface for arc cos (cos inverse)
    ///
    /// \param[in] in is input array
    /// \return arc cos of input
    ///
    /// \ingroup arith_func_acos
    AFAPI array acos   (const array &in);

    /// C++ Interface for arc tan (tan inverse)
    ///
    /// \param[in] in is input array
    /// \return arc tan of input
    ///
    /// \ingroup arith_func_atan
    AFAPI array atan   (const array &in);

    /// \ingroup arith_func_atan
    /// @{
    /// C++ Interface for arc tan of two arrays
    ///
    /// \param[in] lhs value of numerator
    /// \param[in] rhs value of denominator
    /// \return arc tan of the inputs
    AFAPI array atan2  (const array &lhs, const array &rhs);

    /// \copydoc atan2(const array&, const array&)
    AFAPI array atan2  (const array &lhs, const double rhs);

    /// \copydoc atan2(const array&, const array&)
    AFAPI array atan2  (const double lhs, const array &rhs);
    /// @}

    /// \ingroup trig_func_cplx2
    /// @{
    /// C++ Interface for creating complex array from two inputs
    ///
    /// Creates a complex number from two sets of inputs. The left hand side is
    /// the real part and the right hand side is the imaginary part. This
    /// function accepts two \ref af::array or one \ref af::array and a scalar
    /// as nputs.
    ///
    /// \param[in] lhs is real value(s)
    /// \param[in] rhs is imaginary value(s)
    /// \return complex array from inputs
    AFAPI array complex(const array &lhs, const array &rhs);

    /// \copydoc complex(const array&, const array&)
    AFAPI array complex(const array &lhs, const double rhs);

    /// \copydoc complex(const array&, const array&)
    AFAPI array complex(const double lhs, const array &rhs);

    /// C++ Interface for creating complex array from real array
    ///
    /// \param[in] in is real array
    /// \return complex array from \p in
    ///
    /// \ingroup arith_func_cplx
    AFAPI array complex(const array &in);
    /// @}

    /// C++ Interface for getting real part from complex array
    ///
    /// \param[in] in is complex array
    /// \return the real part of \p in
    ///
    /// \ingroup arith_func_real
    AFAPI array real   (const array &in);

    /// C++ Interface for getting imaginary part from complex array
    ///
    /// \param[in] in is complex array
    /// \return the imaginary part of \p in
    ///
    /// \ingroup arith_func_imag
    AFAPI array imag   (const array &in);

    /// C++ Interface for getting the complex conjugate of input array
    ///
    /// \param[in] in is complex array
    /// \return the complex conjugate of \p in
    ///
    /// \ingroup arith_func_conjg
    AFAPI array conjg  (const array &in);

    /// C++ Interface for sinh
    ///
    /// \param[in] in is input array
    /// \return sinh of input
    ///
    /// \ingroup arith_func_sinh
    AFAPI array sinh    (const array &in);

    /// C++ Interface for cosh
    ///
    /// \param[in] in is input array
    /// \return cosh of input
    ///
    /// \ingroup arith_func_cosh
    AFAPI array cosh    (const array &in);

    /// C++ Interface for tanh
    ///
    /// \param[in] in is input array
    /// \return tanh of input
    ///
    /// \ingroup arith_func_tanh
    AFAPI array tanh    (const array &in);

    /// C++ Interface for sinh inverse
    ///
    /// \param[in] in is input array
    /// \return sinh inverse of input
    ///
    /// \ingroup arith_func_asinh
    AFAPI array asinh   (const array &in);

    /// C++ Interface for cosh inverse
    ///
    /// \param[in] in is input array
    /// \return cosh inverse of input
    ///
    /// \ingroup arith_func_acosh
    AFAPI array acosh   (const array &in);

    /// C++ Interface for tanh inverse
    ///
    /// \param[in] in is input array
    /// \return tanh inverse of input
    ///
    /// \ingroup arith_func_atanh
    AFAPI array atanh   (const array &in);

    /// C++ Interface for nth root
    ///
    /// \param[in] lhs is nth root
    /// \param[in] rhs is value
    /// \return \p lhs th root of \p rhs
    ///
    /// \ingroup arith_func_root
    AFAPI array root    (const array &lhs, const array &rhs);

    /// C++ Interface for nth root
    ///
    /// \param[in] lhs is nth root
    /// \param[in] rhs is value
    /// \return \p lhs th root of \p rhs
    ///
    /// \ingroup arith_func_root
    AFAPI array root    (const array &lhs, const double rhs);

    /// C++ Interface for nth root
    ///
    /// \param[in] lhs is nth root
    /// \param[in] rhs is value
    /// \return \p lhs th root of \p rhs
    ///
    /// \ingroup arith_func_root
    AFAPI array root    (const double lhs, const array &rhs);


    /// \ingroup arith_func_pow
    /// @{
    /// \brief C++ Interface for power
    ///
    /// Computes the value of \p lhs raised to the power of \p rhs. The inputs
    /// can be two arrays or an array and a scalar.
    ///
    /// \param[in] lhs is base
    /// \param[in] rhs is exponent
    /// \return \p lhs raised to power \p rhs
    AFAPI array pow    (const array &lhs, const array &rhs);

    /// \copydoc pow(const array&, const array&)
    AFAPI array pow    (const array &lhs, const double rhs);

    /// \copydoc pow(const array&, const array&)
    AFAPI array pow    (const double lhs, const array &rhs);

    /// C++ Interface for power of 2
    ///
    /// \param[in] in is exponent
    /// \return 2 raised to power of \p in
    ///
    AFAPI array pow2    (const array &in);
    /// @}


    /// C++ Interface for calculating sigmoid function of an array
    ///
    /// \param[in] in is input
    /// \return the sigmoid of \p in
    ///
    /// \ingroup arith_func_sigmoid
    AFAPI array sigmoid (const array &in);

    /// C++ Interface for exponential of an array
    ///
    /// \param[in] in is exponent
    /// \return the exponential of \p in
    ///
    /// \ingroup arith_func_exp
    AFAPI array exp    (const array &in);

    /// C++ Interface for exponential of an array minus 1
    ///
    /// \param[in] in is exponent
    /// \return the exponential of \p in - 1
    ///
    /// \note This function is useful when \p in is small
    /// \ingroup arith_func_expm1
    AFAPI array expm1  (const array &in);

    /// C++ Interface for error function value
    ///
    /// \param[in] in is input
    /// \return the error function value
    ///
    /// \ingroup arith_func_erf
    AFAPI array erf    (const array &in);

    /// C++ Interface for complementary error function value
    ///
    /// \param[in] in is input
    /// \return the complementary error function value
    ///
    /// \ingroup arith_func_erfc
    AFAPI array erfc   (const array &in);

    /// C++ Interface for natural logarithm
    ///
    /// \param[in] in is input
    /// \return the natural logarithm of input
    ///
    /// \ingroup arith_func_log
    AFAPI array log    (const array &in);

    /// C++ Interface for natural logarithm of 1 + input
    ///
    /// \param[in] in is input
    /// \return the natural logarithm of (1 + input)
    ///
    /// \note This function is useful when \p is small
    /// \ingroup arith_func_log1p
    AFAPI array log1p  (const array &in);

    /// C++ Interface for logarithm base 10
    ///
    /// \param[in] in is input
    /// \return the logarithm of input in base 10
    ///
    /// \ingroup arith_func_log10
    AFAPI array log10  (const array &in);

    /// C++ Interface for logarithm base 2
    ///
    /// \param[in] in is input
    /// \return the logarithm of input in base 2
    ///
    /// \ingroup explog_func_log2
    AFAPI array log2   (const array &in);

    /// C++ Interface for square root of input
    ///
    /// \param[in] in is input
    /// \return the square root of input
    ///
    /// \ingroup arith_func_sqrt
    AFAPI array sqrt   (const array &in);

    /// C++ Interface for cube root of input
    ///
    /// \param[in] in is input
    /// \return the cube root of input
    ///
    /// \ingroup arith_func_cbrt
    AFAPI array cbrt   (const array &in);

    ///
    /// C++ Interface for factorial of input
    ///
    /// \param[in] in is input
    /// \return the factorial function of input
    ///
    /// \ingroup arith_func_factorial
    AFAPI array factorial (const array &in);

    /// C++ Interface for gamma function of input
    ///
    /// \param[in] in is input
    /// \return the gamma function of input
    ///
    /// \ingroup arith_func_tgamma
    AFAPI array tgamma (const array &in);

    /// C++ Interface for logarithm of absolute value of gamma function of input
    ///
    /// \param[in] in is input
    /// \return the logarithm of absolute value of gamma function of input
    ///
    /// \ingroup arith_func_tgamma
    AFAPI array lgamma (const array &in);

    /// C++ Interface for checking if values are zero
    ///
    /// \param[in] in is input
    /// \return array containing 1's where input is 0, and 0 otherwise.
    ///
    /// \ingroup arith_func_iszero
    AFAPI array iszero (const array &in);

    /// C++ Interface for checking if values are Infinities
    ///
    /// \param[in] in is input
    /// \return array containing 1's where input is Inf or -Inf, and 0 otherwise.
    ///
    /// \ingroup arith_func_isinf
    AFAPI array isInf  (const array &in);

    /// C++ Interface for checking if values are NaNs
    ///
    /// \param[in] in is input
    /// \return array containing 1's where input is NaN, and 0 otherwise.
    ///
    /// \ingroup arith_func_isnan
    AFAPI array isNaN  (const array &in);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface for adding arrays

       \param[out] out will contain sum of \p lhs and \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_add
    */
    AFAPI af_err af_add   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for subtracting an array from another

       \param[out] out will contain result of \p lhs - \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_sub
    */
    AFAPI af_err af_sub   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for multiplying two arrays

       \param[out] out will contain the product of \p lhs and  \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_mul
    */
    AFAPI af_err af_mul   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for dividing an array by another

       \param[out] out will contain result of \p lhs / \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_div
    */
    AFAPI af_err af_div   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for checking if an array is less than another

       \param[out] out will contain result of \p lhs < \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup logic_func_lt
    */
    AFAPI af_err af_lt    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for checking if an array is greater than another

       \param[out] out will contain result of \p lhs > \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_gt
    */
    AFAPI af_err af_gt    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for checking if an array is less or equal to another

       \param[out] out will contain result of \p lhs <= \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_le
    */
    AFAPI af_err af_le    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for checking if an array is greater or equal to another

       \param[out] out will contain result of \p lhs >= \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_ge
    */
    AFAPI af_err af_ge    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for checking if an array is equal to another

       \param[out] out will contain result of \p lhs == \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_eq
    */
    AFAPI af_err af_eq    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for checking if an array is not equal to another

       \param[out] out will contain result of \p lhs != \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_neq
    */
    AFAPI af_err af_neq   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for performing logical and on two arrays

       \param[out] out will contain result of \p lhs && \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_and
    */
    AFAPI af_err af_and   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for performing logical or on two arrays

       \param[out] out will contain result of \p lhs || \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_or
    */
    AFAPI af_err af_or    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for performing logical not on input

       \param[out] out will contain result of logical not of \p in
       \param[in] in is the input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_not
    */
    AFAPI af_err af_not   (af_array *out, const af_array in);

    /**
       C Interface for performing bitwise and on two arrays

       \param[out] out will contain result of \p lhs & \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_bitand
    */
    AFAPI af_err af_bitand   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for performing bitwise or on two arrays

       \param[out] out will contain result of \p lhs & \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_bitor
    */
    AFAPI af_err af_bitor    (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for performing bitwise xor on two arrays

       \param[out] out will contain result of \p lhs ^ \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_bitxor
    */
    AFAPI af_err af_bitxor   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for left shift on integer arrays

       \param[out] out will contain result of the left shift
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_shiftl
    */
    AFAPI af_err af_bitshiftl(af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for right shift on integer arrays

       \param[out] out will contain result of the right shift
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_shiftr
    */
    AFAPI af_err af_bitshiftr(af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for casting an array from one type to another

       \param[out] out will contain the values in the specified type
       \param[in] in is the input
       \param[in] type is the target data type \ref af_dtype
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cast
    */
    AFAPI af_err af_cast    (af_array *out, const af_array in, const af_dtype type);

    /**
       C Interface for min of two arrays

       \param[out] out will contain minimum of \p lhs and \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_min
    */
    AFAPI af_err af_minof (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for max of two arrays

       \param[out] out will contain maximum of \p lhs and \p rhs
       \param[in] lhs first input
       \param[in] rhs second input
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_max
    */
    AFAPI af_err af_maxof (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for remainder

       \param[out] out will contain the remainder of \p lhs divided by \p rhs
       \param[in] lhs is numerator
       \param[in] rhs is denominator
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_rem
    */
    AFAPI af_err af_rem   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for modulus

       \param[out] out will contain the output of \p lhs modulo \p rhs
       \param[in] lhs is dividend
       \param[in] rhs is divisor
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_mod
    */
    AFAPI af_err af_mod   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for absolute value

       \param[out] out will contain the absolute value of \p in
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_abs
    */
    AFAPI af_err af_abs     (af_array *out, const af_array in);

    /**
       C Interface for finding the phase

       \param[out] out will the phase of \p in
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_arg
    */
    AFAPI af_err af_arg     (af_array *out, const af_array in);

    /**
       C Interface for finding the sign of the input

       \param[out] out will contain the sign of each element of the input arrays
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note output is 1 for negative numbers and 0 for positive numbers

       \ingroup arith_func_round
    */
    AFAPI af_err af_sign   (af_array *out, const af_array in);

    /**
       C Interface for rounding an array of numbers

       \param[out] out will contain values rounded to nearest integer
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note The values are rounded to nearest integer

       \ingroup arith_func_round
    */
    AFAPI af_err af_round   (af_array *out, const af_array in);

    /**
       C Interface for truncating an array of numbers

       \param[out] out will contain values truncated to nearest integer not greater than input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_trunc
    */
    AFAPI af_err af_trunc   (af_array *out, const af_array in);

    /**
       C Interface for flooring an array of numbers

       \param[out] out will contain values rounded to nearest integer less than or equal to in
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_floor
    */
    AFAPI af_err af_floor   (af_array *out, const af_array in);

    /**
       C Interface for ceiling an array of numbers

       \param[out] out will contain values rounded to nearest integer greater than or equal to in
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_ceil
    */
    AFAPI af_err af_ceil    (af_array *out, const af_array in);

    /**
       C Interface for getting length of hypotenuse of two arrays

       \param[out] out will contain the length of the hypotenuse
       \param[in] lhs is the length of first side
       \param[in] rhs is the length of second side
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_floor
    */
    AFAPI af_err af_hypot (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for sin

       \param[out] out will contain sin of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_sin
    */
    AFAPI af_err af_sin     (af_array *out, const af_array in);

    /**
       C Interface for cos

       \param[out] out will contain cos of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cos
    */
    AFAPI af_err af_cos     (af_array *out, const af_array in);

    /**
       C Interface for tan

       \param[out] out will contain tan of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_tan
    */
    AFAPI af_err af_tan     (af_array *out, const af_array in);

    /**
       C Interface for arc sin

       \param[out] out will contain arc sin of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_asin
    */
    AFAPI af_err af_asin    (af_array *out, const af_array in);

    /**
       C Interface for arc cos

       \param[out] out will contain arc cos of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_acos
    */
    AFAPI af_err af_acos    (af_array *out, const af_array in);

    /**
       C Interface for arc tan

       \param[out] out will contain arc tan of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_atan
    */
    AFAPI af_err af_atan    (af_array *out, const af_array in);

    /**
       C Interface for arc tan of two inputs

       \param[out] out will arc tan of the inputs
       \param[in] lhs value of numerator
       \param[in] rhs value of denominator
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_atan
    */
    AFAPI af_err af_atan2 (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for creating complex array from two input arrays

       \param[out] out will contain the complex array generated from inputs
       \param[in] lhs is real array
       \param[in] rhs is imaginary array
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cplx
    */
    AFAPI af_err af_cplx2 (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for creating complex array from real array

       \param[out] out will contain complex array created from real input \p in
       \param[in] in is real array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cplx
    */
    AFAPI af_err af_cplx    (af_array *out, const af_array in);

    /**
       C Interface for getting real part from complex array

       \param[out] out will contain the real part of \p in
       \param[in] in is complex array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_real
    */
    AFAPI af_err af_real    (af_array *out, const af_array in);

    /**
       C Interface for getting imaginary part from complex array

       \param[out] out will contain the imaginary part of \p in
       \param[in] in is complex array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_imag
    */
    AFAPI af_err af_imag    (af_array *out, const af_array in);

    /**
       C Interface for getting the complex conjugate of input array

       \param[out] out will contain the complex conjugate of \p in
       \param[in] in is complex array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_conjg
    */
    AFAPI af_err af_conjg   (af_array *out, const af_array in);

    /**
       C Interface for sinh

       \param[out] out will contain sinh of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_sinh
    */
    AFAPI af_err af_sinh    (af_array *out, const af_array in);

    /**
       C Interface for cosh

       \param[out] out will contain cosh of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cosh
    */
    AFAPI af_err af_cosh    (af_array *out, const af_array in);

    /**
       C Interface for tanh

       \param[out] out will contain tanh of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_tanh
    */
    AFAPI af_err af_tanh    (af_array *out, const af_array in);

    /**
       C Interface for asinh

       \param[out] out will contain inverse sinh of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_asinh
    */
    AFAPI af_err af_asinh   (af_array *out, const af_array in);

    /**
       C Interface for acosh

       \param[out] out will contain inverse cosh of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_acosh
    */
    AFAPI af_err af_acosh   (af_array *out, const af_array in);

    /**
       C Interface for atanh

       \param[out] out will contain inverse tanh of input
       \param[in] in is input array
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_atanh
    */
    AFAPI af_err af_atanh   (af_array *out, const af_array in);

    /**
       C Interface for root

       \param[out] out will contain \p lhs th root of \p rhs
       \param[in] lhs is nth root
       \param[in] rhs is value
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_root
    */
    AFAPI af_err af_root   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);


    /**
       C Interface for power

       \param[out] out will contain \p lhs raised to power \p rhs
       \param[in] lhs is base
       \param[in] rhs is exponent
       \param[in] batch specifies if operations need to be performed in batch mode
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_pow
    */
    AFAPI af_err af_pow   (af_array *out, const af_array lhs, const af_array rhs, const bool batch);

    /**
       C Interface for power of two

       \param[out] out will contain the values of 2 to the power \p in
       \param[in] in is exponent
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_pow2
    */
    AFAPI af_err af_pow2     (af_array *out, const af_array in);

    /**
       C Interface for exponential of an array

       \param[out] out will contain the exponential of \p in
       \param[in] in is exponent
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_exp
    */
    AFAPI af_err af_exp     (af_array *out, const af_array in);

    /**
       C Interface for calculating sigmoid function of an array

       \param[out] out will contain the sigmoid of \p in
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_sigmoid
    */
    AFAPI af_err af_sigmoid (af_array *out, const af_array in);

    /**
       C Interface for exponential of an array minus 1

       \param[out] out will contain the exponential of \p in - 1
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_expm1
    */
    AFAPI af_err af_expm1   (af_array *out, const af_array in);

    /**
       C Interface for error function value

       \param[out] out will contain the error function value of \p in
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_erf
    */
    AFAPI af_err af_erf     (af_array *out, const af_array in);

    /**
       C Interface for complementary error function value

       \param[out] out will contain the complementary error function value of \p in
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_erfc
    */
    AFAPI af_err af_erfc    (af_array *out, const af_array in);

    /**
       C Interface for natural logarithm

       \param[out] out will contain the natural logarithm of \p in
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_log
    */
    AFAPI af_err af_log     (af_array *out, const af_array in);

    /**
       C Interface for logarithm of (in + 1)

       \param[out] out will contain the logarithm of of (in + 1)
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_log1p
    */
    AFAPI af_err af_log1p   (af_array *out, const af_array in);

    /**
       C Interface for logarithm base 10

       \param[out] out will contain the base 10 logarithm of \p in
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_log10
    */
    AFAPI af_err af_log10   (af_array *out, const af_array in);

    /**
       C Interface for logarithm base 2

       \param[out] out will contain the base 2 logarithm of \p in
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup explog_func_log2
    */
    AFAPI af_err af_log2   (af_array *out, const af_array in);

    /**
       C Interface for square root

       \param[out] out will contain the square root of \p in
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_sqrt
    */
    AFAPI af_err af_sqrt    (af_array *out, const af_array in);

    /**
       C Interface for cube root

       \param[out] out will contain the cube root of \p in
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_cbrt
    */
    AFAPI af_err af_cbrt    (af_array *out, const af_array in);

    /**
       C Interface for the factorial

       \param[out] out will contain the result of factorial of \p in
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_factorial
    */
    AFAPI af_err af_factorial   (af_array *out, const af_array in);

    /**
       C Interface for the gamma function

       \param[out] out will contain the result of gamma function of \p in
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_tgamma
    */
    AFAPI af_err af_tgamma   (af_array *out, const af_array in);

    /**
       C Interface for the logarithm of absolute values of gamma function

       \param[out] out will contain the result of logarithm of absolute values of gamma function of \p in
       \param[in] in is input
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup arith_func_lgamma
    */
    AFAPI af_err af_lgamma   (af_array *out, const af_array in);

    /**
        C Interface for checking if values are zero

        \param[out] out will contain 1's where input is 0, and 0 otherwise.
        \param[in] in is input
        \return \ref AF_SUCCESS if the execution completes properly

        \ingroup arith_func_iszero
    */
    AFAPI af_err af_iszero  (af_array *out, const af_array in);

    /**
        C Interface for checking if values are infinities

        \param[out] out will contain 1's where input is Inf or -Inf, and 0 otherwise.
        \param[in] in is input
        \return \ref AF_SUCCESS if the execution completes properly

        \ingroup arith_func_isinf
    */
    AFAPI af_err af_isinf   (af_array *out, const af_array in);

    /**
        C Interface for checking if values are NaNs

        \param[out] out will contain 1's where input is NaN, and 0 otherwise.
        \param[in] in is input
        \return \ref AF_SUCCESS if the execution completes properly

        \ingroup arith_func_nan
    */
    AFAPI af_err af_isnan   (af_array *out, const af_array in);
#ifdef __cplusplus
}
#endif
