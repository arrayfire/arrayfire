/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

#ifdef __cplusplus
namespace af
{
    class array;

    /**
        \defgroup print_func_print print

        \brief Print the array to screen

        \ingroup arrayfire_func
    */
    /**
        \param[in] exp is an expression, generally the name of the array
        \param[in] arr is the input array

        \ingroup print_func_print
    */
    AFAPI void print(const char *exp, const array &arr);

    /**
        \param[in] exp is an expression, generally the name of the array
        \param[in] arr is the input array
        \param[in] precision is the precision length for display

        \ingroup print_func_print
    */
    AFAPI void print(const char *exp, const array &arr, const int precision);

    // Purpose of Addition: "How to add Function" documentation
    AFAPI array exampleFunction(const array& in, const af_someenum_t param);
}

#define AF_PRINT1(exp)            af::print(#exp, exp);
#define AF_PRINT2(exp, precision) af::print(#exp, exp, precision);

#define GET_PRINT_MACRO(_1, _2, NAME, ...) NAME

/**
    Macro to print an array along with the variable name

    \param[in] exp the array to be printed
    \param[in] precision (optional) is the number of decimal places to be printed

    \code
    af::array myarray = randu(3, 3);
    int myprecision = 2;

    af_print(myarray);                  // Defaults precision to 4 decimal places
    // A [3 3 1 1]
    //     0.0010   311.3614     1.6264
    //    60.3298   497.9737   359.5948
    //   165.4467   113.7310     5.2294

    af_print(myarray, myprecision);     // Uses myprecision decimal places
    // A [3 3 1 1]
    //     0.00   311.36     1.63
    //    60.33   497.97   359.59
    //   165.45   113.73     5.23

    af_print(myarray, 6);               // Uses 6 decimal places
    // A [3 3 1 1]
    //     0.001029   311.361402     1.626432
    //    60.329828   497.973728   359.594787
    //   165.446732   113.730984     5.229350
    \endcode

    \ingroup print_func_print
*/
#define af_print(...) GET_PRINT_MACRO(__VA_ARGS__, AF_PRINT2, AF_PRINT1)(__VA_ARGS__)

#endif //__cplusplus

#ifdef __cplusplus
extern "C" {
#endif

    /**
        \param[in] arr is the input array

        \returns error codes

        \ingroup print_func_print
    */
    AFAPI af_err af_print_array(af_array arr);

    /**
        \param[in] arr is the input array
        \param[in] precision precision for the display

        \returns error codes

        \ingroup print_func_print
    */
    AFAPI af_err af_print_array_p(af_array arr, const int precision);

    // Purpose of Addition: "How to add Function" documentation
    AFAPI af_err af_example_function(af_array* out, const af_array in, const af_someenum_t param);

    ///
    ///Get the version information of the library
    ///
    AFAPI af_err af_get_version(int *major, int *minor, int *patch);

#ifdef __cplusplus
}
#endif
