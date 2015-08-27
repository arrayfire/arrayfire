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
        \param[in] exp is an expression, generally the name of the array
        \param[in] arr is the input array

        \ingroup print_func_print
    */
    AFAPI void print(const char *exp, const array &arr);

#if AF_API_VERSION >= 31
    /**
        \param[in] exp is an expression, generally the name of the array
        \param[in] arr is the input array
        \param[in] precision is the precision length for display

        \ingroup print_func_print
    */
    AFAPI void print(const char *exp, const array &arr, const int precision);
#endif

#if AF_API_VERSION >= 31
    /**
        \param[in] key is an expression used as tag/key for the array during \ref readArray
        \param[in] arr is the array to be written
        \param[in] filename is the path to the location on disk
        \param[in] append is used to append to an existing file when true and create or
        overwrite an existing file when false

        \returns index of the saved array in the file

        \ingroup stream_func_save
    */
    AFAPI int saveArray(const char *key, const array &arr, const char *filename, const bool append = false);
#endif

#if AF_API_VERSION >= 31
    /**
        \param[in] filename is the path to the location on disk
        \param[in] index is the 0-based sequential location of the array to be read

        \returns array read from the index location

        \note This function will throw an exception if the index is out of bounds

        \ingroup stream_func_read
    */
    AFAPI array readArray(const char *filename, const unsigned index);
#endif

#if AF_API_VERSION >= 31
    /**
        \param[in] filename is the path to the location on disk
        \param[in] key is the tag/name of the array to be read. The key needs to have an exact match.

        \returns array read by key

        \note This function will throw an exception if the key is not found.

        \ingroup stream_func_read
    */
    AFAPI array readArray(const char *filename, const char *key);
#endif

#if AF_API_VERSION >= 31
    /**
        When reading by key, it may be a good idea to run this function first to check for the key
        and then call the readArray using the index. This will avoid exceptions in case of key not found.

        \param[in] filename is the path to the location on disk
        \param[in] key is the tag/name of the array to be read. The key needs to have an exact match.

        \returns index of the array in the file if the key is found. -1 if key is not found.

        \ingroup stream_func_read
    */
    AFAPI int readArrayCheck(const char *filename, const char *key);
#endif

#if AF_API_VERSION >= 31
    /**
        \param[out] output is the pointer to the c-string that will hold the data. The memory for
        output is allocated by the function. The user is responsible for deleting the memory.
        \param[in] exp is an expression, generally the name of the array
        \param[in] arr is the input array
        \param[in] precision is the precision length for display
        \param[in] transpose determines whether or not to transpose the array before storing it in
        the string

        \ingroup print_func_tostring
    */
    AFAPI void toString(char **output, const char *exp, const array &arr,
                        const int precision = 4, const bool transpose = true);
#endif

    // Purpose of Addition: "How to add Function" documentation
    AFAPI array exampleFunction(const array& in, const af_someenum_t param);
}

#if AF_API_VERSION >= 31

#define AF_PRINT1(exp)            af::print(#exp, exp);
#define AF_PRINT2(exp, precision) af::print(#exp, exp, precision);

#define GET_PRINT_MACRO(_1, _2, NAME, ...) NAME

#define af_print(...) GET_PRINT_MACRO(__VA_ARGS__, AF_PRINT2, AF_PRINT1)(__VA_ARGS__)

#else

#define af_print(exp) af::print(#exp, exp);

#endif

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

#if AF_API_VERSION >= 31
    /**
        \param[in] exp is the expression or name of the array
        \param[in] arr is the input array
        \param[in] precision precision for the display

        \returns error codes

        \ingroup print_func_print
    */
    AFAPI af_err af_print_array_gen(const char *exp, const af_array arr, const int precision);
#endif

#if AF_API_VERSION >= 31
    /**
        \param[out] index is the index location of the array in the file
        \param[in] key is an expression used as tag/key for the array during \ref readArray()
        \param[in] arr is the array to be written
        \param[in] filename is the path to the location on disk
        \param[in] append is used to append to an existing file when true and create or
        overwrite an existing file when false

        \ingroup stream_func_save
    */
    AFAPI af_err af_save_array(int *index, const char* key, const af_array arr, const char *filename, const bool append);
#endif

#if AF_API_VERSION >= 31
    /**
        \param[out] out is the array read from index
        \param[in] filename is the path to the location on disk
        \param[in] index is the 0-based sequential location of the array to be read

        \note This function will throw an exception if the key is not found.

        \ingroup stream_func_read
    */
    AFAPI af_err af_read_array_index(af_array *out, const char *filename, const unsigned index);
#endif

#if AF_API_VERSION >= 31
    /**
        \param[out] out is the array read from key
        \param[in] filename is the path to the location on disk
        \param[in] key is the tag/name of the array to be read. The key needs to have an exact match.

        \note This function will throw an exception if the key is not found.

        \ingroup stream_func_read
    */
    AFAPI af_err af_read_array_key(af_array *out, const char *filename, const char* key);
#endif

#if AF_API_VERSION >= 31
    /**
        When reading by key, it may be a good idea to run this function first to check for the key
        and then call the readArray using the index. This will avoid exceptions in case of key not found.

        \param[out] index of the array in the file if the key is found. -1 if key is not found.
        \param[in] filename is the path to the location on disk
        \param[in] key is the tag/name of the array to be read. The key needs to have an exact match.

        \ingroup stream_func_read
    */
    AFAPI af_err af_read_array_key_check(int *index, const char *filename, const char* key);
#endif

#if AF_API_VERSION >= 31
    /**
        \param[out] output is the pointer to the c-string that will hold the data. The memory for
        output is allocated by the function. The user is responsible for deleting the memory.
        \param[in] exp is an expression, generally the name of the array
        \param[in] arr is the input array
        \param[in] precision is the precision length for display
        \param[in] transpose determines whether or not to transpose the array before storing it in
        the string

        \ingroup print_func_tostring
    */
    AFAPI af_err af_array_to_string(char **output, const char *exp, const af_array arr,
                                    const int precision, const bool transpose);
#endif

    // Purpose of Addition: "How to add Function" documentation
    AFAPI af_err af_example_function(af_array* out, const af_array in, const af_someenum_t param);

    ///
    ///Get the version information of the library
    ///
    AFAPI af_err af_get_version(int *major, int *minor, int *patch);

#ifdef __cplusplus
}
#endif
