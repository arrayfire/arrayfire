/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>          // Needed if you use dim4 class

#include <af/util.h>            // Include header where function is delcared

#include <af/defines.h>         // Include this header to access any enums,
                                // #defines or constants declared

#include <err_common.hpp>       // Header with error checking functions & macros

#include <backend.hpp>          // This header make sures appropriate backend
                                // related namespace is being used

#include <Array.hpp>            // Header in which backend specific Array class
                                // is defined

#include <handle.hpp>           // Header that helps you retrieve backend specific
                                // Arrays based on the af_array
                                // (typedef in defines.h) handle.

#include <exampleFunction.hpp>  // This is the backend specific header
                                // where your new function declaration
                                // is written

using namespace detail;         // detail is an alias to appropriate backend
                                // defined in backend.hpp. You don't need to
                                // change this

template<typename T>
af_array example(const af_array& in, const af_someenum_t& param)
{
    // getArray<T> function is defined in handle.hpp
    // and it returns backend specific Array, namely one of the following
    //      * cpu::Array<T>
    //      * cuda::Array<T>
    //      * opencl::Array<T>
    // getHandle<T> function is defined in handle.hpp takes one of the
    // above backend specific detail::Array<T> and returns the
    // universal array handle af_array
    return getHandle<T>( exampleFunction(getArray<T>(in), param) );
}

af_err af_example_function(af_array* out,
                            const af_array in,
                            const af_someenum_t param)
{
    try {
        af_array output = 0;
        ArrayInfo info = getInfo(in);       // ArrayInfo is the base class which
                                            // each backend specific Array inherits
                                            // This class stores the basic array meta-data
                                            // such as type of data, dimensions,
                                            // offsets and strides. This class is declared
                                            // in src/backend/ArrayInfo.hpp

        af::dim4 dims = info.dims();


        ARG_ASSERT(2, (dims.ndims()>=0 && dims.ndims()<=3));
                                            // defined in err_common.hpp
                                            // there are other useful Macros
                                            // for different purposes, feel free
                                            // to look at the header

        af_dtype type = info.getType();

        switch(type) {                      // Based on the data type, call backend specific
                                            // implementation
            case f64: output = example<double >(in, param); break;
            case f32: output = example<float  >(in, param); break;
            case s32: output = example<int    >(in, param); break;
            case u32: output = example<uint   >(in, param); break;
            case  u8: output = example<uchar  >(in, param); break;
            case  b8: output = example<char   >(in, param); break;
            case c32: output = example<cfloat >(in, param); break;
            case c64: output = example<cdouble>(in, param); break;
            default : TYPE_ERROR(1, type);  // Another helpful macro from err_common.hpp
                                            // that helps throw type based error messages
        }

        std::swap(*out, output);            // if the function has returned successfully,
                                            // swap the temporary 'output' variable with
                                            // '*out'
    }
    CATCHALL;                               // All throws/exceptions from any internal
                                            // implementations are caught by this CATCHALL
                                            // macro and handled appropriately.

    return AF_SUCCESS;                      // In case of successfull completion, return AF_SUCCESS
                                            // There are set of error codes defined in defines.h
                                            // which you are used by CATCHALL to return approriate code
}
