/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <traits.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

namespace opencl
{
    namespace kernel
    {
        template<typename Tk, typename Tv>
        void sort0ByKeyIterative(Param pKey, Param pVal, bool isAscending);

        template<typename Tk_, typename Tv_>
        void sortByKeyBatched(Param pKey, Param pVal, const int dim, bool isAscending);

        template<typename Tk, typename Tv>
        void sort0ByKey(Param pKey, Param pVal, bool isAscending);
    }
}
