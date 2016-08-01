#pragma once
#include <Param.hpp>
namespace cuda
{
    namespace kernel
    {
        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename Tk, typename Tv>
        void thrustSortByKey(Tk *keyPtr, Tv *valPtr, int elements, bool isAscending);
    }
}
