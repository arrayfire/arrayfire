/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_cpu.hpp>

namespace cpu
{
    namespace kernel
    {
        static const int copyPairIter = 4;

        template <typename T>
        using IndexPair = std::pair<T, uint>;

        template <typename T, bool isAscending>
        struct IPCompare
        {
            bool operator()(const IndexPair<T> &lhs, const IndexPair<T> &rhs)
            {
                // Check stable sort condition
                if(isAscending) return (lhs.first < rhs.first);
                else return (lhs.first > rhs.first);
            }
        };

        template <typename T>
        using KeyIndexPair = std::pair<IndexPair<T>, uint>;

        template <typename T, bool isAscending>
        struct KIPCompareV
        {
            bool operator()(const KeyIndexPair<T> &lhs, const KeyIndexPair<T> &rhs)
            {
                // Check stable sort condition
                if(isAscending) return (lhs.first.first < rhs.first.first);
                else return (lhs.first.first > rhs.first.first);
            }
        };

        template <typename T, bool isAscending>
        struct KIPCompareK
        {
            bool operator()(const KeyIndexPair<T> &lhs, const KeyIndexPair<T> &rhs)
            {
                if(isAscending) return (lhs.second < rhs.second);
                else return (lhs.second > rhs.second);
            }
        };
    }
}
