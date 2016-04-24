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
        template <typename Tk, typename Tv>
        using IndexPair = std::pair<Tk, Tv>;

        template <typename Tk, typename Tv, bool isAscending>
        struct IPCompare
        {
            bool operator()(const IndexPair<Tk, Tv> &lhs, const IndexPair<Tk, Tv> &rhs)
            {
                // Check stable sort condition
                if(isAscending) return (lhs.first < rhs.first);
                else return (lhs.first > rhs.first);
            }
        };

        template <typename Tk, typename Tv>
        using KeyIndexPair = std::pair<IndexPair<Tk, Tv>, uint>;

        template <typename Tk, typename Tv, bool isAscending>
        struct KIPCompareV
        {
            bool operator()(const KeyIndexPair<Tk, Tv> &lhs, const KeyIndexPair<Tk, Tv> &rhs)
            {
                // Check stable sort condition
                if(isAscending) return (lhs.first.first < rhs.first.first);
                else return (lhs.first.first > rhs.first.first);
            }
        };

        template <typename Tk, typename Tv, bool isAscending>
        struct KIPCompareK
        {
            bool operator()(const KeyIndexPair<Tk, Tv> &lhs, const KeyIndexPair<Tk, Tv> &rhs)
            {
                if(isAscending) return (lhs.second < rhs.second);
                else return (lhs.second > rhs.second);
            }
        };
    }
}
