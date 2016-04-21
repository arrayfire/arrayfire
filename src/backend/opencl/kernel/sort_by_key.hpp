/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <kernel/sort_helper.hpp>
#include <kernel/iota.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/sort_by_key.hpp>
#include <boost/compute/functional/operator.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/get.hpp>
#include <boost/compute/functional/field.hpp>
#include <boost/compute/types/pair.hpp>

namespace compute = boost::compute;

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    namespace kernel
    {
        template<typename Tk, typename Tv, bool isAscending>
        void sort0ByKeyIterative(Param pKey, Param pVal)
        {
            try {
                compute::command_queue c_queue(getQueue()());

                compute::buffer pKey_buf((*pKey.data)());
                compute::buffer pVal_buf((*pVal.data)());

                for(int w = 0; w < pKey.info.dims[3]; w++) {
                    int pKeyW = w * pKey.info.strides[3];
                    int pValW = w * pVal.info.strides[3];
                    for(int z = 0; z < pKey.info.dims[2]; z++) {
                        int pKeyWZ = pKeyW + z * pKey.info.strides[2];
                        int pValWZ = pValW + z * pVal.info.strides[2];
                        for(int y = 0; y < pKey.info.dims[1]; y++) {

                            int pKeyOffset = pKeyWZ + y * pKey.info.strides[1];
                            int pValOffset = pValWZ + y * pVal.info.strides[1];

                            compute::buffer_iterator< type_t<Tk> > start= compute::make_buffer_iterator< type_t<Tk> >(pKey_buf, pKeyOffset);
                            compute::buffer_iterator< type_t<Tk> > end  = compute::make_buffer_iterator< type_t<Tk> >(pKey_buf, pKeyOffset + pKey.info.dims[0]);
                            compute::buffer_iterator< type_t<Tv> > vals = compute::make_buffer_iterator< type_t<Tv> >(pVal_buf, pValOffset);
                            if(isAscending) {
                                compute::sort_by_key(start, end, vals, c_queue);
                            } else {
                                compute::sort_by_key(start, end, vals,
                                                     compute::greater< type_t<Tk> >(), c_queue);
                            }
                        }
                    }
                }

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }

        template<typename Tk_, typename Tv_, bool isAscending, int dim>
        void sortByKeyBatched(Param pKey, Param pVal)
        {
            typedef type_t<Tk_> Tk;
            typedef type_t<Tv_> Tv;
            typedef std::pair<Tk, Tv> IndexPair;

            try {
                af::dim4 inDims;
                for(int i = 0; i < 4; i++)
                    inDims[i] = pKey.info.dims[i];

                // Sort dimension
                // tileDims * seqDims = inDims
                af::dim4 tileDims(1);
                af::dim4 seqDims = inDims;
                tileDims[dim] = inDims[dim];
                seqDims[dim] = 1;

                // Create/call iota
                // Array<Tv> key = iota<Tv>(seqDims, tileDims);
                dim4 keydims = inDims;
                cl::Buffer* key = bufferAlloc(keydims.elements() * sizeof(unsigned));
                Param pSeq;
                pSeq.data = key;
                pSeq.info.offset = 0;
                pSeq.info.dims[0] = keydims[0];
                pSeq.info.strides[0] = 1;
                for(int i = 1; i < 4; i++) {
                    pSeq.info.dims[i] = keydims[i];
                    pSeq.info.strides[i] = pSeq.info.strides[i - 1] * pSeq.info.dims[i - 1];
                }
                kernel::iota<unsigned>(pSeq, seqDims, tileDims);

                int elements = inDims.elements();

                // Flat - Not required since inplace and both are continuous
                //val.modDims(inDims.elements());
                //key.modDims(inDims.elements());

                // Sort indices
                // sort_by_key<T, Tv, isAscending>(*resVal, *resKey, val, key, 0);
                //kernel::sort0_by_key<T, Tv, isAscending>(pVal, pKey);
                compute::command_queue c_queue(getQueue()());
                compute::context c_context(getContext()());

                // Create buffer iterators for seq
                compute::buffer pSeq_buf((*pSeq.data)());
                compute::buffer_iterator<unsigned> seq0 = compute::make_buffer_iterator<unsigned>(pSeq_buf, 0);
                compute::buffer_iterator<unsigned> seqN = compute::make_buffer_iterator<unsigned>(pSeq_buf, elements);

                // Copy key, val into X pair
                cl::Buffer* X = bufferAlloc(elements * sizeof(IndexPair));
                // Use Tk_ and Tv_ here, not Tk and Tv
                kernel::makePair<Tk_, Tv_>(X, pKey.data, pVal.data, elements);
                compute::buffer X_buf((*X)());
                compute::buffer_iterator<IndexPair> X0 = compute::make_buffer_iterator<IndexPair>(X_buf, 0);
                compute::buffer_iterator<IndexPair> XN = compute::make_buffer_iterator<IndexPair>(X_buf, elements);

                // FIRST SORT CALL
                compute::function<bool(IndexPair, IndexPair)> IPCompare =
                    makeCompareFunction<Tk, Tv, isAscending>();

                compute::sort_by_key(X0, XN, seq0, IPCompare, c_queue);
                getQueue().finish();

                // Needs to be ascending (true) in order to maintain the indices properly
                //kernel::sort0_by_key<Tv, T, true>(pKey, pVal);
                //
                // Because we use a pair as values, we need to use a custom comparator
                BOOST_COMPUTE_FUNCTION(bool, Compare_Seq, (const unsigned lhs, const unsigned rhs),
                    {
                        return lhs < rhs;
                    }
                );
                compute::sort_by_key(seq0, seqN, X0, Compare_Seq, c_queue);
                getQueue().finish();

                kernel::splitPair<Tk_, Tv_>(pKey.data, pVal.data, X, elements);

                //// No need of doing moddims here because the original Array<T>
                //// dimensions have not been changed
                ////val.modDims(inDims);

                CL_DEBUG_FINISH(getQueue());
                bufferFree(key);
                bufferFree(X);
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }

        template<typename Tk, typename Tv, bool isAscending>
        void sort0ByKey(Param pKey, Param pVal)
        {
            int higherDims =  pKey.info.dims[1] * pKey.info.dims[2] * pKey.info.dims[3];
            // TODO Make a better heurisitic
            if(higherDims > 5)
                kernel::sortByKeyBatched<Tk, Tv, isAscending, 0>(pKey, pVal);
            else
                kernel::sort0ByKeyIterative<Tk, Tv, isAscending>(pKey, pVal);
        }
    }
}

#pragma GCC diagnostic pop
