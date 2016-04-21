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
        template<typename T, bool isAscending>
        void sort0IndexIterative(Param val, Param idx)
        {
            try {
                compute::command_queue c_queue(getQueue()());

                compute::buffer val_buf((*val.data)());
                compute::buffer idx_buf((*idx.data)());

                for(int w = 0; w < (int)val.info.dims[3]; w++) {
                    int valW = w * (int)val.info.strides[3];
                    int idxW = w * idx.info.strides[3];
                    for(int z = 0; z < (int)val.info.dims[2]; z++) {
                        int valWZ = valW + z * (int)val.info.strides[2];
                        int idxWZ = idxW + z * idx.info.strides[2];
                        for(int y = 0; y < (int)val.info.dims[1]; y++) {

                            int valOffset = valWZ + y * val.info.strides[1];
                            int idxOffset = idxWZ + y * idx.info.strides[1];

                            if(isAscending) {
                                compute::sort_by_key(
                                        compute::make_buffer_iterator< type_t<T> >(val_buf, valOffset),
                                        compute::make_buffer_iterator< type_t<T> >(val_buf, valOffset + val.info.dims[0]),
                                        compute::make_buffer_iterator< type_t<uint> >(idx_buf, idxOffset),
                                        compute::less< type_t<T> >(), c_queue);
                            } else {
                                compute::sort_by_key(
                                        compute::make_buffer_iterator< type_t<T> >(val_buf, valOffset),
                                        compute::make_buffer_iterator< type_t<T> >(val_buf, valOffset + val.info.dims[0]),
                                        compute::make_buffer_iterator< type_t<uint> >(idx_buf, idxOffset),
                                        compute::greater< type_t<T> >(), c_queue);
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

        template<typename T, bool isAscending, int dim>
        void sortIndexBatched(Param pVal, Param pIdx)
        {
            typedef type_t<T> Tk;
            typedef uint Tv;
            typedef std::pair<Tk, Tv> IndexPair;

            try {
                af::dim4 inDims;
                for(int i = 0; i < 4; i++)
                    inDims[i] = pVal.info.dims[i];

                // Sort dimension
                // tileDims * seqDims = inDims
                af::dim4 tileDims(1);
                af::dim4 seqDims = inDims;
                tileDims[dim] = inDims[dim];
                seqDims[dim] = 1;

                // Create/call iota
                // Array<Tv> key = iota<Tv>(seqDims, tileDims);
                dim4 keydims = inDims;
                cl::Buffer* key = bufferAlloc(keydims.elements() * sizeof(Tv));
                Param pSeq;
                pSeq.data = key;
                pSeq.info.offset = 0;
                pSeq.info.dims[0] = keydims[0];
                pSeq.info.strides[0] = 1;
                for(int i = 1; i < 4; i++) {
                    pSeq.info.dims[i] = keydims[i];
                    pSeq.info.strides[i] = pSeq.info.strides[i - 1] * pSeq.info.dims[i - 1];
                }
                kernel::iota<Tv>(pSeq, seqDims, tileDims);

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
                compute::buffer_iterator<Tv> seq0 = compute::make_buffer_iterator<Tv>(pSeq_buf, 0);
                compute::buffer_iterator<Tv> seqN = compute::make_buffer_iterator<Tv>(pSeq_buf, elements);

                // Copy val, idx into X pair
                cl::Buffer* X = bufferAlloc(elements * sizeof(IndexPair));
                // Use T here, not Tk
                kernel::makePair<T, Tv>(X, pVal.data, pIdx.data, elements);
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
                BOOST_COMPUTE_FUNCTION(bool, Compare_Tv, (const Tv lhs, const Tv rhs),
                    {
                        return lhs < rhs;
                    }
                );
                compute::sort_by_key(seq0, seqN, X0, Compare_Tv, c_queue);
                getQueue().finish();

                kernel::splitPair<T, Tv>(pVal.data, pIdx.data, X, elements);

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

        template<typename T, bool isAscending>
        void sort0Index(Param val, Param idx)
        {
            int higherDims =  val.info.dims[1] * val.info.dims[2] * val.info.dims[3];
            // TODO Make a better heurisitic
            if(higherDims > 5)
              sortIndexBatched<T, isAscending, 0>(val, idx);
            else
                kernel::sort0IndexIterative<T, isAscending>(val, idx);
        }
    }
}

#pragma GCC diagnostic pop
