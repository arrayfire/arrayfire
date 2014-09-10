#pragma once

#include <af/defines.h>
#include <kernel_headers/approx1.hpp>
#include <kernel_headers/approx2.hpp>
#include <cl.hpp>
#include <platform.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <mutex>
#include <dispatch.hpp>

typedef struct
{
    dim_type dim[4];
} dims_t;

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
        static const dim_type TX = 16;
        static const dim_type TY = 16;

        static const dim_type THREADS = 256;

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename Ty, typename Tp, af_interp_type method>
        void approx1(      Buffer out, const dim_type *odims, const dim_type oElems,
                     const Buffer in,  const dim_type *idims, const dim_type iElems,
                     const Buffer pos, const dim_type *pdims, const dim_type *ostrides,
                     const dim_type *istrides, const dim_type *pstrides, const float offGrid,
                     const dim_type iOffset, const dim_type pOffset)
        {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static Program            apxProgs[DeviceManager::MAX_DEVICES];
            static Kernel           apxKernels[DeviceManager::MAX_DEVICES];

            int device = getActiveDeviceId();

            std::call_once( compileFlags[device], [device] () {
                        Program::Sources setSrc;
                        setSrc.emplace_back(approx1_cl, approx1_cl_len);

                        apxProgs[device] = Program(getContext(), setSrc);

                        std::ostringstream options;
                        options << " -D Ty="        << dtype_traits<Ty>::getName()
                                << " -D Tp="        << dtype_traits<Tp>::getName()
                                << " -D dim_type="  << dtype_traits<dim_type>::getName();

                        if((af_dtype) dtype_traits<Ty>::af_type == c32 ||
                           (af_dtype) dtype_traits<Ty>::af_type == c64) {
                            options << " -D CPLX=1";
                        } else {
                            options << " -D CPLX=0";
                        }

                        switch(method) {
                            case AF_INTERP_NEAREST:
                                options << " -D INTERP=NEAREST";
                                break;
                            case AF_INTERP_LINEAR:
                                options << " -D INTERP=LINEAR";
                                break;
                            default:
                                break;
                        }
                        apxProgs[device].build(options.str().c_str());

                        apxKernels[device] = Kernel(apxProgs[device], "approx1_kernel");
                    });


            auto approx1Op = make_kernel<Buffer, const dims_t, const dim_type,
                                  const Buffer, const dims_t, const dim_type,
                                  const Buffer, const dims_t,
                                  const dims_t, const dims_t, const dims_t,
                                  const float, const dim_type,
                                  const dim_type, const dim_type>
                                  (apxKernels[device]);

            NDRange local(THREADS, 1, 1);
            dim_type blocksPerMat = divup(odims[0], local[0]);
            NDRange global(blocksPerMat * local[0] * odims[1],
                           odims[2]     * odims[3] * local[0],
                           1);

            dims_t _odims = {{odims[0], odims[1], odims[2], odims[3]}};
            dims_t _idims = {{idims[0], idims[1], idims[2], idims[3]}};
            dims_t _pdims = {{pdims[0], pdims[1], pdims[2], pdims[3]}};
            dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
            dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};
            dims_t _pstrides = {{pstrides[0], pstrides[1], pstrides[2], pstrides[3]}};

            approx1Op(EnqueueArgs(getQueue(), global, local),
                          out, _odims, oElems, in, _idims, iElems, pos, _pdims,
                          _ostrides, _istrides, _pstrides, offGrid, blocksPerMat, iOffset, pOffset);
        }

        template <typename Ty, typename Tp, af_interp_type method>
        void approx2(      Buffer out, const dim_type *odims, const dim_type oElems,
                     const Buffer in,  const dim_type *idims, const dim_type iElems,
                     const Buffer pos, const dim_type *pdims,
                     const Buffer qos, const dim_type *qdims,
                     const dim_type *ostrides, const dim_type *istrides,
                     const dim_type *pstrides, const dim_type *qstrides,
                     const float offGrid, const dim_type iOffset, const dim_type pOffset,
                     const dim_type qOffset)
        {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static Program            apxProgs[DeviceManager::MAX_DEVICES];
            static Kernel           apxKernels[DeviceManager::MAX_DEVICES];

            int device = getActiveDeviceId();

            std::call_once( compileFlags[device], [device] () {
                        Program::Sources setSrc;
                        setSrc.emplace_back(approx2_cl, approx2_cl_len);

                        apxProgs[device] = Program(getContext(), setSrc);

                        std::ostringstream options;
                        options << " -D Ty="        << dtype_traits<Ty>::getName()
                                << " -D Tp="        << dtype_traits<Tp>::getName()
                                << " -D dim_type="  << dtype_traits<dim_type>::getName();

                        if((af_dtype) dtype_traits<Ty>::af_type == c32 ||
                           (af_dtype) dtype_traits<Ty>::af_type == c64) {
                            options << " -D CPLX=1";
                        } else {
                            options << " -D CPLX=0";
                        }

                        switch(method) {
                            case AF_INTERP_NEAREST:
                                options << " -D INTERP=NEAREST";
                                break;
                            case AF_INTERP_LINEAR:
                                options << " -D INTERP=LINEAR";
                                break;
                            default:
                                break;
                        }
                        apxProgs[device].build(options.str().c_str());

                        apxKernels[device] = Kernel(apxProgs[device], "approx2_kernel");
                    });

            auto approx2Op = make_kernel<Buffer, const dims_t, const dim_type,
                                  const Buffer, const dims_t, const dim_type,
                                  const Buffer, const dims_t, const Buffer, const dims_t,
                                  const dims_t, const dims_t, const dims_t, const dims_t,
                                  const float, const dim_type, const dim_type,
                                  const dim_type, const dim_type, const dim_type>
                                  (apxKernels[device]);

            NDRange local(TX, TY, 1);
            dim_type blocksPerMatX = divup(odims[0], local[0]);
            dim_type blocksPerMatY = divup(odims[1], local[1]);
            NDRange global(blocksPerMatX * local[0] * odims[2],
                           blocksPerMatY * local[1] * odims[3],
                           1);


            dims_t _odims = {{odims[0], odims[1], odims[2], odims[3]}};
            dims_t _idims = {{idims[0], idims[1], idims[2], idims[3]}};
            dims_t _pdims = {{pdims[0], pdims[1], pdims[2], pdims[3]}};
            dims_t _qdims = {{qdims[0], qdims[1], qdims[2], qdims[3]}};
            dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
            dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};
            dims_t _pstrides = {{pstrides[0], pstrides[1], pstrides[2], pstrides[3]}};
            dims_t _qstrides = {{qstrides[0], qstrides[1], qstrides[2], qstrides[3]}};

            approx2Op(EnqueueArgs(getQueue(), global, local),
                          out, _odims, oElems, in, _idims, iElems, pos, _pdims, qos, _qdims,
                          _ostrides, _istrides, _pstrides, _qstrides, offGrid,
                          blocksPerMatX, blocksPerMatY, iOffset, pOffset, qOffset);
        }
    }
}
