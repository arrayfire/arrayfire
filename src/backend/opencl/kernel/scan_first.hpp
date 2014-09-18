#pragma once
#include <string>
#include <mutex>
#include <kernel_headers/scan_first.hpp>
#include <kernel_headers/ops.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <type_util.hpp>
#include "names.hpp"
#include "config.hpp"

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

    template<typename Ti, typename To, af_op_t op, bool isFinalPass, uint threads_x>
    static Kernel get_scan_first_kernels(int kerIdx)
    {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static Program           scanProgs[DeviceManager::MAX_DEVICES];
        static Kernel            scanKerns[DeviceManager::MAX_DEVICES];
        static Kernel           bcastKerns[DeviceManager::MAX_DEVICES];

        int device= getActiveDeviceId();

        std::call_once(compileFlags[device], [device] () {

                const uint threads_y = THREADS_PER_GROUP / threads_x;
                const uint SHARED_MEM_SIZE = (threads_x + 1) * (2 * threads_y);

                Binary<To, op> scan;
                ToNum<To> toNum;

                std::ostringstream options;
                options << " -D To=" << dtype_traits<To>::getName()
                        << " -D Ti=" << dtype_traits<Ti>::getName()
                        << " -D T=To"
                        << " -D DIMX=" << threads_x
                        << " -D DIMY=" << threads_y
                        << " -D SHARED_MEM_SIZE=" << SHARED_MEM_SIZE
                        << " -D init=" << toNum(scan.init())
                        << " -D " << binOpName<op>()
                        << " -D CPLX=" << af::iscplx<Ti>()
                        << " -D isFinalPass=" << (int)(isFinalPass);

                const char *ker_strs[] = {ops_cl, scan_first_cl};
                const int   ker_lens[] = {ops_cl_len, scan_first_cl_len};
                buildProgram(scanProgs[device], 2, ker_strs, ker_lens, options.str());

                scanKerns[device] = Kernel(scanProgs[device],  "scan_first_kernel");
                bcastKerns[device] = Kernel(scanProgs[device],  "bcast_first_kernel");

            });

        return (kerIdx == 0) ? scanKerns[device] : bcastKerns[device];
    }

    template<typename Ti, typename To, af_op_t op, bool isFinalPass, uint threads_x>
    void scan_first_launcher(Param out,
                             Param tmp,
                             const Param in,
                             const uint groups_x,
                             const uint groups_y)
    {
        Kernel ker = get_scan_first_kernels<Ti, To, op, isFinalPass, threads_x>(0);

        NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
        NDRange global(groups_x * out.info.dims[2] * local[0],
                       groups_y * out.info.dims[3] * local[1]);

        uint lim = divup(out.info.dims[0], (threads_x * groups_x));

        auto scanOp = make_kernel<Buffer, KParam,
                                  Buffer, KParam,
                                  Buffer, KParam,
                                  uint, uint, uint>(ker);

        scanOp(EnqueueArgs(getQueue(), global, local),
               out.data, out.info, tmp.data, tmp.info, in.data, in.info,
               groups_x, groups_y, lim);

        CL_DEBUG_FINISH(getQueue());
    }

    template<typename Ti, typename To, af_op_t op, bool isFinalPass, uint threads_x>
    void bcast_first_launcher(Param out,
                              Param tmp,
                              const uint groups_x,
                              const uint groups_y)
    {

        Kernel ker = get_scan_first_kernels<Ti, To, op, isFinalPass, threads_x>(1);

        NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
        NDRange global(groups_x * out.info.dims[2] * local[0],
                       groups_y * out.info.dims[3] * local[1]);

        uint lim = divup(out.info.dims[0], (threads_x * groups_x));

        auto bcastOp = make_kernel<Buffer, KParam,
                                   Buffer, KParam,
                                   uint, uint, uint>(ker);

        bcastOp(EnqueueArgs(getQueue(), global, local),
                out.data, out.info, tmp.data, tmp.info,
                groups_x, groups_y, lim);

        CL_DEBUG_FINISH(getQueue());
    }


    template<typename Ti, typename To, af_op_t op, bool isFinalPass>
    void scan_first_fn(Param out,
                       Param tmp,
                       const Param in,
                       const uint groups_x,
                       const uint groups_y,
                       const uint threads_x)
    {

        switch (threads_x) {
        case 32:
            (scan_first_launcher<Ti, To, op, isFinalPass,  32>)(
                out, tmp, in, groups_x, groups_y); break;
        case 64:
            (scan_first_launcher<Ti, To, op, isFinalPass,  64>)(
                out, tmp, in, groups_x, groups_y); break;
        case 128:
            (scan_first_launcher<Ti, To, op, isFinalPass, 128>)(
                out, tmp, in, groups_x, groups_y); break;
        case 256:
            (scan_first_launcher<Ti, To, op, isFinalPass, 256>)(
                out, tmp, in, groups_x, groups_y); break;
        }

    }

    template<typename Ti, typename To, af_op_t op, bool isFinalPass>
    void bcast_first_fn(Param out,
                        Param tmp,
                        const uint groups_x,
                        const uint groups_y,
                        const uint threads_x)
    {

        switch (threads_x) {
        case 32:
            (bcast_first_launcher<Ti, To, op, isFinalPass,  32>)(
                out, tmp, groups_x, groups_y); break;
        case 64:
            (bcast_first_launcher<Ti, To, op, isFinalPass,  64>)(
                out, tmp, groups_x, groups_y); break;
        case 128:
            (bcast_first_launcher<Ti, To, op, isFinalPass, 128>)(
                out, tmp, groups_x, groups_y); break;
        case 256:
            (bcast_first_launcher<Ti, To, op, isFinalPass, 256>)(
                out, tmp, groups_x, groups_y); break;
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void scan_first(Param out, const Param in)
    {
        uint threads_x = nextpow2(std::max(32u, (uint)out.info.dims[0]));
        threads_x = std::min(threads_x, THREADS_PER_GROUP);
        uint threads_y = THREADS_PER_GROUP / threads_x;

        uint groups_x = divup(out.info.dims[0], threads_x * REPEAT);
        uint groups_y = divup(out.info.dims[1], threads_y);

        if (groups_x == 1) {
            scan_first_fn<Ti, To, op, true>(out, out, in,
                                            groups_x, groups_y,
                                            threads_x);

        } else {

            Param tmp = out;
            // FIXME: Do I need to free this ?
            tmp.data = cl::Buffer(getContext(), CL_MEM_READ_WRITE,
                                  groups_x *
                                  out.info.dims[1] *
                                  out.info.dims[2] *
                                  out.info.dims[3] *
                                  sizeof(To));

            tmp.info.dims[0] = groups_x;
            for (int k = 1; k < 4; k++) tmp.info.strides[k] *= groups_x;

            scan_first_fn<Ti, To, op, false>(out, tmp, in,
                                             groups_x, groups_y,
                                             threads_x);

            if (op == af_notzero_t) {
                scan_first_fn<To, To, af_add_t, true>(tmp, tmp, tmp,
                                                      1, groups_y,
                                                      threads_x);
            } else {
                scan_first_fn<To, To,       op, true>(tmp, tmp, tmp,
                                                      1, groups_y,
                                                      threads_x);
            }

            bcast_first_fn<To, To, op, true>(out, tmp,
                                             groups_x,
                                             groups_y,
                                             threads_x);

        }
    }

}
}
