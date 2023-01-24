int index(int i, int j, int k, int jstride, int kstride) {
    return i + j * jstride + k * kstride;
}

template<typename T, typename aT>
class conv3HelperCreateKernel {
   public:
    conv3HelperCreateKernel(write_accessor<T> out, KParam oInfo,
                            read_accessor<T> signal, KParam sInfo,
                            local_accessor<aT> localMem,
                            read_accessor<aT> impulse, KParam fInfo, int nBBS0,
                            int nBBS1, int ostep1, int ostep2, int ostep3,
                            int sstep1, int sstep2, int sstep3,
                            const bool EXPAND)
        : out_(out)
        , oInfo_(oInfo)
        , signal_(signal)
        , sInfo_(sInfo)
        , localMem_(localMem)
        , impulse_(impulse)
        , fInfo_(fInfo)
        , nBBS0_(nBBS0)
        , nBBS1_(nBBS1)
        , ostep1_(ostep1)
        , ostep2_(ostep2)
        , ostep3_(ostep3)
        , sstep1_(sstep1)
        , sstep2_(sstep2)
        , sstep3_(sstep3)
        , EXPAND_(EXPAND) {}
    void operator()(sycl::nd_item<3> it) const {
        sycl::group g = it.get_group();
        int fLen0     = fInfo_.dims[0];
        int fLen1     = fInfo_.dims[1];
        int fLen2     = fInfo_.dims[2];
        int radius0   = fLen0 - 1;
        int radius1   = fLen1 - 1;
        int radius2   = fLen2 - 1;
        int shrdLen0  = g.get_local_range(0) + 2 * radius0;
        int shrdLen1  = g.get_local_range(1) + 2 * radius1;
        int shrdLen2  = g.get_local_range(2) + 2 * radius2;
        int skStride  = shrdLen0 * shrdLen1;
        int fStride   = fLen0 * fLen1;
        unsigned b2   = g.get_group_id(0) / nBBS0_;

        T *dst =
            out_.get_pointer() +
            (b2 * oInfo_.strides[3] + /* activated with batched input signal_ */
             ostep3_ *
                 oInfo_.strides[3]); /* activated with batched input filter */

        const T *src =
            signal_.get_pointer() + sInfo_.offset +
            (b2 * sInfo_.strides[3] + /* activated with batched input signal_ */
             sstep3_ *
                 sInfo_.strides[3]); /* activated with batched input filter */

        int lx  = it.get_local_id(0);
        int ly  = it.get_local_id(1);
        int lz  = it.get_local_id(2);
        int gx  = g.get_local_range(0) * (g.get_group_id(0) - b2 * nBBS0_) + lx;
        int gy  = g.get_local_range(1) * g.get_group_id(1) + ly;
        int gz  = g.get_local_range(2) * g.get_group_id(2) + lz;
        int lx2 = lx + g.get_local_range(0);
        int ly2 = ly + g.get_local_range(1);
        int lz2 = lz + g.get_local_range(2);
        int gx2 = gx + g.get_local_range(0);
        int gy2 = gy + g.get_local_range(1);
        int gz2 = gz + g.get_local_range(2);

        int s0 = sInfo_.strides[0];
        int s1 = sInfo_.strides[1];
        int s2 = sInfo_.strides[2];
        int d0 = sInfo_.dims[0];
        int d1 = sInfo_.dims[1];
        int d2 = sInfo_.dims[2];

        for (int c = lz, gz2 = gz; c < shrdLen2;
             c += g.get_local_range(2), gz2 += g.get_local_range(2)) {
            int k     = gz2 - radius2;
            bool is_k = k >= 0 && k < d2;
            for (int b = ly, gy2 = gy; b < shrdLen1;
                 b += g.get_local_range(1), gy2 += g.get_local_range(1)) {
                int j     = gy2 - radius1;
                bool is_j = j >= 0 && j < d1;
                for (int a = lx, gx2 = gx; a < shrdLen0;
                     a += g.get_local_range(0), gx2 += g.get_local_range(0)) {
                    int i     = gx2 - radius0;
                    bool is_i = i >= 0 && i < d0;
                    localMem_[c * skStride + b * shrdLen0 + a] =
                        (is_i && is_j && is_k ? src[i * s0 + j * s1 + k * s2]
                                              : (T)(0));
                }
            }
        }
        it.barrier();

        if (gx < oInfo_.dims[0] && gy < oInfo_.dims[1] && gz < oInfo_.dims[2]) {
            int ci = lx + radius0 + (EXPAND_ ? 0 : fLen0 >> 1);
            int cj = ly + radius1 + (EXPAND_ ? 0 : fLen1 >> 1);
            int ck = lz + radius2 + (EXPAND_ ? 0 : fLen2 >> 1);

            aT accum = (aT)(0);
            for (int fk = 0; fk < fLen2; ++fk) {
                for (int fj = 0; fj < fLen1; ++fj) {
                    for (int fi = 0; fi < fLen0; ++fi) {
                        aT f_val = impulse_[index(fi, fj, fk, fLen0, fStride)];
                        T s_val  = localMem_[index(ci - fi, cj - fj, ck - fk,
                                                   shrdLen0, skStride)];

                        // binOp will do MUL_OP for convolution operation
                        accum = accum + binOp((aT)s_val, (aT)f_val);
                    }
                }
            }
            dst[index(gx, gy, gz, oInfo_.strides[1], oInfo_.strides[2])] =
                (T)accum;
        }
    }

   private:
    write_accessor<T> out_;
    KParam oInfo_;
    read_accessor<T> signal_;
    KParam sInfo_;
    local_accessor<aT> localMem_;
    read_accessor<aT> impulse_;
    KParam fInfo_;
    int nBBS0_;
    int nBBS1_;
    int ostep1_;
    int ostep2_;
    int ostep3_;
    int sstep1_;
    int sstep2_;
    int sstep3_;
    const bool EXPAND_;
};

template<typename T, typename aT>
void conv3Helper(const conv_kparam_t<aT> &param, Param<T> &out,
                 const Param<T> &signal, const Param<aT> &impulse,
                 const int rank, const bool EXPAND) {
    auto Q = getQueue();
    Q.submit([&](auto &h) {
        sycl::accessor<aT, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            localMem(param.loc_size, h);
        sycl::accessor outAcc{*out.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor signalAcc{*signal.data, h, sycl::read_only};
        sycl::accessor impulseAcc{*param.impulse, h, sycl::read_only};
        h.parallel_for(
            sycl::nd_range{param.global, param.local},
            conv3HelperCreateKernel<T, aT>(
                outAcc, out.info, signalAcc, signal.info, localMem, impulseAcc,
                impulse.info, param.nBBS0, param.nBBS1, param.o[0], param.o[1],
                param.o[2], param.s[0], param.s[1], param.s[2], EXPAND));
    });
    ONEAPI_DEBUG_FINISH(Q);
}

template<typename T, typename aT>
void conv3(conv_kparam_t<aT> &p, Param<T> &out, const Param<T> &sig,
           const Param<aT> &filt, const bool expand) {
    size_t se_size = filt.info.dims[0] * filt.info.dims[1] * filt.info.dims[2];
    sycl::buffer<aT> impulse{sycl::range(se_size)};
    int f0Off = filt.info.offset;

    for (int b3 = 0; b3 < filt.info.dims[3]; ++b3) {
        int f3Off = b3 * filt.info.strides[3];

        const size_t srcOffset = f3Off + f0Off;
        if constexpr (!(std::is_same_v<T, double> ||
                        std::is_same_v<T, cdouble> ||
                        std::is_same_v<aT, double> ||
                        std::is_same_v<aT, cdouble>)) {
            memcpyBuffer(impulse, *filt.data, se_size, srcOffset);
        }
        p.impulse = &impulse;

        p.o[2] = (p.outHasNoOffset ? 0 : b3);
        p.s[2] = (p.inHasNoOffset ? 0 : b3);

        if constexpr (!(std::is_same_v<T, double> ||
                        std::is_same_v<T, cdouble> ||
                        std::is_same_v<aT, double> ||
                        std::is_same_v<aT, cdouble>)) {
            conv3Helper<T, aT>(p, out, sig, filt, 3, expand);
        }
    }
}

#define INSTANTIATE_CONV3(T, aT)                                    \
    template void conv3<T, aT>(conv_kparam_t<aT> &, Param<T> &,     \
                               const Param<T> &, const Param<aT> &, \
                               const bool);

INSTANTIATE_CONV3(cdouble, cdouble)
INSTANTIATE_CONV3(cfloat, cfloat)
INSTANTIATE_CONV3(double, double)
INSTANTIATE_CONV3(float, float)
INSTANTIATE_CONV3(uint, float)
INSTANTIATE_CONV3(int, float)
INSTANTIATE_CONV3(uchar, float)
INSTANTIATE_CONV3(char, float)
INSTANTIATE_CONV3(ushort, float)
INSTANTIATE_CONV3(short, float)
INSTANTIATE_CONV3(uintl, float)
INSTANTIATE_CONV3(intl, float)
