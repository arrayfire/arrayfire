template<typename T, typename aT>
class conv2HelperCreateKernel {
   public:
    conv2HelperCreateKernel(write_accessor<T> out, KParam oInfo,
                            read_accessor<T> signal, KParam sInfo,
                            read_accessor<aT> impulse, KParam fInfo, int nBBS0,
                            int nBBS1, int ostep2, int ostep3, int sstep2,
                            int sstep3, local_accessor<aT> localMem,
                            const int f0, const int f1, const bool expand)
        : out_(out)
        , oInfo_(oInfo)
        , signal_(signal)
        , sInfo_(sInfo)
        , impulse_(impulse)
        , fInfo_(fInfo)
        , nBBS0_(nBBS0)
        , nBBS1_(nBBS1)
        , ostep2_(ostep2)
        , ostep3_(ostep3)
        , sstep2_(sstep2)
        , sstep3_(sstep3)
        , localMem_(localMem)
        , f0_(f0)
        , f1_(f1)
        , expand_(expand) {}
    void operator()(sycl::nd_item<3> it) const {
        sycl::group g = it.get_group();

        int radius0  = f0_ - 1;
        int radius1  = f1_ - 1;
        int padding0 = 2 * radius0;
        int padding1 = 2 * radius1;
        int shrdLen0 = g.get_local_range(0) + padding0;
        int shrdLen1 = g.get_local_range(1) + padding1;

        unsigned b0 = g.get_group_id(0) / nBBS0_;
        unsigned b1 = g.get_group_id(1) / nBBS1_;

        T *dst =
            out_.get_pointer() +
            (b0 * oInfo_.strides[2] + /* activated with batched input signal_ */
             ostep2_ *
                 oInfo_.strides[2] +  /* activated with batched input filter */
             b1 * oInfo_.strides[3] + /* activated with batched input signal_ */
             ostep3_ *
                 oInfo_.strides[3]); /* activated with batched input filter */

        const T *src =
            signal_.get_pointer() + sInfo_.offset +
            (b0 * sInfo_.strides[2] + /* activated with batched input signal_ */
             sstep2_ *
                 sInfo_.strides[2] +  /* activated with batched input filter */
             b1 * sInfo_.strides[3] + /* activated with batched input signal_ */
             sstep3_ *
                 sInfo_.strides[3]); /* activated with batched input filter */

        int lx = it.get_local_id(0);
        int ly = it.get_local_id(1);
        int gx = g.get_local_range(0) * (g.get_group_id(0) - b0 * nBBS0_) + lx;
        int gy = g.get_local_range(1) * (g.get_group_id(1) - b1 * nBBS1_) + ly;

        // below loops are traditional loops, they only run multiple
        // times filter length is more than launch size
        int s0 = sInfo_.strides[0];
        int s1 = sInfo_.strides[1];
        int d0 = sInfo_.dims[0];
        int d1 = sInfo_.dims[1];
        for (int b = ly, gy2 = gy; b < shrdLen1;
             b += g.get_local_range(1), gy2 += g.get_local_range(1)) {
            int j     = gy2 - radius1;
            bool is_j = j >= 0 && j < d1;
            // move row_set g.get_local_range(1) along coloumns
            for (int a = lx, gx2 = gx; a < shrdLen0;
                 a += g.get_local_range(0), gx2 += g.get_local_range(0)) {
                int i     = gx2 - radius0;
                bool is_i = i >= 0 && i < d0;
                localMem_[b * shrdLen0 + a] =
                    (is_i && is_j ? src[i * s0 + j * s1] : (T)(0));
            }
        }
        it.barrier();

        if (gx < oInfo_.dims[0] && gy < oInfo_.dims[1]) {
            int ci = lx + radius0 + (expand_ ? 0 : f0_ >> 1);
            int cj = ly + radius1 + (expand_ ? 0 : f1_ >> 1);

            aT accum = (aT)(0);
            for (int fj = 0; fj < f1_; ++fj) {
                for (int fi = 0; fi < f0_; ++fi) {
                    aT f_val = impulse_[fj * f0_ + fi];
                    T s_val  = localMem_[(cj - fj) * shrdLen0 + (ci - fi)];

                    // binOp will do MUL_OP for convolution operation
                    accum = accum + binOp((aT)s_val, (aT)f_val);
                }
            }
            dst[gy * oInfo_.strides[1] + gx] = (T)accum;
        }
    }

   private:
    write_accessor<T> out_;
    KParam oInfo_;
    read_accessor<T> signal_;
    KParam sInfo_;
    read_accessor<aT> impulse_;
    KParam fInfo_;
    int nBBS0_;
    int nBBS1_;
    int ostep2_;
    int ostep3_;
    int sstep2_;
    int sstep3_;
    local_accessor<aT> localMem_;
    const int f0_;
    const int f1_;
    const bool expand_;
};

template<typename T, typename aT>
void conv2Helper(const conv_kparam_t<aT> &param, Param<T> out,
                 const Param<T> signal, const Param<aT> filter,
                 const bool expand) {
    constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    const int f0 = filter.info.dims[0];
    const int f1 = filter.info.dims[1];
    const size_t LOC_SIZE =
        (THREADS_X + 2 * (f0 - 1)) * (THREADS_Y + 2 * (f1 - 1));

    auto Q = getQueue();
    Q.submit([&](auto &h) {
        sycl::accessor<aT, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            localMem(LOC_SIZE, h);
        sycl::accessor outAcc{*out.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor signalAcc{*signal.data, h, sycl::read_only};
        sycl::accessor impulseAcc{*param.impulse, h, sycl::read_only};
        h.parallel_for(
            sycl::nd_range{param.global, param.local},
            conv2HelperCreateKernel<T, aT>(
                outAcc, out.info, signalAcc, signal.info, impulseAcc,
                filter.info, param.nBBS0, param.nBBS1, param.o[1], param.o[2],
                param.s[1], param.s[2], localMem, f0, f1, expand));
    });
    ONEAPI_DEBUG_FINISH(Q);
}

template<typename T, typename aT>
void conv2(conv_kparam_t<aT> &p, Param<T> &out, const Param<T> &sig,
           const Param<aT> &filt, const bool expand) {
    size_t se_size = filt.info.dims[0] * filt.info.dims[1];
    sycl::buffer<aT> impulse{sycl::range(se_size)};
    int f0Off = filt.info.offset;

    for (int b3 = 0; b3 < filt.info.dims[3]; ++b3) {
        int f3Off = b3 * filt.info.strides[3];

        for (int b2 = 0; b2 < filt.info.dims[2]; ++b2) {
            int f2Off = b2 * filt.info.strides[2];

            const size_t srcOffset = f2Off + f3Off + f0Off;
            if constexpr (!(std::is_same_v<T, double> ||
                            std::is_same_v<T, cdouble> ||
                            std::is_same_v<aT, double> ||
                            std::is_same_v<aT, cdouble>)) {
                memcpyBuffer(impulse, *filt.data, se_size, srcOffset);
            }
            p.impulse = &impulse;

            p.o[1] = (p.outHasNoOffset ? 0 : b2);
            p.o[2] = (p.outHasNoOffset ? 0 : b3);
            p.s[1] = (p.inHasNoOffset ? 0 : b2);
            p.s[2] = (p.inHasNoOffset ? 0 : b3);

            if constexpr (!(std::is_same_v<T, double> ||
                            std::is_same_v<T, cdouble> ||
                            std::is_same_v<aT, double> ||
                            std::is_same_v<aT, cdouble>)) {
                conv2Helper<T, aT>(p, out, sig, filt, expand);
            }
        }
    }
}

#define INSTANTIATE_CONV2(T, aT)                                    \
    template void conv2<T, aT>(conv_kparam_t<aT> &, Param<T> &,     \
                               const Param<T> &, const Param<aT> &, \
                               const bool);

INSTANTIATE_CONV2(char, float)
INSTANTIATE_CONV2(cfloat, cfloat)
INSTANTIATE_CONV2(cdouble, cdouble)
INSTANTIATE_CONV2(float, float)
INSTANTIATE_CONV2(double, double)
INSTANTIATE_CONV2(short, float)
INSTANTIATE_CONV2(int, float)
INSTANTIATE_CONV2(intl, float)
INSTANTIATE_CONV2(ushort, float)
INSTANTIATE_CONV2(uint, float)
INSTANTIATE_CONV2(uintl, float)
INSTANTIATE_CONV2(uchar, float)
