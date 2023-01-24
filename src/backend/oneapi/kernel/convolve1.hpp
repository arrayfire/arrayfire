template<typename T, typename aT>
class conv1HelperCreateKernel {
   public:
    conv1HelperCreateKernel(write_accessor<T> out, KParam oInfo,
                            read_accessor<T> signal, KParam sInfo,
                            local_accessor<aT> localMem,
                            read_accessor<aT> impulse, KParam fInfo, int nBBS0,
                            int nBBS1, int ostep1, int ostep2, int ostep3,
                            int sstep1, int sstep2, int sstep3,
                            const bool expand)
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
        , expand_(expand) {}
    void operator()(sycl::nd_item<3> it) const {
        sycl::group g = it.get_group();

        int fLen          = fInfo_.dims[0];
        int padding       = fLen - 1;
        int shrdLen       = g.get_local_range(0) + 2 * padding;
        const unsigned b1 = g.get_group_id(0) / nBBS0_;
        const unsigned b0 = g.get_group_id(0) - nBBS0_ * b1;
        const unsigned b3 = g.get_group_id(1) / nBBS1_;
        const unsigned b2 = g.get_group_id(1) - nBBS1_ * b3;

        T *dst =
            out_.get_pointer() +
            (b1 * oInfo_.strides[1] + /* activated with batched input signal_ */
             ostep1_ *
                 oInfo_.strides[1] +  /* activated with batched input filter */
             b2 * oInfo_.strides[2] + /* activated with batched input signal_ */
             ostep2_ *
                 oInfo_.strides[2] +  /* activated with batched input filter */
             b3 * oInfo_.strides[3] + /* activated with batched input signal_ */
             ostep3_ *
                 oInfo_.strides[3]); /* activated with batched input filter */

        T const *src =
            signal_.get_pointer() + sInfo_.offset +
            (b1 * sInfo_.strides[1] + /* activated with batched input signal_ */
             sstep1_ *
                 sInfo_.strides[1] +  /* activated with batched input filter */
             b2 * sInfo_.strides[2] + /* activated with batched input signal_ */
             sstep2_ *
                 sInfo_.strides[2] +  /* activated with batched input filter */
             b3 * sInfo_.strides[3] + /* activated with batched input signal_ */
             sstep3_ *
                 sInfo_.strides[3]); /* activated with batched input filter */

        int gx = g.get_local_range(0) * b0;

        for (int i = it.get_local_id(0); i < shrdLen;
             i += g.get_local_range(0)) {
            int idx      = gx - padding + i;
            localMem_[i] = (idx >= 0 && idx < sInfo_.dims[0])
                               ? src[idx * sInfo_.strides[0]]
                               : (T)(0);
        }
        it.barrier();
        gx += it.get_local_id(0);

        if (gx >= 0 && gx < oInfo_.dims[0]) {
            int lx   = it.get_local_id(0) + padding + (expand_ ? 0 : fLen >> 1);
            aT accum = (aT)(0);
            for (int f = 0; f < fLen; ++f) {
                // binOp will do MUL_OP for convolution operation
                accum = accum + binOp((aT)localMem_[lx - f], (aT)impulse_[f]);
            }
            dst[gx] = (T)accum;
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
    const bool expand_;
};

template<typename T, typename aT>
void conv1Helper(const conv_kparam_t<aT> &param, Param<T> &out,
                 const Param<T> &signal, const Param<aT> &filter,
                 const int rank, const bool expand) {
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
            conv1HelperCreateKernel<T, aT>(
                outAcc, out.info, signalAcc, signal.info, localMem, impulseAcc,
                filter.info, param.nBBS0, param.nBBS1, param.o[0], param.o[1],
                param.o[2], param.s[0], param.s[1], param.s[2], expand));
    });
    ONEAPI_DEBUG_FINISH(Q);
}

template<typename T, typename aT>
void conv1(conv_kparam_t<aT> &p, Param<T> &out, const Param<T> &sig,
           const Param<aT> &filt, const bool expand) {
    const size_t se_size = filt.info.dims[0];
    sycl::buffer<aT> impulse{sycl::range(filt.info.dims[0])};
    int f0Off = filt.info.offset;
    for (int b3 = 0; b3 < filt.info.dims[3]; ++b3) {
        int f3Off = b3 * filt.info.strides[3];

        for (int b2 = 0; b2 < filt.info.dims[2]; ++b2) {
            int f2Off = b2 * filt.info.strides[2];

            for (int b1 = 0; b1 < filt.info.dims[1]; ++b1) {
                int f1Off = b1 * filt.info.strides[1];

                const size_t srcOffset = f0Off + f1Off + f2Off + f3Off;
                if constexpr (!(std::is_same_v<T, double> ||
                                std::is_same_v<T, cdouble> ||
                                std::is_same_v<aT, double> ||
                                std::is_same_v<aT, cdouble>)) {
                    memcpyBuffer(impulse, *filt.data, se_size, srcOffset);
                }
                p.impulse = &impulse;

                p.o[0] = (p.outHasNoOffset ? 0 : b1);
                p.o[1] = (p.outHasNoOffset ? 0 : b2);
                p.o[2] = (p.outHasNoOffset ? 0 : b3);
                p.s[0] = (p.inHasNoOffset ? 0 : b1);
                p.s[1] = (p.inHasNoOffset ? 0 : b2);
                p.s[2] = (p.inHasNoOffset ? 0 : b3);

                if constexpr (!(std::is_same_v<T, double> ||
                                std::is_same_v<T, cdouble> ||
                                std::is_same_v<aT, double> ||
                                std::is_same_v<aT, cdouble>)) {
                    conv1Helper<T, aT>(p, out, sig, filt, 1, expand);
                }
            }
        }
    }
}

#define INSTANTIATE_CONV1(T, aT)                                    \
    template void conv1<T, aT>(conv_kparam_t<aT> &, Param<T> &,     \
                               const Param<T> &, const Param<aT> &, \
                               const bool);

INSTANTIATE_CONV1(cdouble, cdouble)
INSTANTIATE_CONV1(cfloat, cfloat)
INSTANTIATE_CONV1(double, double)
INSTANTIATE_CONV1(float, float)
INSTANTIATE_CONV1(uint, float)
INSTANTIATE_CONV1(int, float)
INSTANTIATE_CONV1(uchar, float)
INSTANTIATE_CONV1(char, float)
INSTANTIATE_CONV1(ushort, float)
INSTANTIATE_CONV1(short, float)
INSTANTIATE_CONV1(uintl, float)
INSTANTIATE_CONV1(intl, float)
