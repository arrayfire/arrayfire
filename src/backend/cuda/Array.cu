#include <af/dim4.hpp>
#include <Array.hpp>
#include <stdexcept>
#include <copy.hpp>
#include <kernel/elwise.hpp> //set
#include <err_cuda.hpp>
#include <JIT/BufferNode.hpp>
#include <scalar.hpp>

using af::dim4;

namespace cuda
{

    using std::ostream;

    template<typename T>
    T* cudaMallocWrapper(const size_t &elements)
    {
        T* ptr = NULL;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * elements));
        return ptr;
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(cudaMallocWrapper<T>(dims.elements())),
        parent(), node(NULL), ready(true)
    {}

    template<typename T>
    Array<T>::Array(af::dim4 dims, const T * const in_data) :
    ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(cudaMallocWrapper<T>(dims.elements())),
        parent(), node(NULL), ready(true)
    {
        CUDA_CHECK(cudaMemcpy(data, in_data, dims.elements() * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    Array<T>::Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride) :
        ArrayInfo(dims, offset, stride, (af_dtype)dtype_traits<T>::af_type),
        data(NULL),
        parent(&parnt), node(NULL), ready(true)
    { }

    template<typename T>
    Array<T>::Array(Param<T> &tmp) :
        ArrayInfo(af::dim4(tmp.dims[0], tmp.dims[1], tmp.dims[2], tmp.dims[3]),
                  af::dim4(0, 0, 0, 0),
                  af::dim4(tmp.strides[0], tmp.strides[1], tmp.strides[2], tmp.strides[3]),
                  (af_dtype)dtype_traits<T>::af_type),
        data(tmp.ptr),
        parent(), node(NULL), ready(true)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, JIT::Node *n) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(NULL),
        parent(), node(n), ready(false)
    {
    }

    template<typename T>
    Array<T>::~Array() { if (!data) CUDA_CHECK(cudaFree(data)); }


    using JIT::BufferNode;
    using JIT::Node;

    template<typename T>
    Node* Array<T>::getNode() const
    {
        if (node == NULL) {
            CParam<T> this_param = *this;
            BufferNode<T> *buf_node = new BufferNode<T>(irname<T>(),
                                                        shortname<T>(true), this_param);
            const_cast<Array<T> *>(this)->node = reinterpret_cast<Node *>(buf_node);
        }

        return node;
    }

    template<typename T>
    Array<T> *
    createNodeArray(const dim4 &dims, Node *node)
    {
        return new Array<T>(dims, node);
    }

    template<typename T>
    Array<T> *
    createDataArray(const dim4 &size, const T * const data)
    {
        Array<T> *out = new Array<T>(size, data);
        return out;
    }

    template<typename T>
    Array<T>*
    createValueArray(const dim4 &size, const T& value)
    {
        return createScalarNode<T>(size, value);
    }

    template<typename T>
    Array<T>*
    createEmptyArray(const dim4 &size)
    {
        Array<T> *out = new Array<T>(size);
        return out;
    }

    template<typename T>
    Array<T> *
    createSubArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride)
    {

        Array<T> *out = new Array<T>(parent, dims, offset, stride);

        // FIXME: Implement this for CUDA
        if (stride[0] != 1 ||
            stride[1] <  0 ||
            stride[2] <  0 ||
            stride[3] <  0) {
            out = copyArray(*out);
        }

        return out;
    }

    template<typename inType, typename outType>
    Array<outType> *
    createPaddedArray(Array<inType> const &in, dim4 const &dims, outType default_value, double factor)
    {
        Array<outType> *ret = createEmptyArray<outType>(dims);

        copy<inType, outType>(*ret, in, default_value, factor);

        return ret;
    }

    template<typename T>
    Array<T>*
    createParamArray(Param<T> &tmp)
    {
        Array<T> *out = new Array<T>(tmp);
        return out;
    }

    template<typename T>
    void
    destroyArray(Array<T> &A)
    {
        delete &A;
    }

    template<typename T>
    void Array<T>::eval()
    {
        if (isReady()) return;

        data = cudaMallocWrapper<T>(elements());

        Param<T> res;
        res.ptr = data;

        for (int  i = 0; i < 4; i++) {
            res.dims[i] = dims()[i];
            res.strides[i] = strides()[i];
        }

        evalNodes(res, this->getNode());
        ready = true;

        // Replace the current node in any JIT possible trees with the new BufferNode
        Node *prev = node;
        node = NULL;
        prev->resetFlags();
        prev->replace(getNode());
    }

    template<typename T>
    void Array<T>::eval() const
    {
        if (isReady()) return;
        const_cast<Array<T> *>(this)->eval();
    }

#define INSTANTIATE(T)                                                  \
    template       Array<T>*  createDataArray<T>  (const dim4 &size, const T * const data); \
    template       Array<T>*  createValueArray<T> (const dim4 &size, const T &value); \
    template       Array<T>*  createEmptyArray<T> (const dim4 &size);   \
    template       Array<T>*  createParamArray<T> (Param<T> &tmp);      \
    template       Array<T>*  createSubArray<T>       (const Array<T> &parent, const dim4 &dims, const dim4 &offset, const dim4 &stride); \
    template       void       destroyArray<T>     (Array<T> &A);        \
    template       Array<T>*  createNodeArray<T>   (const dim4 &size, JIT::Node *node); \
    template                  Array<T>::~Array();                       \
    template       void Array<T>::eval();                               \
    template       void Array<T>::eval() const;                         \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

#define INSTANTIATE_CREATE_PADDED_ARRAY(SRC_T) \
    template Array<float  >* createPaddedArray<SRC_T, float  >(Array<SRC_T> const &src, dim4 const &dims, float   default_value, double factor); \
    template Array<double >* createPaddedArray<SRC_T, double >(Array<SRC_T> const &src, dim4 const &dims, double  default_value, double factor); \
    template Array<cfloat >* createPaddedArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble>* createPaddedArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor); \
    template Array<int    >* createPaddedArray<SRC_T, int    >(Array<SRC_T> const &src, dim4 const &dims, int     default_value, double factor); \
    template Array<uint   >* createPaddedArray<SRC_T, uint   >(Array<SRC_T> const &src, dim4 const &dims, uint    default_value, double factor); \
    template Array<uchar  >* createPaddedArray<SRC_T, uchar  >(Array<SRC_T> const &src, dim4 const &dims, uchar   default_value, double factor); \
    template Array<char   >* createPaddedArray<SRC_T, char   >(Array<SRC_T> const &src, dim4 const &dims, char    default_value, double factor);

    INSTANTIATE_CREATE_PADDED_ARRAY(float )
    INSTANTIATE_CREATE_PADDED_ARRAY(double)
    INSTANTIATE_CREATE_PADDED_ARRAY(int   )
    INSTANTIATE_CREATE_PADDED_ARRAY(uint  )
    INSTANTIATE_CREATE_PADDED_ARRAY(uchar )
    INSTANTIATE_CREATE_PADDED_ARRAY(char  )

#define INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(SRC_T) \
    template Array<cfloat >* createPaddedArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble>* createPaddedArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor);

    INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(cfloat )
    INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(cdouble)

}
