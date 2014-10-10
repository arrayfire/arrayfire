#include <af/dim4.hpp>
#include <Array.hpp>
#include <iostream>
#include <stdexcept>
#include <copy.hpp>
#include <JIT/BufferNode.hpp>

using af::dim4;

namespace opencl
{
    using std::ostream;

    template<typename T>
    Array<T>::Array(af::dim4 dims) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(getContext(), CL_MEM_READ_WRITE, ArrayInfo::elements()*sizeof(T)),
        parent(), node(NULL), ready(true)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, JIT::Node *n) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(),
        parent(), node(n), ready(false)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, T val) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(getContext(), CL_MEM_READ_WRITE, ArrayInfo::elements()*sizeof(T)),
        parent(), node(NULL), ready(true)
    {
        set(data, val, elements());
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, const T * const in_data) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(getContext(), CL_MEM_READ_WRITE, ArrayInfo::elements()*sizeof(T)),
        parent(), node(NULL), ready(true)
    {
        getQueue().enqueueWriteBuffer(data,CL_TRUE,0,sizeof(T)*ArrayInfo::elements(),in_data);
    }

    template<typename T>
    Array<T>::Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride) :
        ArrayInfo(dims, offset, stride, (af_dtype)dtype_traits<T>::af_type),
        data(0),
        parent(&parnt), node(NULL), ready(true)
    { }


    template<typename T>
    Array<T>::Array(Param &tmp) :
        ArrayInfo(af::dim4(tmp.info.dims[0], tmp.info.dims[1], tmp.info.dims[2], tmp.info.dims[3]),
                  af::dim4(0, 0, 0, 0),
                  af::dim4(tmp.info.strides[0], tmp.info.strides[1],
                           tmp.info.strides[2], tmp.info.strides[3]),
                  (af_dtype)dtype_traits<T>::af_type),
        data(tmp.data),
        parent(), node(NULL), ready(true)
    {
    }

    template<typename T>
    Array<T>::~Array()
    { }

    using JIT::BufferNode;
    using JIT::Node;

    template<typename T>
    Node* Array<T>::getNode() const
    {
        if (node == NULL) {
            BufferNode *buf_node = new BufferNode(dtype_traits<T>::getName(),
                                                  shortname<T>(true), *this);
            const_cast<Array<T> *>(this)->node = reinterpret_cast<Node *>(buf_node);
        }

        return node;
    }

    using af::dim4;

    template<typename T>
    Array<T> *
    createNodeArray(const dim4 &dims, Node *node)
    {
        return new Array<T>(dims, node);
    }

    template<typename T>
    Array<T> *
    createSubArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride)
    {

        Array<T> *out = new Array<T>(parent, dims, offset, stride);

        if (stride[0] != 1 ||
            stride[1] <  0 ||
            stride[2] <  0 ||
            stride[3] <  0) {

            out = copyArray(*out);
        }

        return out;
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
        Array<T> *out = new Array<T>(size, value);
        return out;
    }

    template<typename T>
    Array<T>*
    createEmptyArray(const dim4 &size)
    {
        Array<T> *out = new Array<T>(size);
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
    createParamArray(Param &tmp)
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
        std::stringstream Stream;

        int id = node->setId(0) - 1;

        Stream << "__kernel " << std::endl;
        Stream << "Kernel_";
        node->genKerName(Stream, false);
        Stream << "_";
        node->genKerName(Stream, true);
        Stream << "(" << std::endl;

        node->genParams(Stream);
        Stream << "__global " << node->getTypeStr() << " *out, KParam oInfo," << std::endl;
        Stream << "uint groups_0, uint groups_1)" << std::endl;

        Stream << "{" << std::endl << std::endl;

        Stream << "uint id2 = get_group_id(0) / groups_0;" << std::endl;
        Stream << "uint id3 = get_group_id(1) / groups_1;" << std::endl;
        Stream << "uint groupId_0 = get_group_id(0) - id2 * groups_0;" << std::endl;
        Stream << "uint groupId_1 = get_group_id(1) - id3 * groups_1;" << std::endl;
        Stream << "uint id1 = get_local_id(1) + groupId_1 * get_local_size(1);" << std::endl;
        Stream << "uint id0 = get_local_id(0) + groupId_0 * get_local_size(0);" << std::endl;
        Stream << std::endl;

        Stream << "bool cond = " << std::endl;
        Stream << "id0 < oInfo.dims[0] && " << std::endl;
        Stream << "id1 < oInfo.dims[1] && " << std::endl;
        Stream << "id2 < oInfo.dims[2] && " << std::endl;
        Stream << "id3 < oInfo.dims[3];" << std::endl << std::endl;

        Stream << "if (!cond) return;" << std::endl << std::endl;

        node->genOffsets(Stream);
        Stream << "int idx = ";
        Stream << "oInfo.strides[3] * id3 + oInfo.strides[2] * id2 + ";
        Stream << "oInfo.strides[1] * id1 + id0 + oInfo.offset;" << std::endl << std::endl;

        node->genFuncs(Stream);
        Stream << std::endl;

        Stream << "out[idx] = val"
               << id << ";"  << std::endl;

        Stream << "}" << std::endl;

        std::cout << Stream.str();

        data = cl::Buffer(getContext(), CL_MEM_READ_WRITE, elements() * sizeof(T));
        set(data, 0, elements());
        node = nullptr;
        ready = true;
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
    template       Array<T>*  createParamArray<T> (Param &tmp);         \
    template       Array<T>*  createSubArray<T>   (const Array<T> &parent, const dim4 &dims, \
                                                   const dim4 &offset, const dim4 &stride); \
    template       Array<T>*  createNodeArray<T>   (const dim4 &size, JIT::Node *node); \
    template       JIT::Node* Array<T>::getNode() const;                \
    template       void Array<T>::eval();                               \
    template       void Array<T>::eval() const;                         \
    template       void       destroyArray<T>     (Array<T> &A);        \
    template                  Array<T>::~Array();

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
