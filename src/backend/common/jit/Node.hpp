/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <backend.hpp>
#include <common/defines.hpp>
#include <optypes.hpp>
#include <platform.hpp>
#include <types.hpp>
#include <af/defines.h>

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

enum class kJITHeuristics {
    Pass                = 0, /* no eval necessary */
    TreeHeight          = 1, /* eval due to jit tree height */
    KernelParameterSize = 2, /* eval due to many kernel parameters */
    MemoryPressure      = 3  /* eval due to memory pressure */
};

namespace arrayfire {
namespace common {
class Node;
}  // namespace common
}  // namespace arrayfire

#ifdef AF_CPU
namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void evalMultiple(std::vector<Param<T>> arrays,
                  std::vector<std::shared_ptr<common::Node>> output_nodes_);
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
#endif

namespace std {
template<>
struct hash<arrayfire::common::Node *> {
    /// Calls the getHash function of the Node pointer
    size_t operator()(arrayfire::common::Node *const n) const noexcept;
};
}  // namespace std

namespace arrayfire {
namespace common {
class Node;
struct Node_ids;

/// A equal_to class that calls the dereference nodes equality operator
struct NodePtr_equalto {
    bool operator()(const Node *l, const Node *r) const noexcept;
};

using Node_map_t =
    std::unordered_map<Node *, int, std::hash<Node *>, NodePtr_equalto>;
using Node_map_iter = Node_map_t::iterator;

using Node_ptr = std::shared_ptr<Node>;

static const char *getFullName(af::dtype type) {
    switch (type) {
        case f32: return detail::getFullName<float>();
        case f64: return detail::getFullName<double>();
        case c32: return detail::getFullName<detail::cfloat>();
        case c64: return detail::getFullName<detail::cdouble>();
        case u32: return detail::getFullName<unsigned>();
        case s32: return detail::getFullName<int>();
        case u64: return detail::getFullName<unsigned long long>();
        case s64: return detail::getFullName<long long>();
        case u16: return detail::getFullName<unsigned short>();
        case s16: return detail::getFullName<short>();
        case b8: return detail::getFullName<char>();
        case u8: return detail::getFullName<unsigned char>();
        case f16: return "half";
    }
    return "";
}

static const char *getShortName(af::dtype type) {
    switch (type) {
        case f32: return detail::shortname<float>();
        case f64: return detail::shortname<double>();
        case c32: return detail::shortname<detail::cfloat>();
        case c64: return detail::shortname<detail::cdouble>();
        case u32: return detail::shortname<unsigned>();
        case s32: return detail::shortname<int>();
        case u64: return detail::shortname<unsigned long long>();
        case s64: return detail::shortname<long long>();
        case u16: return detail::shortname<unsigned short>();
        case s16: return detail::shortname<short>();
        case b8: return detail::shortname<char>();
        case u8: return detail::shortname<unsigned char>();
        case f16: return "h";
    }
    return "";
}

class Node {
   public:
    static const int kMaxChildren = 3;

   protected:
   public:
    std::array<Node_ptr, kMaxChildren> m_children;
    af::dtype m_type;
    int m_height;

    template<typename T>
    friend class NodeIterator;
    Node() = default;
    Node(const af::dtype type, const int height,
         const std::array<Node_ptr, kMaxChildren> children)
        : m_children(children), m_type(type), m_height(height) {
        static_assert(std::is_nothrow_move_assignable<Node>::value,
                      "Node is not move assignable");
    }

    void swap(Node &other) noexcept {
        using std::swap;
        for (int i = 0; i < kMaxChildren; i++) {
            swap(m_children[i], other.m_children[i]);
        }
        swap(m_type, other.m_type);
        swap(m_height, other.m_height);
    }

    /// Default move constructor operator
    Node(Node &&node) noexcept = default;

    /// Default copy constructor operator
    Node(const Node &node) = default;

    /// Default copy assignment operator
    Node &operator=(const Node &node) = default;

    /// Default move assignment operator
    Node &operator=(Node &&node) noexcept = default;

    virtual af_op_t getOp() const noexcept { return af_none_t; }

    int getNodesMap(Node_map_t &node_map, std::vector<Node *> &full_nodes,
                    std::vector<Node_ids> &full_ids);

    /// Generates the string that will be used to hash the kernel
    virtual void genKerName(std::string &kerString,
                            const Node_ids &ids) const = 0;

    /// Generates the function parameters for the node.
    ///
    /// \param[in/out] kerStream  The string will be written to this stream
    /// \param[in]     ids        The integer id of the node and its children
    /// \param[in]     is_linear  True if the kernel is a linear kernel
    virtual void genParams(std::stringstream &kerStream, int id,
                           bool is_linear) const {
        UNUSED(kerStream);
        UNUSED(id);
        UNUSED(is_linear);
    }

    virtual void calc(int x, int y, int z, int w, int lim) {
        UNUSED(x);
        UNUSED(y);
        UNUSED(z);
        UNUSED(w);
    }

    virtual void calc(int idx, int lim) {
        UNUSED(idx);
        UNUSED(lim);
    }

    const std::array<Node_ptr, kMaxChildren> &getChildren() const {
        return m_children;
    }

    /// Generates the variable that stores the thread's/work-item's offset into
    /// the memory.
    ///
    /// \param[in/out] kerStream  The string will be written to this stream
    /// \param[in]     ids        The integer id of the node and its children
    /// \param[in]     is_linear  True if the kernel is a linear kernel
    virtual void genOffsets(std::stringstream &kerStream, int id,
                            bool is_linear) const {
        UNUSED(kerStream);
        UNUSED(id);
        UNUSED(is_linear);
    }

    /// Generates the code for the operation of the node.
    ///
    /// Generates the soruce code of the operation that the node needs to
    /// perform. For example this function will create the string
    /// "val2 = __add(val1, val2);" for the addition node.
    ///
    /// \param[in/out] kerStream  The string will be written to this stream
    /// \param[in]     ids        The integer id of the node and its children
    /// \param[in]     is_linear  True if the kernel is a linear kernel
    virtual void genFuncs(std::stringstream &kerStream,
                          const Node_ids &ids) const = 0;

    /// Calls the setArg function on each of the arguments passed into the
    /// kernel
    ///
    /// \param[in] start_id The index of the staring argument
    /// \param[in] is_linear determines if the kernel should be linear or not
    /// \param[in] setArg the function that will be called for each argument
    ///
    /// \returns the next index that will need to be set in the kernl. This
    ///          is usually start_id + the number of times setArg is called
    virtual int setArgs(
        int start_id, bool is_linear,
        std::function<void(int id, const void *ptr, size_t arg_size)> setArg)
        const {
        UNUSED(is_linear);
        UNUSED(setArg);
        return start_id;
    }

    virtual void getInfo(unsigned &len, unsigned &buf_count,
                         unsigned &bytes) const {
        UNUSED(buf_count);
        UNUSED(bytes);
        len++;
    }

    // Return the size of the parameter in bytes that will be passed to the
    // kernel
    virtual size_t getParamBytes() const { return 0; }

    // Return the size of the size of the buffer node in bytes. Zero otherwise
    virtual size_t getBytes() const { return 0; }

    // Returns true if this node is a Buffer
    virtual bool isBuffer() const { return false; }

    // Returns true if this node is a Scalar
    virtual bool isScalar() const { return false; }

    /// Returns true if the buffer is linear
    virtual bool isLinear(const dim_t dims[4]) const;

    /// Returns the type
    af::dtype getType() const { return m_type; }

    /// Returns the string representation of the type
    std::string getTypeStr() const { return getFullName(m_type); }

    /// Returns the height of the JIT tree from this node
    int getHeight() const { return m_height; }

    /// Returns the short name for this type
    /// \note For the shift node this is "Sh" appended by the short name of the
    ///       type
    virtual std::string getNameStr() const { return getShortName(m_type); }

    /// Default destructor
    virtual ~Node() noexcept = default;

    /// Returns the hash of the node. For all Nodes other than the Buffer node,
    /// this is the pointer of the object
    virtual size_t getHash() const noexcept {
        std::hash<const void *> ptr_hash;
        std::hash<af::dtype> aftype_hash;
        std::hash<int> int_hash;
        const void *ptr = this;
        size_t h =
            ptr_hash(ptr) ^ (aftype_hash(m_type) << 1) ^ (int_hash(m_height));
        return h;
    }

    /// A very bad equality operator used only for the hash function.
    virtual bool operator==(const Node &other) const noexcept {
        return this == &other;
    }
    virtual std::unique_ptr<Node> clone() = 0;

#ifdef AF_CPU
    template<typename U>
    friend void arrayfire::cpu::kernel::evalMultiple(
        std::vector<arrayfire::cpu::Param<U>> arrays,
        std::vector<common::Node_ptr> output_nodes_);

    virtual void setShape(af::dim4 new_shape) { UNUSED(new_shape); }

#endif
};

struct Node_ids {
    std::array<int, Node::kMaxChildren> child_ids;
    int id;
};

std::string getFuncName(const std::vector<Node *> &output_nodes,
                        const std::vector<Node *> &full_nodes,
                        const std::vector<Node_ids> &full_ids,
                        const bool is_linear, const bool loop0,
                        const bool loop1, const bool loop2, const bool loop3);

auto isBuffer(const Node &ptr) -> bool;

auto isScalar(const Node &ptr) -> bool;

}  // namespace common
}  // namespace arrayfire
