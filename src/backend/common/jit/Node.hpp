/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <optypes.hpp>
#include <platform.hpp>

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace common {
class Node;
struct Node_ids;

using Node_ptr      = std::shared_ptr<Node>;
using Node_map_t    = std::unordered_map<const Node *, int>;
using Node_map_iter = Node_map_t::iterator;

class Node {
   public:
    static const int kMaxChildren = 3;

   protected:
    const std::array<Node_ptr, kMaxChildren> m_children;
    const std::string m_type_str;
    const std::string m_name_str;
    const int m_height;
    template<typename T>
    friend class NodeIterator;

   public:
    Node(const char *type_str, const char *name_str, const int height,
         const std::array<Node_ptr, kMaxChildren> children)
        : m_children(children)
        , m_type_str(type_str)
        , m_name_str(name_str)
        , m_height(height) {}

    int getNodesMap(Node_map_t &node_map, std::vector<const Node *> &full_nodes,
                    std::vector<Node_ids> &full_ids) const;

    virtual void genKerName(std::stringstream &kerStream,
                            const Node_ids &ids) const {
        UNUSED(kerStream);
        UNUSED(ids);
    }
    virtual void genParams(std::stringstream &kerStream, int id,
                           bool is_linear) const {
        UNUSED(kerStream);
        UNUSED(id);
        UNUSED(is_linear);
    }
    virtual void genOffsets(std::stringstream &kerStream, int id,
                            bool is_linear) const {
        UNUSED(kerStream);
        UNUSED(id);
        UNUSED(is_linear);
    }
    virtual void genFuncs(std::stringstream &kerStream,
                          const Node_ids &ids) const {
        UNUSED(kerStream);
        UNUSED(ids);
    }

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
    virtual bool isBuffer() const { return false; }
    virtual bool isLinear(dim_t dims[4]) const {
        UNUSED(dims);
        return true;
    }
    std::string getTypeStr() const { return m_type_str; }
    int getHeight() const { return m_height; }
    std::string getNameStr() const { return m_name_str; }

    virtual ~Node() {}
};

struct Node_ids {
    std::array<int, Node::kMaxChildren> child_ids;
    int id;
};
}  // namespace common
