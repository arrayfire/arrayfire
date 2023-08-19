/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <jit/BufferNode.hpp>
#include <jit/kernel_generators.hpp>

#include <backend.hpp>
#include <iomanip>

#include <array>
#include <memory>
#include <sstream>
#include <string>

namespace arrayfire {
namespace common {

template<typename BufferNode>
class ShiftNodeBase : public Node {
   private:
    std::shared_ptr<BufferNode> m_buffer_node;
    std::array<int, 4> m_shifts;

   public:
    ShiftNodeBase(const af::dtype type, std::shared_ptr<BufferNode> buffer_node,
                  const std::array<int, 4> shifts)
        : Node(type, 0, {}, kNodeType::Shift)
        , m_buffer_node(buffer_node)
        , m_shifts(shifts) {
        static_assert(std::is_nothrow_move_assignable<ShiftNodeBase>::value,
                      "ShiftNode is not move assignable");
        static_assert(std::is_nothrow_move_constructible<ShiftNodeBase>::value,
                      "ShiftNode is not move constructible");
    }

    /// Default move copy constructor
    ShiftNodeBase(const ShiftNodeBase &other) = default;

    /// Default move constructor
    ShiftNodeBase(ShiftNodeBase &&other) = default;

    /// Default move/copy assignment operator(Rule of 4)
    ShiftNodeBase &operator=(ShiftNodeBase node) noexcept {
        swap(node);
        return *this;
    }

    std::array<int, 4> &getShifts() { return m_shifts; }

    std::unique_ptr<Node> clone() final {
        return std::make_unique<ShiftNodeBase>(*this);
    }

    // Swap specilization
    void swap(ShiftNodeBase &other) noexcept {
        using std::swap;
        Node::swap(other);
        swap(m_buffer_node, other.m_buffer_node);
        swap(m_shifts, other.m_shifts);
    }

    BufferNode &getBufferNode() { return *m_buffer_node; }
    const BufferNode &getBufferNode() const { return *m_buffer_node; }

    bool isLinear(const dim_t dims[4]) const final {
        UNUSED(dims);
        return false;
    }

    void genKerName(std::string &kerString,
                    const common::Node_ids &ids) const final {
        kerString += '_';
        kerString += getNameStr();
        kerString += ',';
        kerString += std::to_string(ids.id);
    }

    void genParams(std::stringstream &kerStream, int id,
                   bool is_linear) const final {
        m_buffer_node->genParams(kerStream, id, is_linear);
        for (int i = 0; i < 4; i++) {
            kerStream << "int shift" << id << "_" << i << ",\n";
        }
    }

    int setArgs(int start_id, bool is_linear,
                std::function<void(int id, const void *ptr, size_t arg_size,
                                   bool is_buffer)>
                    setArg) const {
        int curr_id = m_buffer_node->setArgs(start_id, is_linear, setArg);
        for (int i = 0; i < 4; i++) {
            const int &d = m_shifts[i];
            setArg(curr_id + i, static_cast<const void *>(&d), sizeof(int),
                   false);
        }
        return curr_id + 4;
    }

    void genOffsets(std::stringstream &kerStream, int id,
                    bool is_linear) const final {
        detail::generateShiftNodeOffsets(kerStream, id, is_linear,
                                         getTypeStr());
    }

    void genFuncs(std::stringstream &kerStream,
                  const common::Node_ids &ids) const final {
        detail::generateShiftNodeRead(kerStream, ids.id, getTypeStr());
    }

    void getInfo(unsigned &len, unsigned &buf_count,
                 unsigned &bytes) const final {
        m_buffer_node->getInfo(len, buf_count, bytes);
    }

    std::string getNameStr() const final {
        return std::string("Sh") + getShortName(m_type);
    }
};
}  // namespace common
}  // namespace arrayfire
