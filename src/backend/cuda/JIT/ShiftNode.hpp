/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include "BufferNode.hpp"
#include "Node.hpp"
#include <iomanip>
#include <mutex>

namespace cuda
{

namespace JIT
{
    template<typename T>
    class ShiftNode : public Node
    {
    private:

        Node_ptr m_buffer_node;
        const std::array<int, 4> m_shifts;

    public:

        ShiftNode(const char *type_str,
                  const char *name_str,
                  Node_ptr buffer_node,
                  const std::array<int, 4> shifts)
            : Node(type_str, name_str, 0, {}),
              m_buffer_node(buffer_node),
              m_shifts(shifts)
        {
        }

        void setData(Param<T> param, std::shared_ptr<T> data, const unsigned bytes, bool is_linear)
        {
            auto node_ptr = m_buffer_node.get();
            dynamic_cast<BufferNode<T> *>(node_ptr)->setData(param, data, bytes, is_linear);
        }

        bool isLinear(dim_t dims[4]) const final
        {
            return false;
        }

        void genKerName(std::stringstream &kerStream, Node_ids ids) const final
        {
            kerStream << "_" << m_name_str;
            kerStream << std::setw(3) << std::setfill('0') << std::dec << ids.id << std::dec;
        }

        void genParams(std::stringstream &kerStream, int id, bool is_linear) const final
        {
            auto node_ptr = m_buffer_node.get();
            dynamic_cast<BufferNode<T> *>(node_ptr)->genParams(kerStream, id, is_linear);
            for (int i = 0; i < 4; i++) {
                kerStream << "int shift" << id << "_" << i << ",\n";
            }
        }

        void setArgs(std::vector<void *> &args, bool is_linear) const final
        {
            auto node_ptr = m_buffer_node.get();
            dynamic_cast<BufferNode<T> *>(node_ptr)->setArgs(args, is_linear);
            for (int i = 0; i < 4; i++) {
                const int &d = m_shifts[i];
                args.push_back((void *)&d);
            }
        }

        void genOffsets(std::stringstream &kerStream, int id, bool is_linear) const final
        {
            std::string idx_str = std::string("idx") + std::to_string(id);
            std::string info_str = std::string("in") + std::to_string(id);
            std::string id_str = std::string("sh_id_") + std::to_string(id) + "_";
            std::string shift_str = std::string("shift") + std::to_string(id) + "_";

            for (int i = 0; i < 4; i++) {
                kerStream << "int " << id_str << i
                          << " = __circular_mod(id" << i
                          << " + " << shift_str << i
                          << ", " << info_str << ".dims[" << i << "]"
                          << ");\n";
            }

            kerStream << "int " << idx_str << " = "
                      << "(" << id_str << "3 < " << info_str << ".dims[3]) * "
                      << info_str << ".strides[3] * " << id_str << "3;\n";
            kerStream << idx_str  << " += "
                      << "(" << id_str << "2 < " << info_str << ".dims[2]) * "
                      << info_str << ".strides[2] * " << id_str << "2;\n";
            kerStream << idx_str  << " += "
                      << "(" << id_str << "1 < " << info_str << ".dims[1]) * "
                      << info_str << ".strides[1] * " << id_str << "1;\n";
            kerStream << idx_str  << " += "
                      << "(" << id_str << "0 < " << info_str << ".dims[0]) * "
                      << id_str << "0;"
                      << "\n";
            kerStream << m_type_str << " *in" << id << "_ptr = in" << id << ".ptr;\n";
        }

        void genFuncs(std::stringstream &kerStream, Node_ids ids) const final
        {
            kerStream << m_type_str << " val" << ids.id << " = "
                      << "in" << ids.id << "_ptr[idx" << ids.id << "];"
                      << "\n";
        }

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes) const final
        {
            auto node_ptr = m_buffer_node.get();
            dynamic_cast<BufferNode<T> *>(node_ptr)->getInfo(len, buf_count, bytes);
        }
    };
}

}
