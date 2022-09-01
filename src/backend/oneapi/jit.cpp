/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/compile_module.hpp>
#include <common/dispatch.hpp>
#include <common/jit/ModdimNode.hpp>
#include <common/jit/Node.hpp>
#include <common/jit/NodeIterator.hpp>
#include <common/kernel_cache.hpp>
#include <common/util.hpp>
#include <copy.hpp>
#include <device_manager.hpp>
#include <err_oneapi.hpp>
#include <af/dim4.hpp>

//#include <jit/BufferNode.hpp>

#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using common::getFuncName;
using common::Node;
using common::Node_ids;
using common::Node_map_t;

using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

namespace oneapi {

string getKernelString(const string &funcName, const vector<Node *> &full_nodes,
                       const vector<Node_ids> &full_ids,
                       const vector<int> &output_ids, bool is_linear) {
    ONEAPI_NOT_SUPPORTED("");
    return "";
}

/*
cl::Kernel getKernel(const vector<Node *> &output_nodes,
                     const vector<int> &output_ids,
                     const vector<Node *> &full_nodes,
                     const vector<Node_ids> &full_ids, const bool is_linear) {
    ONEAPI_NOT_SUPPORTED("");
    return common::getKernel("", "", true).get();
}
*/

/*
void evalNodes(vector<Param> &outputs, const vector<Node *> &output_nodes) {
    ONEAPI_NOT_SUPPORTED("");
}

void evalNodes(Param &out, Node *node) {
    ONEAPI_NOT_SUPPORTED("");
}
*/

}  // namespace oneapi
