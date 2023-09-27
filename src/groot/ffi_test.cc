#include "../c_api_common.h"
#include "array_scatter.h"
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/parallel_for.h>
#include <dgl/runtime/registry.h>
#include <vector>
#include <zmq.hpp>

using namespace dgl::runtime;

namespace dgl {
namespace groot {

struct TestObject {
  int a = 0;
  int update() {
    a = a + 1;
    return a;
  }
  static TestObject *getObject() {
    static TestObject obj;
    return &obj;
  }
  void compile_check() {
    // initialize the zmq context with a single IO thread
    zmq::context_t context{1};
  }
};
DGL_REGISTER_GLOBAL("groot._CAPI_testffi")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      *rv = TestObject::getObject()->update();
    });

} // namespace groot

} // namespace dgl
