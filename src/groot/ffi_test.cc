#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/parallel_for.h>
#include <dgl/runtime/registry.h>
#include <dgl/packed_func_ext.h>
#include <zmq.hpp>
#include <dgl/array.h>
#include <dgl/runtime/container.h>
#include "../c_api_common.h"
#include <vector>
#include "cuda/array_scatter.h"
#include "array_scatter.h"
#include <vector>

using namespace dgl::runtime;

namespace dgl{
    namespace groot{


      struct TestObject{
        int a = 0;
        int update(){
            a = a + 1;
            return a;
        }
        static TestObject * getObject(){
            static TestObject obj;
            return &obj;
        }
        void compile_check(){
            // initialize the zmq context with a single IO thread
            zmq::context_t context{1};
        }

     };   
    DGL_REGISTER_GLOBAL("groot._CAPI_testffi")
        .set_body([](DGLArgs args, DGLRetValue* rv) {
        *rv = TestObject::getObject()->update();
        });



     DGL_REGISTER_GLOBAL("groot._CAPI_ScatterObjectCreate")
        .set_body([](DGLArgs args, DGLRetValue* rv) {
        IdArray src_array = args[0];
        IdArray partition = args[1];
        int num_partitions = args[2];
        CHECK_EQ(src_array->shape[0], partition->shape[0]);
        auto ob = std::make_shared<ScatteredArrayObject>(src_array, partition, num_partitions);
        *rv = ob;
        });
    }

}
