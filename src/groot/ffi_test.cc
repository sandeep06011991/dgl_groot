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
using namespace dgl::runtime;

namespace dgl{
    namespace groot{

        class ScatteredArrayObject : public Object {
            public:
                IdArray source_array;
                IdArray partition_map;
                IdArray index;
                int num_partitions;
                List<Value>  scattered_arrays;
                
                List<Value> csum;
 
                List<Value> scatter_inverse;
                //  List<Value> list;           // works
                //  *     list.push_back(Value(MakeValue(1)));  // works
                //  *     list.push_back(Value(MakeValue(NDArray::Empty(shape, dtype, ctx))));  //
                List<Value>  scatter_sizes;
                
                ScatteredArrayObject(IdArray source, IdArray partition, int num_partitions){
                    this->source_array = source;
                    this->partition_map = partition; 
                    this->num_partitions = num_partitions;
                    DGLContext ctx = source->ctx;
                    uint8_t nbits = 32;
                    index = aten::NewIdArray(source->shape[0],  ctx, nbits );
                    index = scatter_index(source, partition);
 
                    // DGLContext context{kDGLCPU, 0};
                    // source_array = aten::Range(10, 100, 32, context);

                }

                void VisitAttrs(AttrVisitor *v) final {
                    v->Visit("source_array", &source_array);
                    v->Visit("partition_map", &partition_map);
                    v->Visit("index", &index);
                    v->Visit("num_partitions", &num_partitions);
                    v->Visit("scattered_arrays", &scattered_arrays);
                    v->Visit("scatter_inverse", &scatter_inverse);
                    v->Visit("scatter_sizes", &scatter_sizes);
                }

            static constexpr const char* _type_key = "ScatteredArray";
            DGL_DECLARE_OBJECT_TYPE_INFO(ScatteredArrayObject, Object);
        };

        class ScatteredArray : public ObjectRef {
        public:
            const ScatteredArrayObject* operator->() const {
                return static_cast<const ScatteredArrayObject*>(obj_.get());
            }
            using ContainerType = ScatteredArrayObject;
        };
        void testSocket(){

        }

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
    


     DGL_REGISTER_GLOBAL("groot._CAPI_dummyScatter")
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
