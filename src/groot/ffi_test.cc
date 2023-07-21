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
// Todo difference between IDArray and NDArray 
// Is it  the same thing. ?

namespace dgl{
    namespace groot{
        class ScatteredArrayObject : public Object {
            public:
                IdArray source_array;
                IdArray partition_map;
                IdArray index;
                IdArray sizes;
                int num_partitions;
                List<Value>  scattered_arrays;

                List<Value> csum;

                List<Value> scatter_inverse;
                //  List<Value> list;           // works
                //  *     list.push_back(Value(MakeValue(1)));  // works
                //  *     list.push_back(Value(MakeValue(NDArray::Empty(shape, dtype, ctx))));  //
                List<Value>  scatter_sizes;
                // TODO:
                //  FFI does not need to have all this logic
                ScatteredArrayObject(IdArray source, IdArray partition, int num_partitions){
                    this->source_array = source;
                    this->partition_map = partition; 
                    this->num_partitions = num_partitions;
                    DGLContext ctx = source->ctx;
                    CHECK_EQ(source->ctx , partition->ctx);
                    CHECK_EQ(source->ctx.device_type, DGLDeviceType::kDGLCUDA);
                    index = impl::scatter_index(partition, num_partitions);
                    sizes = getBoundaryOffsets(index, num_partitions);
                    std::vector<IdArray> partitioned;
                    std::vector<IdArray> indexes;
                    for(int64_t i = 0; i < num_partitions; i ++ ){
                        IdArray index_out = aten::NewIdArray(aten::IndexSelect<int>(sizes, i), \
                        partition->ctx, 32);
                        IdArray partitioned_out = aten::NewIdArray(aten::IndexSelect<int>(sizes, i), \
                        partition->ctx, 32);
                        partitioned.push_back(partitioned_out);
                        indexes.push_back(index_out);
                    }
                }

                void VisitAttrs(AttrVisitor *v) final {
                    v->Visit("source_array", &source_array);
                    v->Visit("partition_map", &partition_map);
                    v->Visit("sizes", &sizes);
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
