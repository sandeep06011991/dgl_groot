#include "context.h"

namespace dgl {
namespace ds {

typedef dmlc::ThreadLocalStore<DSThreadEntry> DSThreadStore;

DSThreadEntry *DSThreadEntry::ThreadLocal() {
  // TODO: singleton not thread local
  // Thread local singleton requires 2 communication setups
    static DSThreadEntry singleton_object;
    return &singleton_object;
  //  return DSThreadStore::Get();
}

} // namespace ds
} // namespace dgl
