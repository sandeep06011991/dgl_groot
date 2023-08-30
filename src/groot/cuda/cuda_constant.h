//
// Created by juelin on 8/15/23.
//

#ifndef DGL_CUDA_CONSTANT_H
#define DGL_CUDA_CONSTANT_H
namespace dgl {
    namespace groot {
        namespace Constant {
            const inline int kEmptyKey = -1, kBufferSize = 256, kKilobytes = 1024, kMegabytes =
                    1024 * 1024, kGigabytes = 1024 * 1024 * 1024,
                    kCudaWarpSize = 32, kCudaBlockSize = 256, kCudaTileSize = 1024;
        };
    }  // namespace groot
}  // namespace dgl
#endif  // DGL_CUDA_CONSTANT_H
