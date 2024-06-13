#include <cstdint>

namespace mmdeploy {
namespace operation {
namespace cuda {
namespace impl {

template <typename From>
__global__ void _ThresholdMask(const From* src, uint8_t* dst, size_t n, float threshold) {
    auto idx = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
      dst[i] = (src[i] > threshold) ? 255 : 0;
    }
}

template <typename From>
void ThresholdMask(const From* src, uint8_t* dst, size_t n, float threshold, cudaStream_t stream) {
  size_t n_threads = 256;
  size_t n_blocks = (n + n_threads - 1) / n_threads;
  _ThresholdMask<<<n_blocks, n_threads, 0, stream>>>(src, dst, n, threshold);
}

template void ThresholdMask<float>(const float* src, uint8_t* dst, size_t n, float threshold, cudaStream_t stream);

}  // namespace impl
}  // namespace cuda
}  // namespace operation
}  // namespace mmdeploy
