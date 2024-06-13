#include <cuda_runtime.h>

#include "mmdeploy/operation/vision.h"

namespace mmdeploy::operation::cuda {

namespace impl {
template <typename T>
void ThresholdMask(const T* src, uint8_t* dst, size_t n, float threshold, cudaStream_t stream);
}

class ThresholdMaskImpl : public ThresholdMask {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst, float threshold) override {
    auto src_desc = src.desc();
    if (DataType::kFLOAT == src_desc.data_type) {
        TensorDesc dst_desc{device(), DataType::kINT8, src_desc.shape, src_desc.name};
        Tensor dst_tensor{dst_desc};
        auto output = dst_tensor.data<uint8_t>();
        auto cuda_stream = GetNative<cudaStream_t>(stream());

        impl::ThresholdMask<float>(src.data<float>(), output, src.size(), threshold, cuda_stream);

        dst = std::move(dst_tensor);
        return success();
    }
    throw_exception(eNotSupported);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ThresholdMask, (cuda, 0), [] { return std::make_unique<ThresholdMaskImpl>(); });

}  // namespace mmdeploy::operation::cuda
