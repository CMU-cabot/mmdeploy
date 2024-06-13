#include "mmdeploy/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

namespace mmdeploy::operation::cpu {

class ThresholdMaskImpl : public ThresholdMask {
 public:
  Result<void> apply(const Tensor& src, Tensor& dst, float threshold) override {
    cv::Mat mat = mmdeploy::cpu::Tensor2CVMat(src);
    cv::Mat dst_mat = mat > threshold;
    dst = mmdeploy::cpu::CVMat2Tensor(dst_mat);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ThresholdMask, (cpu, 0), []() { return std::make_unique<ThresholdMaskImpl>(); });

}  // namespace mmdeploy::operation::cpu
