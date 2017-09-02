#include <vector>

#include "caffe/layers/base_roi_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingRoIDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // LOG(INFO)<<"prefetch_free_: "<<prefetch_free_.size();
  // LOG(INFO)<<"prefetch_full_: "<<prefetch_full_.size();
  if (this->phase_ == TEST && count_img % 100 == 0) {
    LOG(INFO) << "Processed images: " << count_img;
  }
  count_img++;
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data.
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
             top[0]->mutable_gpu_data());

  // Reshape to loaded roi.
  top[1]->ReshapeLike(batch->roi_);
  // Copy the roi.
  caffe_copy(batch->roi_.count(), batch->roi_.gpu_data(),
             top[1]->mutable_gpu_data());

  // Reshape to loaded roi_score.
  top[2]->ReshapeLike(batch->roi_score_);
  // Copy the roi_score.
  caffe_copy(batch->roi_score_.count(), batch->roi_score_.gpu_data(),
             top[2]->mutable_gpu_data());

  // Reshape to loaded roi_num.
  top[3]->ReshapeLike(batch->roi_num_);
  // Copy the roi_num.
  caffe_copy(batch->roi_num_.count(), batch->roi_num_.gpu_data(),
             top[3]->mutable_gpu_data());

  // Reshape to loaded label.
  top[4]->ReshapeLike(batch->label_);
  // Copy the label.
  caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
             top[4]->mutable_gpu_data());

  // Reshape to loaded box.
  top[5]->ReshapeLike(batch->box_);
  // Copy the box.
  caffe_copy(batch->box_.count(), batch->box_.gpu_data(),
             top[5]->mutable_gpu_data());

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingRoIDataLayer);

}  // namespace caffe
