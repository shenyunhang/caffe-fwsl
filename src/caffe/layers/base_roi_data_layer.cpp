#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_roi_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
BaseRoIDataLayer<Dtype>::BaseRoIDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param), transform_param_(param.transform_param()) {}

template <typename Dtype>
void BaseRoIDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  if (top.size() == 4) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  // TODO(YH): we set output_labels_ false currently
  output_labels_ = false;
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingRoIDataLayer<Dtype>::BasePrefetchingRoIDataLayer(
    const LayerParameter& param)
    : BaseRoIDataLayer<Dtype>(param), prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void BasePrefetchingRoIDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseRoIDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    prefetch_[i].roi_.mutable_cpu_data();
    prefetch_[i].roi_score_.mutable_cpu_data();
    prefetch_[i].label_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].box_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      prefetch_[i].roi_.mutable_gpu_data();
      prefetch_[i].roi_score_.mutable_gpu_data();
      prefetch_[i].label_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].box_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingRoIDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BasePrefetchingRoIDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // LOG(INFO)<<"prefetch_free_: "<<prefetch_free_.size();
  // LOG(INFO)<<"prefetch_full_: "<<prefetch_full_.size();
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());

  // Reshape to loaded roi.
  top[1]->ReshapeLike(batch->roi_);
  // Copy the roi.
  caffe_copy(batch->roi_.count(), batch->roi_.cpu_data(),
             top[1]->mutable_cpu_data());

  // Reshape to loaded roi_score.
  top[2]->ReshapeLike(batch->roi_score_);
  // Copy the roi_score.
  caffe_copy(batch->roi_score_.count(), batch->roi_score_.cpu_data(),
             top[2]->mutable_cpu_data());

  // Reshape to loaded label.
  top[3]->ReshapeLike(batch->label_);
  // Copy the label.
  caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
             top[3]->mutable_cpu_data());

  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded box.
    top[4]->ReshapeLike(batch->box_);
    // Copy the box.
    caffe_copy(batch->box_.count(), batch->box_.cpu_data(),
               top[4]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingRoIDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseRoIDataLayer);
INSTANTIATE_CLASS(BasePrefetchingRoIDataLayer);

}  // namespace caffe
