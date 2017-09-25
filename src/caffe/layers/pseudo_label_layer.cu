#include <cfloat>
#include <vector>

#include "caffe/layers/pseudo_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Get_roi_det_blob(const int nthreads, const Dtype* roi_data,
                                 const Dtype* det_data, const int num_roi,
                                 const int num_det, Dtype* roi_det_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int roi_idx = index / num_det;
    const int det_idx = index % num_det;

    // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    const Dtype* roi = roi_data + roi_idx * 5;
    //[image_id, label, confidence, xmin, ymin, xmax, ymax]
    const Dtype* det = det_data + det_idx * 7;

    if (roi[0] != det[0]) {
      roi_det_data[roi_idx * num_det + roi_idx] = 0;
      continue;
    }

    roi += 1;
    det += 3;
    if (roi[0] > det[2] || roi[2] < det[0] || roi[1] > det[3] ||
        roi[3] < det[1]) {
      roi_det_data[roi_idx * num_det + roi_idx] = 0;
      continue;
    }

    const Dtype intersect_x1 = max(roi[0], det[0]);
    const Dtype intersect_y1 = max(roi[1], det[1]);
    const Dtype intersect_x2 = min(roi[2], det[2]);
    const Dtype intersect_y2 = min(roi[3], det[3]);

    const Dtype intersect_h = intersect_y2 - intersect_y1;
    const Dtype intersect_w = intersect_x2 - intersect_x1;
    if (intersect_h == 0 && intersect_w == 0) {
      roi_det_data[roi_idx * num_det + roi_idx] = 0;
      continue;
    }
    const Dtype intersect_size = intersect_h * intersect_w;
    const Dtype roi_size = (roi[2] - roi[0]) * (roi[3] - roi[1]);
    const Dtype det_size = (det[2] - det[0]) * (det[3] - det[1]);

    roi_det_data[roi_idx * num_det + roi_idx] =
        intersect_size / (roi_size + det_size - intersect_size);
  }
}

template <typename Dtype>
void PseudoLabelLayer<Dtype>::top1forward(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  Get_roi_det_blob<
      Dtype><<<CAFFE_GET_BLOCKS(num_roi_ * num_det_), CAFFE_CUDA_NUM_THREADS>>>(
      num_roi_ * num_det_, bottom[1]->gpu_data(), bottom[3]->gpu_data(),
      num_roi_, num_det_, roi_det_.mutable_gpu_data());

  caffe_gpu_set(det_cls_.count(), Dtype(0.), det_cls_.mutable_gpu_data());
  Dtype* det_cls_data = det_cls_.mutable_cpu_data();
  //[image_id, label, confidence, xmin, ymin, xmax, ymax]
  const Dtype* det_data = bottom[3]->cpu_data();
  for (int i = 0; i < num_det_; ++i) {
    const Dtype* det = det_data + i * 7;
    const int label = det[1];
    CHECK_LT(label, 0) << "found background label in detection result.";
    det_cls_data[i * num_cls_ + label - 1] = det[2];
  }

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_roi_, num_cls_,
                        num_det_, Dtype(1.), roi_det_.gpu_data(),
                        det_cls_.gpu_data(), Dtype(0.),
                        top[1]->mutable_gpu_data());
}

template <typename Dtype>
void PseudoLabelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  top0forward(bottom, top);
  if (bottom.size() == 4) {
    top1forward(bottom, top);
  }
}

template <typename Dtype>
void PseudoLabelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  for (size_t i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), static_cast<Dtype>(0),
                bottom[i]->mutable_cpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PseudoLabelLayer);

}  // namespace caffe
