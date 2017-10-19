#ifndef CAFFE_PSEUDO_LABEL_LAYER_HPP_
#define CAFFE_PSEUDO_LABEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class PseudoLabelLayer : public Layer<Dtype> {
 public:
  explicit PseudoLabelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PseudoLabel"; }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  void Show_blob(const Dtype* data, const int channels, const int height,
                 const int width, const string save_path,
                 const string save_path_jet, const float threshold_ratio,
                 const int fill = 0);
  void vis_roi(const Blob<Dtype>* roi_blob, const Blob<Dtype>* roi_score_blob,
               const Blob<Dtype>* img_blob, Blob<Dtype>& score_map,
               int save_id_);
  void vis_roi_score(const Blob<Dtype>* roi_score_blob, Blob<Dtype>& score_map,
                     int save_id_);

  void top0forward(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top);
  void top1forward(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top);

  int num_img_;
  int num_cls_;
  int num_roi_;

  int reserve_size_;
  int max_per_cls_;

  int num_det_;
  Blob<Dtype> roi_det_;
  Blob<Dtype> det_cls_;

  Blob<Dtype> score_map_;
  int save_id_;
};

}  // namespace caffe

#endif  // CAFFE_PSEUDO_LABEL_LAYER_HPP_
