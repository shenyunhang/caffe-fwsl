#ifndef CAFFE_CPG_LAYER_HPP_
#define CAFFE_CPG_LAYER_HPP_

#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class CPGLayer : public Layer<Dtype> {
 public:
  explicit CPGLayer(const LayerParameter& param)
      : Layer<Dtype>(param), raw_cpg_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CPG"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  void Set_Net(Net<Dtype>* net) {
    LOG(INFO) << "Setting net in CPGLayer";
    net_ = net;
  }

  vector<int> get_gt_class() { return gt_class_; }

  void CPG_back();

  // DEPRECATED. As CPG_back function will not change the diff of param now.
  void Save_param_diff();
  // DEPRECATED. As CPG_back function will not change the diff of param now.
  void Restore_param_diff();

  void Show_info();
  void BackwardDebugInfo(const int layer_id);
  void Clear_split_diff();
  void Get_split_top_blob();
  void Show_cpg(const Dtype* cpg_data, const int cur, const string info = "");
  void Show_im(const Dtype* im_data, const int cur);
  bool Need_Repartition(const int cls_id, const Dtype label, const Dtype score);
  bool Need_Order(const int cls_id, const Dtype label, const Dtype score);
  void After();

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

  Net<Dtype>* net_;

  bool is_cpg_;
  bool debug_info_;
  bool is_contrast_;
  bool is_order_;
  float predict_threshold_;
  float predict_order_;

  vector<string> cpg_blob_name_;
  string predict_blob_name_;
  vector<int> cpg_blob_index_;
  int predict_blob_index_;

  string start_layer_name_;
  string end_layer_name_;
  int start_layer_index_;
  int end_layer_index_;

  shared_ptr<Blob<Dtype> > im_blob_;
  shared_ptr<Blob<Dtype> > predict_blob_;
  vector<shared_ptr<Blob<Dtype> > > cpg_blob_;

  int bottom_label_index_;

  shared_ptr<Blob<Dtype> > raw_cpg_;

  // abandon now
  vector<shared_ptr<Blob<Dtype> > > history_params_;
  vector<shared_ptr<Blob<Dtype> > > history_blobs_;
  bool is_history_init_;

  vector<Blob<Dtype>*> split_top_blob_;
  vector<vector<bool> > bottom_need_backward_;
  vector<vector<bool> > origin_param_propagate_down_;

  int num_im_;
  int channels_im_;
  int height_im_;
  int width_im_;
  int num_class_;
  int channels_cpg_;
  int size_cpg_;

  vector<int> bp_class_;
  vector<int> gt_class_;

  int accum_im_;
  int accum_bp_;
  int accum_gt_;

  int display_;
  int pass_im_;
  int max_num_im_cpg_;

  int ignore_label_;

  int save_id_;
  bool is_show_;

  vector<string> voc_label_;
};

}  // namespace caffe

#endif  // CAFFE_CPG_LAYER_HPP_
