#ifndef CAFFE_ROI_DATA_LAYER_HPP_
#define CAFFE_ROI_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_roi_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
class RoIDataLayer : public BasePrefetchingRoIDataLayer<Dtype> {
 public:
  explicit RoIDataLayer(const LayerParameter& param);
  virtual ~RoIDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  // RoIDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "RoIData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 5; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader<RoIDatum> reader_;
  bool has_anno_type_;
  AnnotatedDatum_AnnotationType anno_type_;
  vector<BatchSampler> batch_samplers_;
  string label_map_file_;

  int max_roi_per_im_;
  int num_class_;
  bool visualize_;
  int num_roi_visualize_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
