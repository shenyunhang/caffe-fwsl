#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/roi_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

template <typename Dtype>
RoIDataLayer<Dtype>::RoIDataLayer(const LayerParameter& param)
    : BasePrefetchingRoIDataLayer<Dtype>(param), reader_(param) {}

template <typename Dtype>
RoIDataLayer<Dtype>::~RoIDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void RoIDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const RoIDataParameter& roi_data_param = this->layer_param_.roi_data_param();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  label_map_file_ = anno_data_param.label_map_file();
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
      this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
          << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }

  // Read a data point, and use it to initialize the top blob.
  RoIDatum& roi_datum = *(reader_.full().peek());
  const AnnotatedDatum& anno_datum = roi_datum.anno_datum();

  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
            << top[0]->channels() << "," << top[0]->height() << ","
            << top[0]->width();

  // roi
  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  vector<int> roi_shape(2, 1);
  max_roi_per_im_ = roi_data_param.max_roi_per_im();
  roi_shape[0] = max_roi_per_im_ * batch_size;
  roi_shape[1] = 5;
  top[1]->Reshape(roi_shape);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].roi_.Reshape(roi_shape);
  }

  // roi_score
  vector<int> roi_score_shape(1);
  roi_score_shape[0] = max_roi_per_im_ * batch_size;
  top[2]->Reshape(roi_score_shape);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].roi_score_.Reshape(roi_score_shape);
  }

  // label
  vector<int> label_shape(2, 1);
  num_class_ = roi_data_param.num_class();
  label_shape[0] = batch_size;
  label_shape[1] = num_class_;
  top[3]->Reshape(label_shape);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }

  // label
  // TODO(YH): add box annotation?
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[4]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template <typename Dtype>
void RoIDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
      this->layer_param_.transform_param();
  RoIDatum& roi_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(roi_datum.anno_datum().datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_box = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_ && !has_anno_type_) {
    top_box = batch->box_.mutable_cpu_data();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  map<int, string> all_roi;
  int num_bboxes = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a roi_datum
    RoIDatum& roi_datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    RoIDatum distort_datum;
    RoIDatum* expand_datum = NULL;
    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(roi_datum);
      this->data_transformer_->DistortImage(
          roi_datum.anno_datum().datum(),
          distort_datum.mutable_anno_datum()->mutable_datum());
      if (transform_param.has_expand_param()) {
        expand_datum = new RoIDatum();
        this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } else {
        expand_datum = &distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
        expand_datum = new RoIDatum();
        this->data_transformer_->ExpandImage(roi_datum, expand_datum);
      } else {
        expand_datum = &roi_datum;
      }
    }
    RoIDatum* sampled_datum = NULL;
    bool has_sampled = false;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new RoIDatum();
        this->data_transformer_->CropImage(
            *expand_datum, sampled_bboxes[rand_idx], sampled_datum);
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);
    timer.Start();
    vector<int> shape = this->data_transformer_->InferBlobShape(
        sampled_datum->anno_datum().datum());
    if (transform_param.has_resize_param()) {
      if (transform_param.resize_param().resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        this->transformed_data_.Reshape(shape);
        batch->data_.Reshape(shape);
        top_data = batch->data_.mutable_cpu_data();
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                         shape.begin() + 1));
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                       shape.begin() + 1));
    }
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;
    string transformed_roi_str;
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->anno_datum().has_type())
            << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum->mutable_anno_datum()->set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum->anno_datum().type())
              << "Different AnnotationType.";
        }
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
        this->data_transformer_->Transform(
            *sampled_datum, &(this->transformed_data_), &transformed_roi_str,
            &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } else {
        this->data_transformer_->Transform(
            *sampled_datum, &(this->transformed_data_), &transformed_roi_str);
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->anno_datum().datum().has_label())
            << "Cannot find any label.";
        top_box[item_id] = sampled_datum->anno_datum().datum().label();
      }
    } else {
      this->data_transformer_->Transform(
          *sampled_datum, &(this->transformed_data_), &transformed_roi_str);
    }
    all_roi[item_id] = transformed_roi_str;
    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<RoIDatum*>(&roi_datum));
  }

  //--------------------------------------------------------
  // Store RoI and RoI_score
  int num_roi = 0;
  float xmin, ymin, xmax, ymax;
  float score;
  for (size_t item_id = 0; item_id < all_roi.size(); ++item_id) {
    std::stringstream ss(all_roi[item_id]);
    // TODO(YH): 这里可能会降低精度
    while (ss >> xmin >> ymin >> xmax >> ymax >> score) {
      num_roi++;
    }
  }

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  vector<int> roi_shape(2, 1);
  roi_shape[0] = num_roi;
  roi_shape[1] = 5;
  batch->roi_.Reshape(roi_shape);
  Dtype* top_roi = batch->roi_.mutable_cpu_data();

  vector<int> roi_score_shape(1);
  roi_score_shape[0] = num_roi;
  batch->roi_score_.Reshape(roi_score_shape);
  Dtype* top_roi_score = batch->roi_score_.mutable_cpu_data();

  for (size_t item_id = 0; item_id < all_roi.size(); ++item_id) {
    std::stringstream ss(all_roi[item_id]);
    // TODO(YH): 这里可能会降低精度
    while (ss >> xmin >> ymin >> xmax >> ymax >> score) {
    //LOG_IF(INFO, Caffe::root_solver())<< xmin << " " << ymin << " " << xmax << " " << ymax << " " << score;
      num_roi++;
      top_roi[0] = item_id;
      top_roi[1] = xmin;
      top_roi[2] = ymin;
      top_roi[3] = xmax;
      top_roi[4] = ymax;
      top_roi += batch->roi_.offset(1);

      top_roi_score[0] = score;
      top_roi_score += batch->roi_score_.offset(1);
    }
  }

  //--------------------------------------------------------
  // Store label
  vector<int> label_shape(2, 1);
  label_shape[0] = batch_size;
  label_shape[1] = num_class_;
  batch->label_.Reshape(label_shape);
  caffe_set<Dtype>(batch->label_.count(), 0, batch->label_.mutable_cpu_data());
  Dtype* top_label = batch->label_.mutable_cpu_data();
  for (size_t item_id = 0; item_id < all_roi.size(); ++item_id) {
    for (int l = 0; l < roi_datum.label_size(); ++l) {
      int ll = roi_datum.label(l);
      top_label[ll] = 1;
    }
    top_label += batch->label_.offset(1);
  }

  //--------------------------------------------------------
  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_label[idx++] = item_id;
              top_label[idx++] = anno_group.group_label();
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              top_label[idx++] = bbox.difficult();
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(RoIDataLayer);
REGISTER_LAYER_CLASS(RoIData);

}  // namespace caffe
