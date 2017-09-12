#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
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
void Show_batch(const Batch<Dtype>* batch,
                const TransformationParameter transform_param,
                const int num_roi_visualize_) {
  const Dtype* data = batch->data_.cpu_data();
  const Dtype* roi = batch->roi_.cpu_data();
  const Dtype* roi_num = batch->roi_num_.cpu_data();

  const int num = batch->data_.num();
  const int channels = batch->data_.channels();
  const int height = batch->data_.height();
  const int width = batch->data_.width();
  const int num_roi = batch->roi_.num();

  const string save_path = "";
  const string save_path_jet = "";
  vector<float> mean_values;
  for (int c = 0; c < transform_param.mean_value_size(); ++c) {
    mean_values.push_back(transform_param.mean_value(c));
  }

  int index, index_mat;
  int xmin, ymin, xmax, ymax;
  for (int n = 0; n < num; ++n) {
    //-----------------------------------------------------------------------
    cv::Mat img_mat;
    cv::Mat jet_mat;
    if (channels == 3) {
      img_mat = cv::Mat(height, width, CV_8UC3);
    } else if (channels == 1) {
      img_mat = cv::Mat(height, width, CV_8UC1);
    } else {
      LOG(FATAL) << "channels should 1 or 3";
    }

    uchar* img_mat_data = img_mat.data;
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          index = (c * height + h) * width + w;
          index_mat = (h * width + w) * channels + c;
          img_mat_data[index_mat] = data[index] + mean_values[c];
        }
      }
    }

    int num_item_roi = 0;
    for (int r = 0; r < num_roi; r++) {
      if (roi[r * 5 + 0] == n) {
      } else {
        continue;
      }
      num_item_roi++;
      if (num_item_roi > num_roi_visualize_) {
        continue;
      }
      xmin = roi[r * 5 + 1] * width;
      ymin = roi[r * 5 + 2] * height;
      xmax = roi[r * 5 + 3] * width;
      ymax = roi[r * 5 + 4] * height;
      // LOG(INFO) << xmin << " " << ymin << " " << xmax << " " << ymax;

      cv::Rect rec = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
      cv::rectangle(img_mat, rec, cv::Scalar(0, 0, 255), 1);
    }
    // LOG(INFO) << "out_RoI num: " << num_item_roi;
    CHECK_EQ(num_item_roi, roi_num[n]) << "num of roi is not consistent.";

    std::stringstream ss;
    ss << "image_" << n;
    cv::imshow(ss.str(), img_mat);

    // cv::imwrite(save_path, opg_mat);
    // LOG(INFO) << "save_path: " << save_path;

    // cv::applyColorMap(opg_mat, opg_mat_jet, cv::COLORMAP_JET);
    // cv::imwrite(save_path_jet, opg_mat_jet);

    //-----------------------------------------------------------------------

    //-----------------------------------------------------------------------
    // show the distubution of opg_blob data
    // int total[26];
    // for (int i = 0; i < 26; ++i) {
    // total[i] = 0;
    //}
    // for (int e = 0; e < channels * height * width; e++) {
    // int level = int(data[e] / 10);
    // total[level]++;
    //}
    // for (int i = 0; i < 26; ++i) {
    // std::cout << i << ":" << total[i] << " ";
    //}
    // std::cout << std::endl;
    ////-----------------------------------------------------------------------
    data += batch->data_.offset(1);
    roi += num_item_roi * 5;
  }
  cv::waitKey(0);
}

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
  visualize_ = roi_data_param.visualize();
  max_roi_per_im_ = roi_data_param.max_roi_per_im();
  num_roi_visualize_ = 100;

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

  // roi_num
  vector<int> roi_num_shape(1);
  roi_num_shape[0] = batch_size;
  top[3]->Reshape(roi_num_shape);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].roi_num_.Reshape(roi_num_shape);
  }

  // label
  vector<int> label_shape(2, 1);
  num_class_ = roi_data_param.num_class();
  label_shape[0] = batch_size;
  label_shape[1] = num_class_;
  top[4]->Reshape(label_shape);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }

  // box
  has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
  vector<int> box_shape(4, 1);
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
      box_shape[0] = 1;
      box_shape[1] = 1;
      // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
      // cpu_data and gpu_data for consistent prefetch thread. Thus we make
      // sure there is at least one bbox.
      box_shape[2] = std::max(num_bboxes, 1);
      box_shape[3] = 8;
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  } else {
    box_shape[0] = batch_size;
  }
  top[5]->Reshape(box_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].box_.Reshape(box_shape);
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

  vector<int> label_shape(2, 1);
  label_shape[0] = batch_size;
  label_shape[1] = num_class_;
  batch->label_.Reshape(label_shape);
  caffe_set<Dtype>(batch->label_.count(), Dtype(0),
                   batch->label_.mutable_cpu_data());

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();
  Dtype* top_box = NULL;  // suppress warnings about uninitialized variables

  if (!has_anno_type_) {
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
    while (roi_datum.difficult()) {
      reader_.free().push(const_cast<RoIDatum*>(&roi_datum));
      roi_datum = *(reader_.full().pop("Waiting for data"));
    }

    //--------------------------------------------------------
    // Store label
    for (int l = 0; l < roi_datum.label_size(); ++l) {
      int ll = roi_datum.label(l);
      top_label[ll] = Dtype(1);
    }
    top_label += batch->label_.offset(1);

    if (visualize_) {
      cv::Mat img_mat =
          DecodeDatumToCVMat(roi_datum.anno_datum().datum(), true);

      const int img_height = img_mat.rows;
      const int img_width = img_mat.cols;
      float xmin, ymin, xmax, ymax;
      int num_item_roi = 0;
      float score;

      std::stringstream ss(roi_datum.roi().data());
      // TODO(YH): 这里可能会降低精度
      while (ss >> xmin >> ymin >> xmax >> ymax >> score) {
        num_item_roi++;
        if (num_item_roi > num_roi_visualize_) {
          continue;
        }
        xmin *= img_width;
        ymin *= img_height;
        xmax *= img_width;
        ymax *= img_height;
        cv::Rect rec = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
        cv::rectangle(img_mat, rec, cv::Scalar(0, 0, 255), 1);
      }
      LOG(INFO) << "Original RoI num: " << num_item_roi;

      ss.str(std::string());
      ss.clear();
      ss << "origin_" << item_id;
      cv::imshow(ss.str(), img_mat);
    }

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

    if (visualize_) {
      cv::Mat img_mat =
          DecodeDatumToCVMat(expand_datum->anno_datum().datum(), true);

      const int img_height = img_mat.rows;
      const int img_width = img_mat.cols;
      float xmin, ymin, xmax, ymax;
      int num_item_roi = 0;
      float score;

      std::stringstream ss(expand_datum->roi().data());
      // TODO(YH): 这里可能会降低精度
      while (ss >> xmin >> ymin >> xmax >> ymax >> score) {
        num_item_roi++;
        if (num_item_roi > num_roi_visualize_) {
          continue;
        }
        xmin *= img_width;
        ymin *= img_height;
        xmax *= img_width;
        ymax *= img_height;
        cv::Rect rec = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
        cv::rectangle(img_mat, rec, cv::Scalar(0, 0, 255), 1);
      }
      LOG(INFO) << "Expanded RoI num: " << num_item_roi;

      ss.str(std::string());
      ss.clear();
      ss << "expand_" << item_id;
      cv::imshow(ss.str(), img_mat);
    }

    RoIDatum* sampled_datum = NULL;
    bool has_sampled = false;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      if (visualize_) {
        LOG(INFO) << "sampled_bboxes size: " << sampled_bboxes.size();
        for (int i = 0; i < sampled_bboxes.size(); i++) {
          LOG(INFO) << "sampled_bboxes: " << sampled_bboxes[i].xmin() << " "
                    << sampled_bboxes[i].ymin() << " "
                    << sampled_bboxes[i].xmax() << " "
                    << sampled_bboxes[i].ymax() << " ";
        }
      }
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new RoIDatum();
        this->data_transformer_->CropImage(
            *expand_datum, sampled_bboxes[rand_idx], sampled_datum);
        has_sampled = true;
        if (visualize_) {
          LOG(INFO) << "Chosen sampled_bboxes: "
                    << sampled_bboxes[rand_idx].xmin() << " "
                    << sampled_bboxes[rand_idx].ymin() << " "
                    << sampled_bboxes[rand_idx].xmax() << " "
                    << sampled_bboxes[rand_idx].ymax() << " ";
        }
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
    string transformed_roi_str = "";
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

    if (visualize_) {
      int num_item_roi = 0;
      float xmin, ymin, xmax, ymax;
      float score;
      std::stringstream ss(transformed_roi_str);
      // TODO(YH): 这里可能会降低精度
      while (ss >> xmin >> ymin >> xmax >> ymax >> score) {
        num_item_roi++;
      }
      LOG(INFO) << "Transformed RoI num: " << num_item_roi;
    }
  }

  //--------------------------------------------------------
  // Store RoI, RoI_score and RoI_num
  int num_roi = 0;
  float xmin, ymin, xmax, ymax;
  float score;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    int num_item_roi = 0;
    std::stringstream ss(all_roi[item_id]);
    // TODO(YH): 这里可能会降低精度
    while (ss >> xmin >> ymin >> xmax >> ymax >> score) {
      num_roi++;
      num_item_roi++;
      if (num_item_roi > max_roi_per_im_) {
        break;
      }
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

  vector<int> roi_num_shape(1);
  roi_num_shape[0] = batch_size;
  batch->roi_num_.Reshape(roi_num_shape);
  Dtype* top_roi_num = batch->roi_num_.mutable_cpu_data();

  CHECK_EQ(batch_size, all_roi.size())
      << "batch_size should equal number of all_anno.";
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    int num_item_roi = 0;
    std::stringstream ss(all_roi[item_id]);
    // TODO(YH): 这里可能会降低精度
    while (ss >> xmin >> ymin >> xmax >> ymax >> score) {
      // LOG(INFO)<< xmin << " " << ymin << " " << xmax
      // << " " << ymax << " " << score;
      top_roi[0] = item_id;
      top_roi[1] = xmin;
      top_roi[2] = ymin;
      top_roi[3] = xmax;
      top_roi[4] = ymax;
      top_roi += batch->roi_.offset(1);

      top_roi_score[0] = score;
      top_roi_score += batch->roi_score_.offset(1);

      num_item_roi++;
      if (num_item_roi > max_roi_per_im_) {
        break;
      }
    }
    top_roi_num[item_id] = num_item_roi;
    // LOG(INFO)<< "item_id: "<<item_id<<" num_item_roi: "<<num_item_roi;
  }

  //--------------------------------------------------------
  // Store "rich" annotation if needed.
  if (has_anno_type_) {
    vector<int> box_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      box_shape[0] = 1;
      box_shape[1] = 1;
      box_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the box.
        box_shape[2] = 1;
        batch->box_.Reshape(box_shape);
        caffe_set<Dtype>(8, -1, batch->box_.mutable_cpu_data());
      } else {
        // Reshape the box and store the annotation.
        box_shape[2] = num_bboxes;
        batch->box_.Reshape(box_shape);
        top_box = batch->box_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_box[idx++] = item_id;
              top_box[idx++] = anno_group.group_label();
              top_box[idx++] = anno.instance_id();
              top_box[idx++] = bbox.xmin();
              top_box[idx++] = bbox.ymin();
              top_box[idx++] = bbox.xmax();
              top_box[idx++] = bbox.ymax();
              top_box[idx++] = bbox.difficult();
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

  if (visualize_) {
    Show_batch(batch, transform_param, num_roi_visualize_);
  }
}

INSTANTIATE_CLASS(RoIDataLayer);
REGISTER_LAYER_CLASS(RoIData);

}  // namespace caffe
