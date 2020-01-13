# Graph-Neural-Networks-for-Image-Understanding-Based-on-Multiple-Cues
Code for Paper Graph Neural Networks for Image Understanding Based on Multiple Cues: Group Emotion Recognition and Event Recognition as Use Cases


# System Requirements:
ubuntu 16.04
Caffe: https://github.com/BVLC/caffe (with cuda installed)
Caffe-Senet (https://github.com/hujie-frank/SENet)
Python 2.7.12
Tensorflow 1.0.1
scikit-learn
numpy


# Database:
We use GroupEmoW in this demo code. GroupEmoW is a new group-level Emotion Dataset in the wild. Images in GroupEmoW are labeled into negative, neutral and positive categories. Data statistics is shown in the following table. (Please email guoxin@udel.edu to download the database, and put it into the ./data directory.)

| Partition | Number of Negative Samples | Number of Neutral Samples | Number of Positive Samples |
| ------------- | ------------- |------------- |------------- |
| Train | 3019  | 3463 | 4645 |
| Val | 861  | 990 | 1327 |
| Test | 431  | 494 | 664 |


# Preprocessing:
## Face patches. 
Faces are extracted and aligned using MTCNN, readers can follow MTCNN link (https://github.com/kpzhang93/MTCNN_face_detection_alignment) to extract faces of their own database.

## Extract vgg face features from face patches. 
The vgg model deploy file and a pre-trained vgg face model for group level emotion recognition are saved in folder pretrained_models. 

Under the main directory, simply run:

'''python
python extract_features.py --model ./pretrained_models/GroupVEmo_face_model.caffemodel --model_protocal ./pretrained_models/vgg_group_deploy.prototxt --data_dir ./data/faces_MTCNN --feature_save_dir ./data/faces_MTCNN_vgg_features --feature_layer_name fc7 --caffe_standard_flag True
'''

to extract face vgg features. Then the object features will be extracted and saved in ./data/faces_MTCNN_vgg_features. 

## Object patches.
Object patches of each image are extracted using a bottom-up attention model, readers can follow the link (https://github.com/peteanderson80/bottom-up-attention) to extract object patches of their own database. 

## Extract object features from object patches. 
The SENet-154 model trained on the ImageNet-1K database is employed to extract a 2048-dimensional feature representation for each salient object by using the output of layer
pool5/7x7 s1. Please refer to https://github.com/hujie-frank/SENet to install SEnet and download the SENet-154 model, save it as SENet-154.caffemodel in folder pretrained_models. 
In the main directory, simply run:

'''python
python extract_features.py --model ./pretrained_models/SENet.caffemodel --model_protocal ./pretrained_models/SENet-154-deploy.prototxt --data_dir ./data/objects --feature_save_dir ./data/object_senet_features --feature_layer_name pool5/7x7_s1 --caffe_standard_flag False
'''
to extract object features. And the object features will be extracted and saved in ./data/object_senet_features

## Extract scene features from whole image. 
A Inception-V2 model pre-trained on group-level emotion recognition is employed to extract scene features from whole images. The pretrained model and deploy file are saved in folder pretrained_models.
In the main directory, simply run:
'''python
python extract_features.py --model ./pretrained_models/group_inception_scene.caffemodel --model_protocal ./pretrained_models/group_inception_scene_deploy.prototxt --data_dir ./data/Original --feature_save_dir ./data/scene_features --feature_layer_name global_pool --caffe_standard_flag True
'''            
Then the scene features will be extracted and saved in ./data/scene_features.

For convience, we provide the following items together with the GroupEmoW dataset. 
1. Extracted and aligned faces of each image using the MTCNN method.
2. Features of each face extracted using a vgg-face model.
3. Objects of each image detected using a bottom-up attention model.
4. Features of each object extracted using a Senet-154 model.
5. Scene features of each image extracted using an Inception-V2 model. 


# Train GNN
'''python
python main.py --train_or_test train --org_path ./data/Original/Train --face_feature_path ./data/faces_MTCNN_vgg_features/Train --object_feature_path ./data/object_senet_features/Train --scene_feature_path ./data/scene_features/Train --model_path ./models/GNN_model
'''

# Test GNN on validation and test data. 
'''python
python main.py --train_or_test test --org_path ./data/Original/Val --face_feature_path ./data/faces_MTCNN_vgg_features/Val --object_feature_path ./data/object_senet_features/Val --scene_feature_path ./data/scene_features/Val --model_path ./models/GNN_model
'''
'''python
python main.py --train_or_test test --org_path ./data/Original/Test --face_feature_path ./data/faces_MTCNN_vgg_features/Test --object_feature_path ./data/object_senet_features/Test --scene_feature_path ./data/scene_features/Test --model_path ./models/GNN_model
'''

# Cite
If you have used any of the data or code, please cite the following paper: 

@INPROCEEDINGS{guo2020graph,
    title={Graph Neural Networks for Image Understanding Based on Multiple Cues: Group Emotion Recognition and Event Recognition as Use Cases},
    author={Xin Guo and Luisa F. Polania and Bin Zhu and Charles Boncelet and Kenneth E. Barner},
    year={2020},
    booktitle={2020 IEEE Winter Conference on Applications of Computer Vision (WACV)},
}
