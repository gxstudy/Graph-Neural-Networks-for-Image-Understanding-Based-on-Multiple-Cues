"""
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import numpy as np
import sys
import argparse
import caffe
caffe.set_mode_gpu()


def caffe_path(standard_flag):
    if not standard_flag:
        # put your path of caffe-Senet here
        sys.path.append('/home/xin/caffe-Senet/distribute/python')
        import caffe
        caffe.set_mode_gpu()
        

def initialize_transformer():
    shape = (1, 3, 224, 224)
    transformer = caffe.io.Transformer({'data': shape})
    channel_mean = np.zeros((3,224,224))

    image_mean = [90,100,128]
    for channel_index, mean_val in enumerate(image_mean):
        channel_mean[channel_index, ...] = mean_val
    transformer.set_mean('data', channel_mean)

    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_transpose('data', (2, 0, 1))
    return transformer


def extract_features_patches(args):
    transformer_RGB = initialize_transformer()
    model_net =  caffe.Net(args.model_protocal, args.model, caffe.TEST)
    partitions = ['Train', 'Val', 'Test']    
    classes = ['Negative', 'Neutral', 'Positive']
    
    for partition in partitions:
        for ite, subclass in enumerate(classes):
            class_path = os.path.join(args.data_dir, partition, subclass)
            for image_name in sorted(os.listdir(class_path)):
                save_folder = os.path.join(args.feature_save_dir, partition, subclass, image_name)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
   
                for object_name in sorted(os.listdir(os.path.join(class_path, image_name))):
                    print(object_name)
                    feature_name = object_name.split('.')[-2] +'.npy'
                    save_feature_path = os.path.join(save_folder, feature_name)
                    # Use the following only when previous run got interrupted 
                    # and you don't want to extract again for the ones that have been processed.
                    #if os.path.isfile(save_feature_path):
                    #    print "exist"
                    #    continue

                    object_image = os.path.join(class_path, image_name, object_name)
                    input_im = caffe.io.load_image(object_image)
	            if input_im.shape[0] != 256 or input_im.shape[1] != 256:
	                input_im = caffe.io.resize_image(input_im, (256,256))
                    model_net.blobs['data'].data[...] = transformer_RGB.preprocess('data', input_im)
                    out = model_net.forward()
                    extracted_features = model_net.blobs[args.feature_layer_name].data[0]
                    np.save(save_feature_path, extracted_features)

    del model_net


def extract_features_images(args):
    transformer_RGB = initialize_transformer()
    model_net =  caffe.Net(args.model_protocal, args.model, caffe.TEST)
    partitions = ['Train', 'Val', 'Test']    
    classes = ['Negative', 'Neutral', 'Positive']
    
    for partition in partitions:
        for ite, subclass in enumerate(classes):
            image_path = os.path.join(args.data_dir, partition, subclass)
            save_folder = os.path.join(args.feature_save_dir, partition, subclass)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for image_name in sorted(os.listdir(image_path)):                 
                print(image_name)
                feature_name = image_name.split('.')[-2] +'.npy'
                save_feature_path = os.path.join(save_folder, feature_name)
                # Use the following only when previous run got interrupted 
                # and you don't want to extract again for the ones that have been processed.
                #if os.path.isfile(save_feature_path):
                #    print "exist"
                #    continue

                input_im = caffe.io.load_image(os.path.join(image_path, image_name))
	        if (input_im.shape[0] != 256 or input_im.shape[1] != 256):
	            input_im = caffe.io.resize_image(input_im, (256,256))
                model_net.blobs['data'].data[...] = transformer_RGB.preprocess('data', input_im)
                out = model_net.forward()
                extracted_features = model_net.blobs[args.feature_layer_name].data[0]
                np.save(save_feature_path, extracted_features)

    del model_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extrace features')
    parser.add_argument('--model', type=str, default='./face_vgg_feature_extraction/GroupVEmo_face_model.caffemodel', 
                        help='model used to extract features')
    parser.add_argument('--model_protocal', type=str, default='./face_vgg_feature_extraction/vgg_group_deploy.prototxt', 
                        help='model prototxt')
    parser.add_argument('--data_dir', type=str, default='./data/faces_MTCNN', 
                        help='path to the input data')
    # Note that --data_dir should be different from --feature_save_dir, since the subfolder names are the same in extract_features_fuc() function. 
    parser.add_argument('--feature_save_dir', type=str, default='./data/faces_MTCNN_vgg_features', 
                        help='path to the extract features')
    parser.add_argument('--feature_layer_name', type=str, default='fc7', choices={'fc7', 'pool5/7x7_s1', 'global_pool'}, help='the layer to be extracted as features.')
    parser.add_argument('--caffe_standard_flag', type=bool, default=True, help="Use standard Caffe (True) or Caffe-Senet (False)")


    args = parser.parse_args()
    assert args.data_dir != args.feature_save_dir
    caffe_path(args.caffe_standard_flag)
    
    if args.data_dir.split('/')[-1] == 'Original':
        extract_features_images(args)
    else:
        extract_features_patches(args)
