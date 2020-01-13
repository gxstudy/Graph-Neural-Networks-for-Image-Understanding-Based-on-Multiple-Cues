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
import tensorflow as tf
import numpy as np
from GNN_model import GNN
import os
import sys
from random import shuffle
import argparse
from sklearn.metrics import accuracy_score


def read_data(image_path, face_data_path, attention_data_path, scene_data_path):

    scene_list = []
    face_list = []
    attention_list = []
    labels = []    
    classes = ['Negative', 'Neutral', 'Positive']

    for i, subclass in enumerate(classes):
        image_list = os.listdir(os.path.join(image_path, subclass))
        for single_image in image_list:
            image_name = single_image.split('.')[0]
            if os.path.exists(os.path.join(face_data_path, subclass, image_name)):
                face_list.append(os.path.join(face_data_path, subclass, image_name))
            else:
                face_list.append('')
            attention_list.append(os.path.join(attention_data_path, subclass, image_name))
            scene_list.append(os.path.join(scene_data_path, subclass, image_name))
            labels.append(i)
    assert len(attention_list) == len(labels)
    assert len(face_list) == len(labels)
    assert len(scene_list) == len(labels)
    return face_list, attention_list, scene_list, labels


def shuffle_list(data_list1, data_list2, data_list3, labels):
    c = list(zip(data_list1, data_list2, data_list3, labels))
    shuffle(c)
    data_list1, data_list2, data_list3, labels = zip(*c)
    return data_list1, data_list2,data_list3, labels


def main(args):
    tf.reset_default_graph()
    train_or_test = args.train_or_test
    num_epoches = 10
    face_feature_len = 4096
    attention_feature_len = 2048 
    scene_feature_len = 1024 
    num_class = 3
    max_attention_nodes = 16
    if train_or_test == 'train': 
        max_face_nodes = 16
    else:
        max_face_nodes = 48
 
    X1 = tf.placeholder("float", [None, face_feature_len])
    X2 = tf.placeholder("float", [None, attention_feature_len])    
    X3 = tf.placeholder("float", [None, scene_feature_len])   
    Y = tf.placeholder("float", [None])
    dropout_flag = tf.placeholder_with_default(0.5, shape=())
    node_len = tf.placeholder("int32", shape=())
    net= GNN()
     
    with tf.name_scope("my_model"):
        net.build(
            features1=X1,
            features2=X2,
            features3=X3,
            face_feature_size=face_feature_len,
            attention_feature_size=attention_feature_len,
            scene_feature_size=scene_feature_len,
            hidden_size=128,
            edge_features_length=128,
            layer_num=1,
            num_face_nodes=node_len,
            num_attention_nodes=max_attention_nodes,
            use_bias=False,
            keep_prob=dropout_flag,
            num_classes=num_class,
            num_steps=4,
        )

    print '\ntrainable variables'
    for v in tf.trainable_variables():
        print v.name, v.get_shape().as_list()

    learning_rate = 0.0001
    corr = tf.equal(tf.argmax(net.probs, 1), tf.to_int64(Y))
    accuracy = tf.reduce_mean(tf.cast(corr, tf.float32)) 

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net.logits, labels=tf.to_int64(Y)))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=20)

    with tf.Session() as sess:
        print("start")
        sess.run(init)
        org_path = args.org_path
        face_path = args.face_feature_path
        attention_path = args.object_feature_path
        scene_path = args.scene_feature_path

        (
            face_whole_list, 
            attention_whole_list, 
            scene_whole_list, 
            label_whole_list,
        ) = read_data(
            org_path, 
            face_path,
            attention_path,
            scene_path
        )
        num_samples = len(face_whole_list)
        
        #if train_or_test in {'test', 'val'}:
        #    np.save(os.path.join(args.model_path, train_or_test + 'labels.npy', label_list))

        for epoch in range(num_epoches):       
            count=0
            if train_or_test == 'train':
                (
                    face_whole_list,
                    attention_whole_list,
                    scene_whole_list,
                    label_whole_list,
                ) = shuffle_list(
                    face_whole_list,
                    attention_whole_list,
                    scene_whole_list,
                    label_whole_list
                )   
                if not os.path.exists(args.model_path):
                    os.makedirs(args.model_path) 
            else:
                saver.restore(sess, os.path.join(args.model_path, "model_epoch"+str(epoch+1)+".ckpt"))
                probs_whole = []

            while count<num_samples:
                #face
                batch_face_list= face_whole_list[count]   
                if batch_face_list!='':                      
                    face_list = os.listdir(batch_face_list)
                    face_nodes_len = len(face_list)
                    batch_face_x = [] 
                    if train_or_test == 'train' and face_nodes_len> max_face_nodes:
                        shuffle(face_list)
                    face_nodes_len = face_nodes_len if face_nodes_len<max_face_nodes else max_face_nodes              
                    for j in range(face_nodes_len):
                        batch_face_x.append(
                            np.reshape(
                                np.load(os.path.join(batch_face_list, face_list[j])), 
                                [face_feature_len,]
                            )
                        )
                else:
                    face_nodes_len=0
                    batch_face_x = np.zeros((1, face_feature_len))

                #attention
                batch_attention_list = attention_whole_list[count]                         
                attention_list = os.listdir(batch_attention_list)
                assert len(attention_list) == max_attention_nodes
                attention_nodes_len = max_attention_nodes
                batch_attention_x = [] 
                for j in range(attention_nodes_len):
                    batch_attention_x.append(
                        np.reshape(
                            np.load(os.path.join(batch_attention_list, attention_list[j])),
                            [attention_feature_len,]
                        )
                    )

                batch_scene_list = scene_whole_list[count]
                batch_scene_x = np.reshape(np.load(batch_scene_list+'.npy'), [1, scene_feature_len])

                batch_y = np.repeat(label_whole_list[count], face_nodes_len+attention_nodes_len+1, axis=0)

                if train_or_test == 'train':
                    sess.run(train_op, 
                        feed_dict = {
                            X1:batch_face_x,
                            X2:batch_attention_x,
                            X3:batch_scene_x,
                            Y:batch_y,
                            dropout_flag:0.5,
                            node_len:face_nodes_len,
                        }
                    )
                else:
                    probs_out = sess.run(net.probs,
                        feed_dict = {
                            X1:batch_face_x,
                            X2:batch_attention_x,
                            X3:batch_scene_x,
                            Y:batch_y,
                            dropout_flag:1.0,
                            node_len:face_nodes_len,
                        }
                    )
                    probs = np.mean(probs_out, axis=0)
                    probs_whole.append(probs)

                count+=1

                if train_or_test == 'train' and count % (500)==0:

                    train_accuracy = sess.run(
                        accuracy,
                        feed_dict = {
                            X1:batch_face_x,
                            X2:batch_attention_x,
                            X3:batch_scene_x,
                            Y:batch_y,
                            dropout_flag:1.0,
                            node_len:face_nodes_len,
                        }
                    )            
                    print(" Step %d, training accuracy %f" % (count, train_accuracy))

            if train_or_test == 'train':
                model_name = os.path.join(args.model_path, "model_epoch"+str(epoch+1)+".ckpt")
                saver.save(sess, model_name)
            else:
                assert len(probs_whole)==num_samples
                #probs_whole=probs_whole.reshape((len(probs_whole),3))
                predicts = np.argmax(probs_whole, 1)
                #np.save(os.path.join(args.model_path, "model_epoch"+str(epoch+1)+'.npy'), probs_whole)
                print "Eopch " + str(epoch+1) + " accuracy is %f" % accuracy_score(label_whole_list, predicts)
       

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train GNN')
    parser.add_argument('--train_or_test', type=str, default='train', choices={'train', 'test'},
                        help='Train or test flag.')
    parser.add_argument('--org_path', type=str, default='./data/Original/Train', 
                        help='path to original data.')
    parser.add_argument('--face_feature_path', type=str, default='./data/faces_MTCNN_vgg_features/Train', 
                        help='path to the face features.')
    parser.add_argument('--object_feature_path', type=str, default='./data/object_senet_features/Train', 
                        help='path to the object features.')
    parser.add_argument('--scene_feature_path', type=str, default='./data/scene_features/Train', 
                        help='path to the scene features.')
    parser.add_argument('--model_path', type=str, default='./models/GNN_model', 
                        help='path to save the generated models or to the saved models.')    
    args = parser.parse_args()

    main(args)
