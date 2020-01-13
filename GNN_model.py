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
import numpy as np
import tensorflow as tf


def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)

class GNN:
    def __init__(self):
        print("Start GNN Right Away!")
    
    def build(
        self, 
        features1, 
        features2,
        features3,
        face_feature_size,
        attention_feature_size,
        scene_feature_size,
        hidden_size,
        num_classes,
        num_steps,
        num_face_nodes,
        num_attention_nodes,
        edge_features_length,
        use_bias,
        keep_prob=0.5,
        layer_num=1
    ):
        
        #Add an extract fully connected layer to shrink the size of features
        self.face_weights = tf.Variable(glorot_init([face_feature_size, hidden_size]), name='face_weights')
        self.face_biases = tf.Variable(np.zeros([hidden_size]).astype(np.float32), name='face_biases')
        self.attention_weights = tf.Variable(glorot_init([attention_feature_size, hidden_size]), name='attention_weights')
        self.attention_biases = tf.Variable(np.zeros([hidden_size]).astype(np.float32), name='attention_biases')
        self.scene_weights = tf.Variable(glorot_init([scene_feature_size, hidden_size]), name='scene_weights')
        self.scene_biases = tf.Variable(np.zeros([hidden_size]).astype(np.float32), name='scene_biases')

        self.face_features = tf.nn.relu(tf.nn.bias_add(tf.matmul(features1, self.face_weights),self.face_biases)) 
        self.attention_features = tf.nn.relu(tf.nn.bias_add(tf.matmul(features2, self.attention_weights), self.attention_biases))
        self.scene_features = tf.nn.relu(tf.nn.bias_add(tf.matmul(features3, self.scene_weights), self.scene_biases))
 
        #define LSTM
        with tf.variable_scope("lstm_scope"):
            self.cell = tf.contrib.rnn.GRUCell(hidden_size)
            self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=keep_prob)
            #self.mlstm_cell = tf.contrib.rnn.MultiRNNCell([self.cell for _ in range(layer_num)])

        #define edge weights, edge bias, and mask used to take average of edge features. 
        self.face_edge_weights = tf.Variable(glorot_init([hidden_size, edge_features_length]), name='face_edge_weights')
        self.face_edge_biases = tf.Variable(np.zeros([edge_features_length]).astype(np.float32),name='face_edge_biases')
        self.attention_edge_weights = tf.Variable(glorot_init([hidden_size, edge_features_length]), name='attention_edge_weights')
        self.attention_edge_biases = tf.Variable(np.zeros([edge_features_length]).astype(np.float32),name='attention_edge_biases')
        self.scene_edge_weights = tf.Variable(glorot_init([hidden_size, edge_features_length]), name='scene_edge_weights')
        self.scene_edge_biases = tf.Variable(np.zeros([edge_features_length]).astype(np.float32),name='scene_edge_biases')
 
        with tf.variable_scope("lstm_scope") as scope:
            mask = tf.ones(
                       [num_face_nodes+num_attention_nodes + 1, num_face_nodes+num_attention_nodes + 1]
                   ) - tf.diag(tf.ones([num_face_nodes+num_attention_nodes + 1]))
            for step in range(num_steps):
                if step>0:
                   tf.get_variable_scope().reuse_variables()
                else:
                   self.state = tf.cond(
                                    tf.equal(num_face_nodes,0),
                                    lambda:tf.concat([self.attention_features, self.scene_features], axis=0),
                                    lambda:tf.concat([self.face_features, self.attention_features,self.scene_features], axis=0)
                                )
                
                m_face = tf.matmul(self.state[:num_face_nodes], tf.nn.dropout(self.face_edge_weights, keep_prob=keep_prob))                
                m_attention = tf.matmul(
                                  self.state[num_face_nodes:num_face_nodes+num_attention_nodes], 
                                  tf.nn.dropout(self.attention_edge_weights, keep_prob=keep_prob)
                              )
                m_scene = tf.matmul(
                              self.state[num_face_nodes+num_attention_nodes:],
                              tf.nn.dropout(self.scene_edge_weights, keep_prob=keep_prob)
                          )

                if use_bias:
                    m_face = tf.nn.bias_add(m_face, self.face_edge_biases)
                    m_face = tf.nn.bias_add(m_attention, self.attention_edge_biases)
                    m_scene = tf.nn.bias_add(m_scene, self.scene_edge_biases)
                m_combine = tf.concat([m_face, m_attention, m_scene], axis=0)
                acts = tf.multiply(tf.matmul(mask, m_combine), 1/(tf.cast(num_face_nodes+num_attention_nodes+1, tf.float32)-1))
                self.rnnoutput, self.state = self.cell(acts, self.state)
          

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [hidden_size, num_classes])
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
        self.logits = tf.matmul(self.rnnoutput, W) + b
        self.probs = tf.nn.softmax(self.logits)
        self.data_dict = None
        print("build model finished")
        
