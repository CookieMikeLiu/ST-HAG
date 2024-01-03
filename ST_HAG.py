
from platform import node
from random import sample
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, Adadelta, Nadam
from params_flow.activations import gelu
from keras_gcn.layers import GraphConv
from utils import *




def ST_HAG_ADHD(feature_dim,node_number):
    """ST-HAG model

    Args:
        feature_dim: number of fMRI time sequence
        node_number: number of ROI nodes

    Returns:
        ST-HAG model
    """
    data_input = keras.layers.Input(shape=(node_number, feature_dim), dtype='float32', name="data_input")
    edge_input = keras.layers.Input(shape=(3*node_number, 3*node_number), dtype='float32', name="edge_input")
    pheno_input = keras.layers.Input((4,), name='pheno_input')
    
    temporal_input = keras.layers.Permute((2, 1), name='temporal_input')(data_input)
    
    
    time_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=node_number,
                                                  vocab_size=1)(temporal_input)


    # Layer 1

    # FSTSA with sliding window attention unit
    time_input_spliting1 = keras.layers.Lambda(slice, arguments={'h1': 0, 'h2': 30})(time_position_embedding)
    time_input_spliting2 = keras.layers.Lambda(slice, arguments={'h1': 15, 'h2': 45})(time_position_embedding)
    time_input_spliting3 = keras.layers.Lambda(slice, arguments={'h1': 30, 'h2': 60})(time_position_embedding)
    time_input_spliting4 = keras.layers.Lambda(slice, arguments={'h1': 45, 'h2': 75})(time_position_embedding)
    time_input_spliting5 = keras.layers.Lambda(slice, arguments={'h1': 60, 'h2': 90})(time_position_embedding)

    time_valueS1_l1, time_attentionS1_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting1)
    time_attention_probS1_l1 = keras.layers.Softmax()(time_attentionS1_l1)
    time_attention_probS1_l1 = keras.layers.Dropout(0.1)(time_attention_probS1_l1)
    time_attention_scoreS1_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS1_l1, time_attention_probS1_l1])
    projection_timeS1_l1 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS1_l1, time_input_spliting1])
    
    time_valueS2_l1, time_attentionS2_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting2)
    time_attention_probS2_l1 = keras.layers.Softmax()(time_attentionS2_l1)
    time_attention_probS2_l1 = keras.layers.Dropout(0.1)(time_attention_probS2_l1)
    time_attention_scoreS2_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS2_l1, time_attention_probS2_l1])
    projection_timeS2_l1 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS2_l1, time_input_spliting2])
    
    time_valueS3_l1, time_attentionS3_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting3)
    time_attention_probS3_l1 = keras.layers.Softmax()(time_attentionS3_l1)
    time_attention_probS3_l1 = keras.layers.Dropout(0.1)(time_attention_probS3_l1)
    time_attention_scoreS3_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS3_l1, time_attention_probS3_l1])
    projection_timeS3_l1 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS3_l1, time_input_spliting3])
    
    time_valueS4_l1, time_attentionS4_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting4)
    time_attention_probS4_l1 = keras.layers.Softmax()(time_attentionS4_l1)
    time_attention_probS4_l1 = keras.layers.Dropout(0.1)(time_attention_probS4_l1)
    time_attention_scoreS4_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS4_l1, time_attention_probS4_l1])
    projection_timeS4_l1 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS4_l1, time_input_spliting4])
    
    time_valueS5_l1, time_attentionS5_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting5)
    time_attention_probS5_l1 = keras.layers.Softmax()(time_attentionS5_l1)
    time_attention_probS5_l1 = keras.layers.Dropout(0.1)(time_attention_probS5_l1)
    time_attention_scoreS5_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS5_l1, time_attention_probS5_l1])
    projection_timeS5_l1 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS5_l1, time_input_spliting5])
    
    bert_out_timeS1_l1_part1,bert_out_timeS1_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15,'h3': 30})(projection_timeS1_l1)
    bert_out_timeS2_l1_part1,bert_out_timeS2_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15,'h3': 30})(projection_timeS2_l1)
    bert_out_timeS3_l1_part1,bert_out_timeS3_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15,'h3': 30})(projection_timeS3_l1)
    bert_out_timeS4_l1_part1,bert_out_timeS4_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15,'h3': 30})(projection_timeS4_l1)
    bert_out_timeS5_l1_part1,bert_out_timeS5_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15,'h3': 30})(projection_timeS5_l1)

    bert_out_timeS1_begin = bert_out_timeS1_l1_part1
    bert_out_timeS1_one = keras.layers.Add()([bert_out_timeS1_l1_part2*0.5,bert_out_timeS2_l1_part1*0.5])
    bert_out_timeS1_two = keras.layers.Add()([bert_out_timeS2_l1_part2*0.5,bert_out_timeS3_l1_part1*0.5])
    bert_out_timeS1_three = keras.layers.Add()([bert_out_timeS3_l1_part2*0.5,bert_out_timeS4_l1_part1*0.5])
    bert_out_timeS1_four = keras.layers.Add()([bert_out_timeS4_l1_part2*0.5,bert_out_timeS5_l1_part1*0.5])
    bert_out_timeS1_end = bert_out_timeS5_l1_part2
    
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_timeS1_begin,bert_out_timeS1_one])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1,bert_out_timeS1_two])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1,bert_out_timeS1_three])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1,bert_out_timeS1_four])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1,bert_out_timeS1_end])
    
    intermediate_time_l1 = keras.layers.Dense(units=4 * node_number, activation=gelu,
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        bert_out_time_l1)
    bert_out_l1 = ProjectionLayer(hidden_size=node_number)([intermediate_time_l1, bert_out_time_l1])
    
    
    # DAN-GCN
    sliced_input1 = Graph_partition(6,15)(data_input)

    conv_out_l1 = []
    for i, element in enumerate(sliced_input1):
        if i == 0:
            t1 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([t1,sliced_input1[i+1]])
            conv_out = GraphConv(
                    units=15,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l1.append(conv_out)
        elif i == 5:
            t3 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([sliced_input1[i - 1], t3])
            conv_out = GraphConv(
                    units=15,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l1.append(conv_out)
        else:
            true_input = keras.layers.Concatenate(axis=1)([sliced_input1[i-1],sliced_input1[i],sliced_input1[i+1]])
            conv_out = GraphConv(
                    units=15,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l1.append(conv_out)
            
    conv_out_l1 = keras.layers.Concatenate()(conv_out_l1)
    conv_layer1 = keras.layers.BatchNormalization()(conv_out_l1)
    conv_layer1 = keras.layers.Activation('relu')(conv_layer1)
    conv_layer1 = keras.layers.Dropout(0.5)(conv_layer1)
    
    conv_layer1_rv = keras.layers.Permute((2, 1))(conv_layer1)
    conv_layer1_out = keras.layers.Multiply()([bert_out_time_l1, conv_layer1_rv])
    conv_layer1_out = keras.layers.Add()([conv_layer1_out, conv_layer1_rv])
    conv_layer1_out = keras.layers.Permute((2, 1))(conv_layer1_out)



    # layer 2
    # FSTSA with sliding window attention unit
    
    time_input_spliting6 = keras.layers.Lambda(slice, arguments={'h1': 0, 'h2': 60})(bert_out_l1)
    time_input_spliting7 = keras.layers.Lambda(slice, arguments={'h1': 30, 'h2': 90})(bert_out_l1)

    time_valueS1_l2, time_attentionS1_l2 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting6)
    time_valueS2_l2, time_attentionS2_l2 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting7)

    time_attention_probS1_l2 = keras.layers.Softmax()(time_attentionS1_l2)
    time_attention_probS1_l2 = keras.layers.Dropout(0.1)(time_attention_probS1_l2)
    time_attention_scoreS1_l2 = MyAttention(num_heads=10, size_per_head=19)([time_valueS1_l2, time_attention_probS1_l2])
    projection_timeS1_l2 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS1_l2, time_input_spliting6])
   
    
    time_attention_probS2_l2 = keras.layers.Softmax()(time_attentionS2_l2)
    time_attention_probS2_l2 = keras.layers.Dropout(0.1)(time_attention_probS2_l2)
    time_attention_scoreS2_l2 = MyAttention(num_heads=10, size_per_head=19)([time_valueS2_l2, time_attention_probS2_l2])
    projection_timeS2_l2 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS2_l2, time_input_spliting7])
    
    bert_out_timeS1_l2_part1, bert_out_timeS1_l2_part2 = keras.layers.Lambda(slice2parts,
                                                                             arguments={'h1': 0, 'h2': 30, 'h3': 60})(projection_timeS1_l2)
    bert_out_timeS2_l2_part1, bert_out_timeS2_l2_part2 = keras.layers.Lambda(slice2parts,
                                                                             arguments={'h1': 0, 'h2': 30, 'h3': 60})(projection_timeS2_l2)

    bert_out_timeS2_begin = bert_out_timeS1_l2_part1
    bert_out_timeS2_one = keras.layers.Add()([bert_out_timeS1_l2_part2 * 0.5, bert_out_timeS2_l2_part1 * 0.5])
    bert_out_timeS2_end = bert_out_timeS2_l2_part2

    bert_out_time_l2 = keras.layers.Concatenate(axis=1)([bert_out_timeS2_begin, bert_out_timeS2_one])
    bert_out_time_l2 = keras.layers.Concatenate(axis=1)([bert_out_time_l2, bert_out_timeS2_end])
    
    intermediate_time_l2 = keras.layers.Dense(units=4 * node_number, activation=gelu,
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        bert_out_time_l2)
    bert_out_l2 = ProjectionLayer(hidden_size=node_number)([intermediate_time_l2, bert_out_time_l2])
    
    # DAN-GCN
    sliced_input2 = Graph_partition(3,30)(conv_layer1_out)

    conv_out_l2 = []
    for i, element in enumerate(sliced_input2):
        if i == 0:
            t1 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([t1, sliced_input1[i + 1]])
            conv_out = GraphConv(
                    units=30,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l2.append(conv_out)
        elif i == 2:
            t3 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([sliced_input1[i-1],t3])
            conv_out = GraphConv(
                    units=30,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l2.append(conv_out)
        else:
            true_input = keras.layers.Concatenate(axis=1)([sliced_input2[i-1],sliced_input2[i],sliced_input2[i+1]])
            conv_out = GraphConv(
                    units=30,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l2.append(conv_out)
            
    conv_out_l2 = keras.layers.Concatenate()(conv_out_l2)
    conv_layer2 = keras.layers.BatchNormalization()(conv_out_l2)
    conv_layer2 = keras.layers.Activation('relu')(conv_layer2)
    conv_layer2 = keras.layers.Dropout(0.5)(conv_layer2)
    
    conv_layer2_rv = keras.layers.Permute((2, 1))(conv_layer2)
    conv_layer2_out = keras.layers.Multiply()([bert_out_time_l2, conv_layer2_rv])
    conv_layer2_out = keras.layers.Add()([conv_layer2_out, conv_layer2_rv])
    conv_layer2_out = keras.layers.Permute((2, 1))(conv_layer2_out)
    
    # Spatial and temporal fusion
    
    bert_out_l2 =  keras.layers.Permute((2, 1))(bert_out_l2)
    fusion_out = STGFU()([conv_layer2_out,bert_out_l2])

    global_avg_pooling = GlobalAvgPool()(fusion_out)
    merge = keras.layers.Concatenate()([global_avg_pooling, pheno_input])
    dense = keras.layers.Dense(64, activation='relu')(merge)
    dense = keras.layers.Dropout(0.5)(dense)
    dense = keras.layers.Dense(10, activation='relu')(dense)
    output = keras.layers.Dense(1, activation='sigmoid')(dense)

    model_ADHD = keras.Model(inputs=[data_input, edge_input, pheno_input], outputs=output)
    # opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    # opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model_ADHD.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model_ADHD



def ST_HAG_ASD(feature_dim,node_number):
    data_input = keras.layers.Input(shape=(node_number, feature_dim), dtype='float32', name="data_input")
    edge_input = keras.layers.Input(shape=(3*node_number, 3*node_number), dtype='float32', name="edge_input")
    # graph_connect_input = keras.layers.Input(shape=(node_number, node_number), dtype='float32', name="graph_connect")
    pheno_input = keras.layers.Input((4,), name='pheno_input')
    
    temporal_input = keras.layers.Permute((2, 1), name='temporal_input')(data_input)
    
    
    time_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=node_number,
                                                  vocab_size=1)(temporal_input)

    
    time_input_spliting1 = keras.layers.Lambda(slice, arguments={'h1': 0, 'h2': 30})(time_position_embedding)
    time_input_spliting2 = keras.layers.Lambda(slice, arguments={'h1': 15, 'h2': 45})(time_position_embedding)
    time_input_spliting3 = keras.layers.Lambda(slice, arguments={'h1': 30, 'h2': 60})(time_position_embedding)
    time_input_spliting4 = keras.layers.Lambda(slice, arguments={'h1': 45, 'h2': 75})(time_position_embedding)
    time_input_spliting5 = keras.layers.Lambda(slice, arguments={'h1': 60, 'h2': 90})(time_position_embedding)

    time_valueS1_l1, time_attentionS1_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting1)
    time_attention_probS1_l1 = keras.layers.Softmax()(time_attentionS1_l1)
    time_attention_probS1_l1 = keras.layers.Dropout(0.1)(time_attention_probS1_l1)
    time_attention_scoreS1_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS1_l1, time_attention_probS1_l1])
    projection_timeS1_l1 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS1_l1, time_input_spliting1])
    
    time_valueS2_l1, time_attentionS2_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting2)
    time_attention_probS2_l1 = keras.layers.Softmax()(time_attentionS2_l1)
    time_attention_probS2_l1 = keras.layers.Dropout(0.1)(time_attention_probS2_l1)
    time_attention_scoreS2_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS2_l1, time_attention_probS2_l1])
    projection_timeS2_l1 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS2_l1, time_input_spliting2])
    
    time_valueS3_l1, time_attentionS3_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting3)
    time_attention_probS3_l1 = keras.layers.Softmax()(time_attentionS3_l1)
    time_attention_probS3_l1 = keras.layers.Dropout(0.1)(time_attention_probS3_l1)
    time_attention_scoreS3_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS3_l1, time_attention_probS3_l1])
    projection_timeS3_l1 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS3_l1, time_input_spliting3])
    
    time_valueS4_l1, time_attentionS4_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting4)
    time_attention_probS4_l1 = keras.layers.Softmax()(time_attentionS4_l1)
    time_attention_probS4_l1 = keras.layers.Dropout(0.1)(time_attention_probS4_l1)
    time_attention_scoreS4_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS4_l1, time_attention_probS4_l1])
    projection_timeS4_l1 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS4_l1, time_input_spliting4])
    
    time_valueS5_l1, time_attentionS5_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting5)
    time_attention_probS5_l1 = keras.layers.Softmax()(time_attentionS5_l1)
    time_attention_probS5_l1 = keras.layers.Dropout(0.1)(time_attention_probS5_l1)
    time_attention_scoreS5_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS5_l1, time_attention_probS5_l1])
    projection_timeS5_l1 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS5_l1, time_input_spliting5])
    
    bert_out_timeS1_l1_part1,bert_out_timeS1_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15,'h3': 30})(projection_timeS1_l1)
    bert_out_timeS2_l1_part1,bert_out_timeS2_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15,'h3': 30})(projection_timeS2_l1)
    bert_out_timeS3_l1_part1,bert_out_timeS3_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15,'h3': 30})(projection_timeS3_l1)
    bert_out_timeS4_l1_part1,bert_out_timeS4_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15,'h3': 30})(projection_timeS4_l1)
    bert_out_timeS5_l1_part1,bert_out_timeS5_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15,'h3': 30})(projection_timeS5_l1)

    bert_out_timeS1_begin = bert_out_timeS1_l1_part1
    bert_out_timeS1_one = keras.layers.Add()([bert_out_timeS1_l1_part2*0.5,bert_out_timeS2_l1_part1*0.5])
    bert_out_timeS1_two = keras.layers.Add()([bert_out_timeS2_l1_part2*0.5,bert_out_timeS3_l1_part1*0.5])
    bert_out_timeS1_three = keras.layers.Add()([bert_out_timeS3_l1_part2*0.5,bert_out_timeS4_l1_part1*0.5])
    bert_out_timeS1_four = keras.layers.Add()([bert_out_timeS4_l1_part2*0.5,bert_out_timeS5_l1_part1*0.5])
    bert_out_timeS1_end = bert_out_timeS5_l1_part2
    
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_timeS1_begin,bert_out_timeS1_one])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1,bert_out_timeS1_two])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1,bert_out_timeS1_three])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1,bert_out_timeS1_four])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1,bert_out_timeS1_end])
    
    intermediate_time_l1 = keras.layers.Dense(units=4 * node_number, activation=gelu,
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        bert_out_time_l1)
    bert_out_l1 = ProjectionLayer(hidden_size=node_number)([intermediate_time_l1, bert_out_time_l1])
    
    
     # layer 1
    sliced_input1 = Graph_partition(6,15)(data_input)

    # spatial level
    conv_out_l1 = []
    for i, element in enumerate(sliced_input1):
        if i == 0:
            t1 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([t1,sliced_input1[i+1]])
            conv_out = GraphConv(
                    units=15,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l1.append(conv_out)
        elif i == 5:
            t3 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([sliced_input1[i - 1], t3])
            conv_out = GraphConv(
                    units=15,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l1.append(conv_out)
        else:
            true_input = keras.layers.Concatenate(axis=1)([sliced_input1[i-1],sliced_input1[i],sliced_input1[i+1]])
            conv_out = GraphConv(
                    units=15,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l1.append(conv_out)
            
    conv_out_l1 = keras.layers.Concatenate()(conv_out_l1)
    conv_layer1 = keras.layers.BatchNormalization()(conv_out_l1)
    conv_layer1 = keras.layers.Activation('relu')(conv_layer1)
    conv_layer1 = keras.layers.Dropout(0.5)(conv_layer1)
    
    conv_layer1_rv = keras.layers.Permute((2, 1))(conv_layer1)
    conv_layer1_out = keras.layers.Multiply()([bert_out_time_l1, conv_layer1_rv])
    conv_layer1_out = keras.layers.Add()([conv_layer1_out, conv_layer1_rv])
    conv_layer1_out = keras.layers.Permute((2, 1))(conv_layer1_out)


    # layer 2
    time_input_spliting6 = keras.layers.Lambda(slice, arguments={'h1': 0, 'h2': 60})(bert_out_l1)
    time_input_spliting7 = keras.layers.Lambda(slice, arguments={'h1': 30, 'h2': 90})(bert_out_l1)

    time_valueS1_l2, time_attentionS1_l2 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting6)
    time_valueS2_l2, time_attentionS2_l2 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting7)

    time_attention_probS1_l2 = keras.layers.Softmax()(time_attentionS1_l2)
    time_attention_probS1_l2 = keras.layers.Dropout(0.1)(time_attention_probS1_l2)
    time_attention_scoreS1_l2 = MyAttention(num_heads=10, size_per_head=19)([time_valueS1_l2, time_attention_probS1_l2])
    projection_timeS1_l2 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS1_l2, time_input_spliting6])
   
    
    time_attention_probS2_l2 = keras.layers.Softmax()(time_attentionS2_l2)
    time_attention_probS2_l2 = keras.layers.Dropout(0.1)(time_attention_probS2_l2)
    time_attention_scoreS2_l2 = MyAttention(num_heads=10, size_per_head=19)([time_valueS2_l2, time_attention_probS2_l2])
    projection_timeS2_l2 = ProjectionLayer(hidden_size=190)(
        [time_attention_scoreS2_l2, time_input_spliting7])
    
    bert_out_timeS1_l2_part1, bert_out_timeS1_l2_part2 = keras.layers.Lambda(slice2parts,
                                                                             arguments={'h1': 0, 'h2': 30, 'h3': 60})(projection_timeS1_l2)
    bert_out_timeS2_l2_part1, bert_out_timeS2_l2_part2 = keras.layers.Lambda(slice2parts,
                                                                             arguments={'h1': 0, 'h2': 30, 'h3': 60})(projection_timeS2_l2)

    bert_out_timeS2_begin = bert_out_timeS1_l2_part1
    bert_out_timeS2_one = keras.layers.Add()([bert_out_timeS1_l2_part2 * 0.5, bert_out_timeS2_l2_part1 * 0.5])
    bert_out_timeS2_end = bert_out_timeS2_l2_part2

    bert_out_time_l2 = keras.layers.Concatenate(axis=1)([bert_out_timeS2_begin, bert_out_timeS2_one])
    bert_out_time_l2 = keras.layers.Concatenate(axis=1)([bert_out_time_l2, bert_out_timeS2_end])
    
    intermediate_time_l2 = keras.layers.Dense(units=4 * node_number, activation=gelu,
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        bert_out_time_l2)
    bert_out_l2 = ProjectionLayer(hidden_size=node_number)([intermediate_time_l2, bert_out_time_l2])
    
    
    
    # spatial level
    sliced_input2 = Graph_partition(3,30)(conv_layer1_out)

    conv_out_l2 = []
    for i, element in enumerate(sliced_input2):
        if i == 0:
            t1 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([t1, sliced_input1[i + 1]])
            conv_out = GraphConv(
                    units=30,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l2.append(conv_out)
        elif i == 2:
            t3 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([sliced_input1[i-1],t3])
            conv_out = GraphConv(
                    units=30,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l2.append(conv_out)
        else:
            true_input = keras.layers.Concatenate(axis=1)([sliced_input2[i-1],sliced_input2[i],sliced_input2[i+1]])
            conv_out = GraphConv(
                    units=30,
                    step_num=1,
                )([true_input, edge_input])
            conv_out = conv_out[:,190:380,:]
            conv_out_l2.append(conv_out)
            
    conv_out_l2 = keras.layers.Concatenate()(conv_out_l2)
    conv_layer2 = keras.layers.BatchNormalization()(conv_out_l2)
    conv_layer2 = keras.layers.Activation('relu')(conv_layer2)
    conv_layer2 = keras.layers.Dropout(0.5)(conv_layer2)
    
    conv_layer2_rv = keras.layers.Permute((2, 1))(conv_layer2)
    conv_layer2_out = keras.layers.Multiply()([bert_out_time_l2, conv_layer2_rv])
    conv_layer2_out = keras.layers.Add()([conv_layer2_out, conv_layer2_rv])
    conv_layer2_out = keras.layers.Permute((2, 1))(conv_layer2_out)
    
    
    bert_out_l2 =  keras.layers.Permute((2, 1))(bert_out_l2)

    fusion_out = STGFU()([conv_layer2_out,bert_out_l2])
    fusion_out =  keras.layers.Permute((2, 1))(fusion_out)

    global_avg_pooling = GlobalAvgPool()(fusion_out)
    merge = keras.layers.Concatenate()([global_avg_pooling, pheno_input])
    dense = keras.layers.Dense(64, activation='relu')(merge)
    dense = keras.layers.Dropout(0.5)(dense)
    dense = keras.layers.Dense(10, activation='relu')(dense)
    output = keras.layers.Dense(1, activation='sigmoid')(dense)

    model_ASD = keras.Model(inputs=[data_input, edge_input, pheno_input], outputs=output)
    # opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    # opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model_ASD.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model_ASD