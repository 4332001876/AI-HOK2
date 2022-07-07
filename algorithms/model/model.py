"""
@author:
@file: model_new.py
@time: 2021/8/2 22:40
@file_desc:
"""
#__author_ _ = "yannickyu"

import tensorflow as tf
import numpy as np
from config.config import ModelConfig as Config
import sys
from config.config import DimConfig
# from SailLearner.algorithms.base.algorithm import Algorithm as AlgorithmBase
import threading


class Singleton(object):
    _instance_lock = threading.Lock()

    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self):
        if self._cls not in self._instance:
            with Singleton._instance_lock:
                if self._cls not in self._instance:
                    self._instance[self._cls] = self._cls()
        return self._instance[self._cls]


# Singleton Pattern
@Singleton
class Model():

    def __init__(self):
        # feature configure parameter
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        #self.need_reinforce_param_button_label_list = Config.NEED_REINFORCE_PARAM_BUTTON_LABEL_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.batch_size = Config.BATCH_SIZE*16

        # for actor:
        self.lstm_time_steps = 1
        self.batch_size = 1

        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        # self.hero_size = Config.HERO_SIZE
        # self.config_id_fea = Config.CONFIG_ID_FEA
        self.cut_points = [ value[0] for value in Config.data_shapes ]

        # Only True when evaluation
        self.deterministic_sample = False
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST # todo remove this?
        # print("legal_action size", self.legal_action_size)

        # net dims
        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST) # todo remove this?
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        self.graph = None

    def get_input_tensors(self):
        return [self.feature_ph, self.legal_action_ph, self.lstm_cell_ph, self.lstm_hidden_ph]

    def get_output_tensors(self):
        return [self.logits, self.value,
                self.lstm_cell_output, self.lstm_hidden_output]
        # + [self.used_legal_action]

    def build_infer_graph(self):
        if self.graph is not None:
            return self.graph
        self.graph = self._build_infer_graph()
        return self.graph

    def _build_infer_graph(self):
        self.graph = tf.Graph()
        # cpu_num = 1
        # config = tf.ConfigProto(device_count={"CPU": cpu_num},
        #                         inter_op_parallelism_threads=cpu_num,
        #                         intra_op_parallelism_threads=cpu_num,
        #                         log_device_placement=False)
        with self.graph.as_default():
            # Build input placeholders
            self.feature_ph = tf.placeholder(shape=(self.batch_size, self.feature_dim), name="feature", dtype=np.float32)
            self.legal_action_ph = tf.placeholder(shape=(self.batch_size, self.legal_action_dim), name="legal_action", dtype=np.float32)
            self.lstm_cell_ph = tf.placeholder(shape=(self.batch_size, self.lstm_hidden_dim), name="lstm_cell", dtype=np.float32)
            self.lstm_hidden_ph = tf.placeholder(shape=(self.batch_size, self.lstm_hidden_dim), name="lstm_hidden", dtype=np.float32)
            print("Build net: ", self.feature_dim, self.legal_action_dim, self.lstm_hidden_dim)

            # build graph (outputs)
            # data_list = tf.split(datas, self.cut_points, axis=1)
            feature = tf.reshape(self.feature_ph, [-1, self.seri_vec_split_shape[0][0]])
            legal_action = tf.reshape(self.legal_action_ph, [-1, np.sum(self.legal_action_size)])
            seri_vec = (feature, legal_action)
            # seri_vec = tf.reshape(self.feature_ph, [-1, self.data_split_shape[0]])
            init_lstm_cell, init_lstm_hidden = self.lstm_cell_ph, self.lstm_hidden_ph
            fc_label_result_list = self._inference(seri_vec, init_lstm_cell, init_lstm_hidden, only_inference=True)
            logits_list, value_list = fc_label_result_list[:-1], fc_label_result_list[-1]
            self.logits = tf.layers.flatten(tf.concat(logits_list, axis=1))
            self.value = tf.layers.flatten(value_list[0])
            # self.init_saver = tf.train.Saver(tf.global_variables())
            self.init = tf.global_variables_initializer()
            # self.sess = tf.Session(config=config)
            # self.sess.run(tf.global_variables_initializer())
        return self.graph

    def build_graph(self, datas, update):
        #add split datas
        data_list = tf.split(datas, self.cut_points, axis=1)
        #  the meaning of each data in data_list should be as the same as that in GpuProxy.py
        for i,data in enumerate(data_list):
            data = tf.reshape(data, [-1])
            data_list[i] = tf.cast(data,dtype=tf.float32)
        seri_vec = data_list[0]
        seri_vec = tf.reshape(seri_vec, [-1, self.data_split_shape[0]])

        reward = data_list[1]
        reward = tf.reshape(reward, [-1, self.data_split_shape[1]])

        advantage = data_list[2]
        advantage = tf.reshape(advantage, [-1, self.data_split_shape[2]])

        label_list = data_list[3:3+len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            #label_list[shape_index] = tf.cast(label_list,dtype=tf.int32)
            label_list[shape_index] = tf.cast(tf.reshape(label_list[shape_index], [-1, self.data_split_shape[3+shape_index]]),dtype=tf.int32)

        squeeze_label_list = []
        for ele in label_list:
            squeeze_label_list.append(tf.squeeze(ele, axis=[1]))


        old_label_probability_list = data_list[3+len(self.label_size_list):3+2*len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = tf.reshape(old_label_probability_list[shape_index], [-1, self.data_split_shape[3+len(self.label_size_list)+shape_index]])

        weight_list = data_list[3+2*len(self.label_size_list):3+3*len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            weight_list[shape_index] = tf.reshape(weight_list[shape_index], [-1, self.data_split_shape[3+2*len(self.label_size_list)+shape_index]])


        is_train = data_list[-3]
        is_train = tf.reshape(is_train, [-1, self.data_split_shape[-3]])

        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]
        # build network
        fc_label_result_list = self._inference(seri_vec, init_lstm_cell, init_lstm_hidden, squeeze_label_list)
        # calculate loss
        loss = self._calculate_loss(label_list, old_label_probability_list, fc_label_result_list[:-1], reward, advantage, fc_label_result_list[-1], seri_vec,is_train, weight_list)

        return loss, [loss, self.value_cost, self.policy_cost, self.entropy_cost]

    def get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.00001)

    def _squeeze_tensor(self, unsqueeze_reward, unsqueeze_advantage, unsqueeze_label_list, unsqueeze_frame_is_train, unsqueeze_weight_list):
        reward = tf.squeeze(unsqueeze_reward, axis=[1])
        advantage = tf.squeeze(unsqueeze_advantage, axis=[1])
        label_list = []
        for ele in unsqueeze_label_list:
            label_list.append(tf.squeeze(ele, axis=[1]))
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(tf.squeeze(weight, axis=[1]))
        frame_is_train = tf.squeeze(unsqueeze_frame_is_train, axis=[1])
        return reward, advantage, label_list, frame_is_train, weight_list

    def _calculate_loss(self, unsqueeze_label_list, old_label_probability_list, fc2_label_list, unsqueeze_reward, unsqueeze_advantage, fc2_value_result, seri_vec, unsqueeze_is_train, unsqueeze_weight_list):
        # todo cannot remove legal action here

        reward, advantage, label_list, is_train, weight_list = self._squeeze_tensor(unsqueeze_reward, unsqueeze_advantage, unsqueeze_label_list, unsqueeze_is_train, unsqueeze_weight_list)
        legal_action_flag_list = []
        split_feature_vec, split_feature_legal_action = tf.split(seri_vec, [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])], axis=1)
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = tf.reshape(split_feature_legal_action, feature_legal_action_shape)
        #self.feature_legal_action = feature_legal_action

        legal_action_flag_list = tf.split(feature_legal_action, self.label_size_list, axis = 1)

        # loss of value net
        fc2_value_result_squeezed = tf.squeeze(fc2_value_result, axis=[1])
        self.value_cost = 0.5 * tf.reduce_mean(tf.square(reward - fc2_value_result_squeezed), axis=0)
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * tf.reduce_mean(tf.square(new_advantage), axis=0)

        # for entropy loss calculate
        label_logits_subtract_max_list = []
        label_sum_exp_logits_list = []
        label_probability_list = []
        # policy loss: ppo clip loss
        self.policy_cost = tf.constant(0.0, dtype=tf.float32)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = tf.constant(0.0, dtype=tf.float32)
                one_hot_actions = tf.one_hot(label_list[task_index], self.label_size_list[task_index])
                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * tf.pow(10.0, 20.0)
                label_logits_subtract_max = tf.clip_by_value((fc2_label_list[task_index] - tf.reduce_max(fc2_label_list[task_index] - legal_action_flag_list_max_mask, axis=1, keep_dims=True)), -tf.pow(10.0, 20.0), 1)
                label_logits_subtract_max_list.append(label_logits_subtract_max)
                label_exp_logits = legal_action_flag_list[task_index] * tf.exp(label_logits_subtract_max) + self.min_policy
                label_sum_exp_logits = tf.reduce_sum(label_exp_logits, axis=1, keep_dims=True)
                label_sum_exp_logits_list.append(label_sum_exp_logits)
                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)
                policy_p = tf.reduce_sum(one_hot_actions * label_probability, axis=1)
                policy_log_p = tf.log(policy_p)
                old_policy_p = tf.reduce_sum(one_hot_actions * old_label_probability_list[task_index], axis=1)
                old_policy_log_p = tf.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = tf.exp(final_log_p)
                clip_ratio = tf.clip_by_value(ratio, 0.0, 3.0)
                surr1 = clip_ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                temp_policy_loss = - tf.reduce_sum(tf.to_float(weight_list[task_index]) * tf.minimum(surr1, surr2)) / tf.maximum(tf.reduce_sum(tf.to_float(weight_list[task_index])), 1.0)
                self.policy_cost = self.policy_cost + temp_policy_loss#- tf.reduce_sum(tf.to_float(weight_list[task_index]) * tf.minimum(surr1, surr2)) / tf.maximum(tf.reduce_sum(tf.to_float(weight_list[task_index])), 1.0) # CLIP loss, add - because need to minize

        # cross entropy loss
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                #temp_entropy_loss = -tf.reduce_sum(label_probability_list[current_entropy_loss_index] * (label_logits_subtract_max_list[current_entropy_loss_index] - tf.log(label_sum_exp_logits_list[current_entropy_loss_index])), axis=1)
                temp_entropy_loss = -tf.reduce_sum(label_probability_list[current_entropy_loss_index] * legal_action_flag_list[task_index]* tf.log(label_probability_list[current_entropy_loss_index]), axis=1)
                temp_entropy_loss = -tf.reduce_sum((temp_entropy_loss * tf.to_float(weight_list[task_index]))) / tf.maximum(tf.reduce_sum(tf.to_float(weight_list[task_index])), 1.0)# add - because need to minize
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = tf.constant(0.0, dtype=tf.float32)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = tf.constant(0.0, dtype=tf.float32)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element
        self.entropy_cost_list = entropy_loss_list
        # sum all type cost
        self.cost_all = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost
        # make output information
        # add loss information
        self.all_loss_list = [self.cost_all, self.value_cost, self.policy_cost, self.entropy_cost]
        return self.cost_all


    def _inference(self, seri_vec, init_lstm_cell, init_lstm_hidden, only_inference=False):

        # model design
        if only_inference:
            # actor input seri_vec as (feature, legal_action)
            # legal_action size differ from learner's
            split_feature_vec, split_feature_legal_action = seri_vec
        else:
            split_feature_vec, split_feature_legal_action = tf.split(seri_vec, [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])], axis=1)
        feature_vec_shape = list(self.seri_vec_split_shape[0])
        # feature_vec_shape.insert(0, -1)
        feature_vec_shape.insert(0, self.batch_size)
        feature_vec = tf.reshape(split_feature_vec, feature_vec_shape)
        feature_vec = tf.identity(feature_vec, name="feature_vec")
        lstm_cell_state = tf.reshape(init_lstm_cell, [-1, self.lstm_unit_size])
        lstm_hidden_state = tf.reshape(init_lstm_hidden, [-1, self.lstm_unit_size])
        lstm_initial_state = tf.nn.rnn_cell.LSTMStateTuple(lstm_cell_state, lstm_hidden_state)
        result_list = []
        with tf.variable_scope("main_fc"):
            fc1_public_weight = self._fc_weight_variable(shape=[feature_vec.shape.as_list()[-1], 512],
                                                         name="fc1_public_weight")
            fc1_public_bias = self._bias_variable(shape=[512], name="fc1_public_bias")
            fc1_public_result = tf.nn.relu((tf.matmul(feature_vec, fc1_public_weight) + fc1_public_bias),
                                           name="fc1_public_result")
            fc2_public_weight = self._fc_weight_variable(shape=[fc1_public_result.shape.as_list()[-1], 512],
                                                         name="fc2_public_weight")
            fc2_public_bias = self._bias_variable(shape=[512], name="fc2_public_bias")
            fc2_public_result = tf.nn.relu((tf.matmul(fc1_public_result, fc2_public_weight) + fc2_public_bias),
                                           name="fc2_public_result")
        reshape_fc_public_result = tf.reshape(fc2_public_result, [-1, self.lstm_time_steps, 512],
                                              name="reshape_fc_public_result")
        with tf.variable_scope("public_lstm"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_unit_size, forget_bias=1.0)
            with tf.variable_scope("rnn"):
                state = lstm_initial_state
                lstm_output_list = []
                for step in range(self.lstm_time_steps):
                    lstm_output, state = lstm_cell(reshape_fc_public_result[:, step, :], state)
                    lstm_output_list.append(lstm_output)
                lstm_outputs = tf.concat(lstm_output_list, axis=1, name="lstm_outputs")
                self.lstm_cell_output = state.c
                self.lstm_hidden_output = state.h
            reshape_lstm_outputs_result = tf.reshape(lstm_outputs, [-1, self.lstm_unit_size],
                                                     name="reshape_lstm_outputs_result")

        ## action layer ###
        for index in range(0, len(self.label_size_list) - 1):
            with tf.variable_scope("fc2_label_%d" % (index)):
                fc2_label_weight = self._fc_weight_variable(shape=[self.lstm_unit_size, self.label_size_list[index]],
                                                            name="fc2_label_%d_weight" % (index))
                fc2_label_bias = self._bias_variable(shape=[self.label_size_list[index]],
                                                     name="fc2_label_%d_bias" % (index))
                fc2_label_result = tf.add(tf.matmul(reshape_lstm_outputs_result, fc2_label_weight), fc2_label_bias,
                                          name="fc2_label_%d_result" % (index))
                result_list.append(fc2_label_result)

        with tf.variable_scope("fc2_label_%d" % (len(self.label_size_list) - 1)):
            fc2_label_weight = self._fc_weight_variable(shape=[self.lstm_unit_size, self.target_embed_dim],
                                                        name="fc2_label_%d_weight" % (len(self.label_size_list) - 1))
            fc2_label_bias = self._bias_variable(shape=[self.target_embed_dim],
                                                 name="fc2_label_%d_bias" % (len(self.label_size_list) - 1))
            fc2_label_result = tf.add(tf.matmul(reshape_lstm_outputs_result, fc2_label_weight), fc2_label_bias,
                                      name="fc2_label_%d_result" % (len(self.label_size_list) - 1))

        with tf.variable_scope("target"):
            fc2_label_weight = self._fc_weight_variable(
                shape=[fc2_label_result.shape.as_list()[-1], 8],
                name="fc_target_weight")
            fc2_label_bias = self._bias_variable(shape=[8],
                                                 name="fc_target_bias")
            fc2_label_result = tf.add(tf.matmul(fc2_label_result, fc2_label_weight), fc2_label_bias,
                                      name="fc_target_result")

            result_list.append(fc2_label_result)

        with tf.variable_scope("fc1_value"):
            fc1_value_weight = self._fc_weight_variable(shape=[self.lstm_unit_size, 64], name="fc1_value_weight")
            fc1_value_bias = self._bias_variable(shape=[64], name="fc1_value_bias")
            fc1_value_result = tf.nn.relu((tf.matmul(fc2_public_result, fc1_value_weight) + fc1_value_bias),
                                          name="fc1_value_result")

        with tf.variable_scope("fc2_value"):
            fc2_value_weight = self._fc_weight_variable(shape=[64, 1], name="fc2_value_weight")
            fc2_value_bias = self._bias_variable(shape=[1], name="fc2_value_bias")
            fc2_value_result = tf.add(tf.matmul(fc1_value_result, fc2_value_weight), fc2_value_bias,
                                      name="fc2_value_result")
            result_list.append(fc2_value_result)

        return result_list

    def _fc_weight_variable(self, shape, name, trainable=True):
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.orthogonal_initializer()
        return tf.get_variable(name, shape=shape, initializer=initializer,trainable=trainable)

    def _bias_variable(self, shape, name, trainable=True):
        initializer = tf.constant_initializer(0.0)
        return tf.get_variable(name, shape=shape, initializer=initializer,trainable=trainable)

    def _embed_variable(self, shape, name, trainable=True):
        initializer = tf.orthogonal_initializer()
        return tf.get_variable(name, shape=shape, initializer=initializer,trainable=trainable)
