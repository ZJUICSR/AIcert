# The functions in this file is used to tackle the gradient computation problem happened in tf>=2.2
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer as lso
# from autokeras.utils.see_rnn import inspect_gen as igrnn


def check_version(cur_version, exp_version="2.2.0"):
    # out =True,means the version is over/equal to tf2.2. use new solution

    MAX_VERSION = "0.0.0"
    CODE = True

    if exp_version == cur_version:
        CODE = True
        MAX_VERSION = exp_version
        return CODE

    cur_versionBITS = cur_version.split(".")
    exp_versionBITS = exp_version.split(".")


    if len(cur_versionBITS) >= len(exp_versionBITS):
        amount = len(cur_versionBITS) - len(exp_versionBITS)
        for i in range(amount):
            exp_versionBITS.append("0")
    else:
        amount = len(exp_versionBITS) - len(cur_versionBITS)
        for i in range(amount):
            cur_versionBITS.append("0")

    for i in range(len(cur_versionBITS)):
        try:
            if int(cur_versionBITS[i]) > int(exp_versionBITS[i]):
                CODE = True
                MAX_VERSION = cur_version
                return CODE
            elif int(cur_versionBITS[i]) < int(exp_versionBITS[i]):
                CODE = False
                MAX_VERSION = exp_version
                return CODE
            else:
                CODE = True
                MAX_VERSION = exp_version
        except IndexError as err:
            pass

    return CODE

# a=check_version("2.1","2.1.0")
# b=check_version("2.1.0","2.1.0")
# c=check_version("2.2.0","2.1.0")
# d=check_version("2.1.0","2.2.0")

def _get_grads_graph(model, x, y, params, sample_weight=None, learning_phase=0):
    sample_weight = sample_weight or np.ones(len(x))

    outputs = model.optimizer.get_gradients(model.total_loss, params)
    inputs  = (model.inputs + model._feed_targets + model._feed_sample_weights
               + [K.learning_phase()])

    grads_fn = K.function(inputs, outputs)
    gradients = grads_fn([x, y, sample_weight, learning_phase])
    return gradients

def _get_grads_eager(model, x, y, params, sample_weight=None, learning_phase=0):
    def _process_input_data(x, y, sample_weight, model):
        iterator = data_adapter.single_batch_iterator(model.distribute_strategy,
                                                      x, y, sample_weight,
                                                      class_weight=None)
        data = next(iterator)
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        return x, y, sample_weight

    def _clip_scale_grads(strategy, tape, optimizer, loss, params):
        with tape:
            if isinstance(optimizer, lso.LossScaleOptimizer):
                loss = optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, params)

        aggregate_grads_outside_optimizer = (
            optimizer._HAS_AGGREGATE_GRAD and not isinstance(
                strategy.extended,
                parameter_server_strategy.ParameterServerStrategyExtended))

        if aggregate_grads_outside_optimizer:
            gradients = optimizer._aggregate_gradients(zip(gradients, params))
        if isinstance(optimizer, lso.LossScaleOptimizer):
            gradients = optimizer.get_unscaled_gradients(gradients)

        # zxy
        # gradients = optimizer._clip_gradients(gradients)
        return gradients

    x, y, sample_weight = _process_input_data(x, y, sample_weight, model)

    with tf.GradientTape() as tape:
        y_pred = model(x, training=bool(learning_phase))
        loss = model.compiled_loss(y, y_pred, sample_weight,
                                   regularization_losses=model.losses)

    gradients = _clip_scale_grads(model.distribute_strategy, tape,
                                  model.optimizer, loss, params)

    # zxy
    new_gradients=[]
    for gra in gradients:
        if isinstance(gradients[0],tuple):
            new_gradients.append(gra[0])
    gradients = K.batch_get_value(new_gradients)
    # gradients = K.batch_get_value(gradients)
    return gradients

def get_gradients(model, x, y, params=None, sample_weight=None, learning_phase=0,
                  evaluate=True):
    if params==None:
        params=model.trainable_weights
    if tf.executing_eagerly():
        return _get_grads_eager(model, x, y, params, sample_weight,
                                learning_phase)
    else:
        return _get_grads_graph(model, x, y, params, sample_weight,
                                learning_phase)


def rnn_get_gradients(model,name, x, y, params=None, sample_weight=None, learning_phase=0):
    gradients=igrnn.get_gradients(model,name, x, y)
    return gradients