import tensorflow as tf


def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(operation_timeout_in_ms=300000, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def count_parameter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(variable)
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

