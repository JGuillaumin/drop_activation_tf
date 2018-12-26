import numpy as np
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"

import tensorflow as tf
from tensorflow.python.client import device_lib


from drop_activation import drop_activation


NB_RUNS = 1000


if __name__ == "__main__":

    print("======================================================================================")
    print("tf.__version__ : ", tf.__version__)
    print("tf.__compiler_version__ : ", tf.__compiler_version__)
    print("tf.__cxx11_abi_flag__ : ", tf.__cxx11_abi_flag__)
    print("tf.__git_version__ : ", tf.__git_version__)
    print("tf.__monolithic_build__ : ", tf.__monolithic_build__)

    print("is_built_with_cuda : ", tf.test.is_built_with_cuda())
    print("is_gpu_available : ", tf.test.is_gpu_available())

    for device in device_lib.list_local_devices():
        print("Device : {} | {}".format(device.name, device.device_type))

    print("======================================================================================")

    graph = tf.Graph()
    with graph.as_default(), tf.device("/cpu:0"):
        input_tf = tf.random_normal(shape=[1000, 1000], name="input")
        training = tf.placeholder(shape=[], dtype=tf.bool, name="training")
        # training = True
        output_tf = drop_activation(input_tf, training=training, p=0.95)

    with tf.Session(graph=graph) as sess:
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        #
        # res = sess.run(output_tf,
        #                feed_dict={training: True},
        #                options=run_options,
        #                run_metadata=run_metadata)
        # print("mean_res : {:.5f}".format(np.mean(res)))
        #
        # options = tf.profiler.ProfileOptionBuilder.time_and_memory()
        # options["min_bytes"] = 0
        # options["min_micros"] = 0
        # options["select"] = ("bytes", "peak_bytes", "output_bytes",
        #                      "residual_bytes")
        # tf.profiler.profile(graph, run_meta=run_metadata, cmd="scope", options=options)
        start = time.time()
        for _ in range(NB_RUNS):
            res = sess.run(output_tf, feed_dict={training: True})
            # res = sess.run(output_tf)
        end = time.time()

    print("{} iterations in {:.5f} s. {:.6f} s/it".format(NB_RUNS, end-start, (end-start)/NB_RUNS))