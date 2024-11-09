import tensorflow as tf


def SGMG():
    # 设置 GPU 显存使用方式
    # 获取 GPU 设备列表
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
        # 设置 GPU 为增长式占用
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("set gpu memory growth over")
        except RuntimeError as e:
            # 打印异常
            print(e)
