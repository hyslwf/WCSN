import tensorflow as tf
import gc
from Data_preprocessing.data_preprocessing import preprocess, repreprocess, model_outputs
from Models.model import auto_encoder, cnn
from Models.losses import auto_encoder_loss, cross_entropy_loss


def model_training(train_x, train_y, val_x, val_y, test_x, test_y, lr, epochs, outputs_dim):
    # 释放内存
    tf.keras.backend.clear_session()
    # 垃圾回收，gc.collect() 返回处理这些循环引用一共释放掉的对象个数
    gc.collect()

    train_data = preprocess(train_x, train_y)
    val_data = preprocess(val_x, val_y)

    train_x, train_y = repreprocess(train_x, train_y)
    val_x, val_y = repreprocess(val_x, val_y)
    test_x, test_y = repreprocess(test_x, test_y)

    subject_number = len(train_data)
    pattern_number = len(train_data[0])
    batch_size = 10 * 7
    losses = []
    t = 0
    delta = 0.00001
    num = int(train_x.shape[0] / batch_size)

    mse = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(lr)

    # 构建模型
    encoder_model = auto_encoder(outputs_dim)
    cnn_model = cnn(subject_number)

    encoder_model.build(input_shape=(None, train_x.shape[1], train_x.shape[2], train_x.shape[3]))
    temp = encoder_model.encoder_subject(val_x[0:2])
    cnn_model.build(input_shape=(None, temp.shape[1], temp.shape[2], temp.shape[3]))
    del temp

    encoder_model.compile(loss=mse, optimizer=optimizer, metrics=['accuracy'])
    cnn_model.compile(loss=mse, optimizer=optimizer, metrics=['accuracy'])

    for epoch in range(epochs):

        # 自编码器训练
        encoder_model = auto_encoder_loss(train_data, encoder_model, optimizer, subject_number, pattern_number)
        # 对抗训练
        encoder_model, cnn_model = cross_entropy_loss(train_x, train_y, encoder_model, cnn_model,
                                                      optimizer, num, batch_size)

        # 求验证集loss并早停，学习率随时间下降
        encoder_loss = auto_encoder_loss(val_data, encoder_model, optimizer, subject_number, pattern_number, 1)
        cnn_ill_loss, cnn_ill_acc = cnn_model.evaluate(encoder_model.encoder_subject(val_x), val_y,
                                                       batch_size=batch_size, verbose=0)

        ill_loss = encoder_loss
        losses.append(ill_loss)
        if (epoch + 1) % 10 == 0:
            print(str(epoch + 1) + '\t:', float(ill_loss), float(cnn_ill_acc))
            lr *= 0.1
            optimizer = tf.keras.optimizers.SGD(lr)
        if len(losses) != 1:
            if ill_loss >= losses[-(2 + t)] - delta:
                t += 1
            else:
                t = 0
        if t > 20:
            break

    pre_test = cnn_model.predict(encoder_model.encoder_subject(test_x))
    pre_test_y = []
    for i in range(test_x.shape[0]):
        if pre_test[i, 0] >= 0.5:
            pre_test_y.append(0)
        else:
            pre_test_y.append(1)

    y = []
    for i in range(test_x.shape[0]):
        if test_y[i, 0] >= 0.5:
            y.append(0)
        else:
            y.append(1)

    return model_outputs(pre_test_y, y, pattern_number)
