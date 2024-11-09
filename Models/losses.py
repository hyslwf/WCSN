import tensorflow as tf
import numpy as np


# 自编码器的loss函数
def auto_encoder_loss(train_data, model, optimizer, subject_number, pattern_number, out_loss=0):

    mse = tf.keras.losses.MeanSquaredError()
    margin = 0.2
    desired_density = 0.1
    losses = 0

    for i in range(subject_number):
        for j in range(pattern_number):
            samples = train_data[i][j]
            same_subject = train_data[i][(j + 1) % pattern_number]
            same_pattern = train_data[(i + 1) % subject_number][j]
            no_same = train_data[(i + 1) % subject_number][(j + 1) % pattern_number]

            # 随机打乱
            permutation = np.random.permutation(samples.shape[0])
            samples = samples[permutation]
            permutation = np.random.permutation(same_subject.shape[0])
            same_subject = same_subject[permutation]
            permutation = np.random.permutation(same_pattern.shape[0])
            same_pattern = same_pattern[permutation]
            permutation = np.random.permutation(no_same.shape[0])
            no_same = no_same[permutation]

            # 截断数组
            n = np.min((samples.shape[0], same_subject.shape[0], same_pattern.shape[0], no_same.shape[0]))
            samples = samples[0:n]
            same_subject = same_subject[0:n]
            same_pattern = same_pattern[0:n]
            no_same = no_same[0:n]

            with tf.GradientTape() as tape:

                # 重构loss
                x1 = model.encoder_subject(samples)
                x2 = model.encoder_pattern(samples)
                outputs = model.decoder(x1 + x2)
                mse_loss = mse(samples, outputs)

                # # KL散度
                # x_num = x1.shape[0] * x1.shape[1] * x1.shape[2] * x1.shape[3]
                # actual_density = ((tf.math.count_nonzero(x1) + tf.math.count_nonzero(x2)) / (2 * x_num))
                # actual_density = tf.cast(actual_density, tf.float32)
                # if actual_density == tf.constant(1.0, dtype=tf.float32):
                #     actual_density = tf.constant(0.999)
                # kl = desired_density * np.log(desired_density / actual_density)
                # kl += (1 - desired_density) * np.log((1 - desired_density) / (1 - actual_density))

                # 交叉loss
                x3 = model.encoder_subject(same_subject)
                x4 = model.encoder_pattern(same_pattern)
                outputs = model.decoder(x3 + x4)
                cross_loss = mse(samples, outputs)

                # 两个三重loss
                samples_subject_outputs = x1
                same_subject_subject_outputs = x3
                no_same_subject_outputs = model.encoder_subject(no_same)
                subject_trip1 = tf.reduce_sum(tf.square(samples_subject_outputs - same_subject_subject_outputs), 1)
                subject_trip2 = tf.reduce_sum(tf.square(samples_subject_outputs - no_same_subject_outputs), 1)
                trip_loss_sub = tf.maximum(0., margin + subject_trip1 - subject_trip2)
                trip_loss_sub = tf.reduce_mean(trip_loss_sub)

                samples_pattern_outputs = x2
                same_pattern_pattern_outputs = x4
                no_same_pattern_outputs = model.encoder_pattern(no_same)
                pattern_trip1 = tf.reduce_sum(tf.square(samples_pattern_outputs - same_pattern_pattern_outputs), 1)
                pattern_trip2 = tf.reduce_sum(tf.square(samples_pattern_outputs - no_same_pattern_outputs), 1)
                trip_loss_pat = tf.maximum(0., margin + pattern_trip1 - pattern_trip2)
                trip_loss_pat = tf.reduce_mean(trip_loss_pat)

                loss = mse_loss + cross_loss + 0.5 * (trip_loss_sub + trip_loss_pat)
                losses += loss

            if out_loss == 0:
                gradients = tape.gradient(target=loss, sources=model.trainable_variables)  # 计算梯度
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 更新梯度

    if out_loss == 0:
        return model
    else:
        return losses


# 对抗训练的loss
def cross_entropy_loss(train_x, train_y, encoder_model, cnn_model, optimizer, num, batch_size):

    ce = tf.keras.losses.CategoricalCrossentropy()

    for i in range(num):
        x = train_x[i * batch_size:(i + 1) * batch_size]
        y = train_y[i * batch_size:(i + 1) * batch_size]

        with tf.GradientTape() as tape:
            temp = encoder_model.encoder_subject(x)
            outputs = cnn_model(temp)
            loss = ce(y, outputs)
        l = []
        l.extend(encoder_model.encoder_subject.trainable_variables)
        l.extend(cnn_model.trainable_variables)
        gradients = tape.gradient(target=loss, sources=l)  # 计算梯度
        optimizer.apply_gradients(zip(gradients, l))  # 更新梯度

    return encoder_model, cnn_model
