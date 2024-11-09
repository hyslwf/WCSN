import tensorflow as tf


class encoder(tf.keras.Model):
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, (5, 5), padding='same')
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.avgp1 = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same')

        self.conv2 = tf.keras.layers.Conv2D(32, (5, 5), padding='same')
        self.BN2 = tf.keras.layers.BatchNormalization()
        self.avgp2 = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.BN1(x)
        x = tf.nn.relu(x)
        x = self.avgp1(x)

        x = self.conv2(x)
        x = self.BN2(x)
        x = tf.nn.relu(x)
        outputs = self.avgp2(x)

        return outputs


class decoder(tf.keras.Model):
    def __init__(self, outputs_dim):
        super(decoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), padding='same')
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.upsp1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(16, (5, 5), padding='same')
        self.BN2 = tf.keras.layers.BatchNormalization()
        self.upsp2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv3 = tf.keras.layers.Conv2D(outputs_dim, (1, 3))

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.BN1(x)
        x = tf.nn.leaky_relu(x, alpha=0.3)
        x = self.upsp1(x)

        x = self.conv2(x)
        x = self.BN2(x)
        x = tf.nn.leaky_relu(x, alpha=0.3)
        x = self.upsp2(x)

        outputs = self.conv3(x)

        return outputs


class auto_encoder(tf.keras.Model):
    def __init__(self, outputs_dim):
        super(auto_encoder, self).__init__()

        # 降噪
        # self.dp = tf.keras.layers.Dropout(0.2)

        # encoder
        self.encoder_pattern = encoder()
        self.encoder_subject = encoder()

        # decoder
        self.decoder = decoder(outputs_dim)

    def call(self, inputs, **kwargs):
        # inputs = self.dp(inputs)

        x1 = self.encoder_pattern.call(inputs)
        x2 = self.encoder_subject.call(inputs)

        outputs = self.decoder.call(x1 + x2)

        return outputs


class cnn(tf.keras.Model):
    def __init__(self, output_size=2):
        super(cnn, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
        self.B1 = tf.keras.layers.BatchNormalization()
        self.avgp1 = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.B2 = tf.keras.layers.BatchNormalization()
        self.avgp2 = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same')

        self.Flat = tf.keras.layers.Flatten()

        self.d1 = tf.keras.layers.Dense(1024)
        self.B3 = tf.keras.layers.BatchNormalization()
        self.d2 = tf.keras.layers.Dense(128)
        self.B4 = tf.keras.layers.BatchNormalization()
        self.d3 = tf.keras.layers.Dense(output_size, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.B1(x)
        x = tf.nn.relu(x)
        x = self.avgp1(x)

        x = self.conv2(x)
        x = self.B2(x)
        x = tf.nn.relu(x)
        x = self.avgp2(x)

        x = self.Flat(x)

        x = self.d1(x)
        x = self.B3(x)
        x = tf.nn.relu(x)

        x = self.d2(x)
        x = self.B4(x)
        x = tf.nn.relu(x)

        outputs = self.d3(x)

        return outputs
