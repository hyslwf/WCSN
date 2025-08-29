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


# Transformer块定义
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # 多头自注意力
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        outputs = self.layernorm2(out1 + ffn_output)

        return outputs


class TransformerClassifier(tf.keras.Model):
    def __init__(self, output_size=2, num_layers=2, d_model=64, num_heads=4, dff=256, dropout_rate=0.1):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # 投影层，把 40*3 → d_model
        self.dense_proj = tf.keras.layers.Dense(d_model)

        # Transformer编码层
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        # 分类头
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(output_size, activation='softmax')

    def call(self, inputs, training=False):
        # 输入: (batch, 40, 50, 3)
        # 转置维度 → (batch, 50, 40, 3)，方便按时间切分
        x = tf.transpose(inputs, perm=[0, 2, 1, 3])

        # 展平频域 + 通道 → (batch, 50, 40*3)
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1))

        # 投影到 d_model → (batch, 50, d_model)
        x = self.dense_proj(x)

        # 通过 Transformer 编码层
        for layer in self.enc_layers:
            x = layer(x, training=training)

        # 全局池化 + 分类
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        outputs = self.classifier(x)
        return outputs
