import tensorflow as tf
from tensorflow.keras import layers, models

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformer_block(x, embed_dim, num_heads, dropout_rate):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
    mlp_output = mlp(attention_output, hidden_units=[embed_dim * 2, embed_dim], dropout_rate=dropout_rate)
    return layers.LayerNormalization(epsilon=1e-6)(attention_output + mlp_output)

def create_vit_classifier(input_shape, num_layers, embed_dim, num_heads, mlp_dim, num_classes, dropout_rate):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(embed_dim)(inputs)
    x = layers.Reshape((input_shape[0] * input_shape[1], embed_dim))(x)
    for _ in range(num_layers):
        x = transformer_block(x, embed_dim, num_heads, dropout_rate)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def decode_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [128, 128])  # Ensure all images have the same size
    return image

def process_path(input_image_path, target_image_path):
    input_image = decode_image(input_image_path)
    target_image = decode_image(target_image_path)
    return input_image, target_image

def prepare_dataset(input_img_paths, target_img_paths, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

input_shape = (128, 128, 3)
num_layers = 4
embed_dim = 64
num_heads = 4
mlp_dim = 128
num_classes = 3
dropout_rate = 0.1
batch_size = 1  # 考虑降低到 8 或更低如果内存问题仍然存在
epochs = 10

model = create_vit_classifier(input_shape, num_layers, embed_dim, num_heads, mlp_dim, num_classes, dropout_rate)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

input_img_paths = ["dataset/deraining/sidd/train/input/1.png"]
target_img_paths =  ["dataset/deraining/sidd/train/target/1.png"]

train_dataset = prepare_dataset(input_img_paths, target_img_paths, batch_size)

try:
    # 训练模型
    model.fit(train_dataset, epochs=epochs)
except Exception as e:
    print("Error occurred:", e)

# 保存整个模型
model.save('model.tf', save_format='tf')  # 保存为 TensorFlow SavedModel 格式，推荐的方式
print("Model saved to ./model.tf")
