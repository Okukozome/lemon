import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# 配置参数
DATASET_DIR = './lemon_dataset_enhanced'
MODEL_SAVE_PATH = './models/lemon_cnn_model.h5'
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 15


def build_model(num_classes):
    """构建卷积神经网络 (CNN)"""
    model = models.Sequential([
        # 归一化层：加速模型收敛
        layers.Rescaling(1. / 255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        # 第一层卷积：提取边缘、颜色等低级特征
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),  # 池化层：降维，减少计算量

        # 第二层卷积：提取更复杂的局部纹理特征
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # 第三层卷积：提取高级语义特征
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # 展平层：将 3D 张量展平为 1D 向量
        layers.Flatten(),

        # 全连接层：整合特征
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout：随机丢弃神经元，防止过拟合

        # 输出层：分类结果 (使用 softmax 输出各类别概率)
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    # 加载数据集并自动划分训练集和验证集 (80% 训练, 20% 验证)
    print("正在加载数据集...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"检测到的类别: {class_names}")

    # 预处理优化：性能提升缓存
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 构建模型
    model = build_model(num_classes=len(class_names))
    model.summary()

    # 训练模型
    print("开始训练模型...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 保存模型
    if not os.path.exists('./models'):
        os.makedirs('./models')
    model.save(MODEL_SAVE_PATH)
    print(f"模型已成功保存至 {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()