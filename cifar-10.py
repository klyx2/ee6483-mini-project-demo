import os, math, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ========= 基本信息 & 复现性 =========
print(f"TensorFlow Version: {tf.__version__}")
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# 目录
OUTDIR   = "res/cifar10_baseline"
MODELDIR = "models"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

# ========= 1) 常量 =========
IMG_SIZE   = (32, 32) #CIFAR-10原始分辨率：32x32x3
BATCH_SIZE = 32 #显存吃紧可降到16/8
EPOCHS     = 100

# ========= 2) 数据加载与预处理 =========
print("Loading Cifar-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.squeeze().astype("int32")
y_test  = y_test.squeeze().astype("int32")
print(f"Training data shape: {x_train.shape}") #应为(50000, 32, 32, 3)
print(f"Test data shape: {x_test.shape}") #应为(10000, 32, 32, 3)

print("Normalizing data to [0,1] ...") #归一化
x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0

# --- 划分训练/验证（避免用 test 做早停） ---
val_ratio = 0.1
n_val = int(len(x_train) * val_ratio)
perm  = np.random.permutation(len(x_train)) #打乱
x_val, y_val = x_train[perm[:n_val]], y_train[perm[:n_val]]
x_tr , y_tr  = x_train[perm[n_val:]], y_train[perm[n_val:]]
print(f"Train: {x_tr.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

# (可选) 保存几张样本图
try:
    plt.figure(figsize=(6,6))
    for i in range(9):
        plt.subplot(3,3,i+1); plt.axis("off"); plt.imshow(x_tr[i])
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "cifar10_samples.png"), dpi=160)
    plt.close()
    print("Saved sample images.")
except Exception as e:
    print("Skip sample figure:", e)

# ========= 3) 数据增强（仅训练集） =========
train_datagen = ImageDataGenerator(
    rotation_range=15, #旋转
    width_shift_range=0.1, #左右平移
    height_shift_range=0.1, #上下平移
    zoom_range=0.1, #缩放
    horizontal_flip=True, #水平翻转
    fill_mode='nearest' #像素填充
)
# 注意：我们已归一化到 [0,1]，不需要 rescale；也不需要 datagen.fit()

train_flow = train_datagen.flow(x_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True)
steps_per_epoch = math.ceil(len(x_tr) / BATCH_SIZE)

# 验证集直接用数组（不用 generator），这样就不需要 validation_steps
val_data = (x_val, y_val)

# ========= 4) 模型 =========
print("\nBuilding merged CNN model...")
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.summary()

# ========= 5) 编译 =========
print("\nCompiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ========= 6) 回调 =========
early_stopper = EarlyStopping(
    monitor='val_loss', patience=7, verbose=1, restore_best_weights=True
)
ckpt_path = os.path.join(MODELDIR, 'best_cifar10_model.keras')
model_checkpoint = ModelCheckpoint(
    ckpt_path, monitor='val_loss', save_best_only=True, verbose=1
)

# ========= 7) 训练 =========
print("\nStarting training with augmentation...")
history = model.fit(
    train_flow,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[early_stopper, model_checkpoint],
    verbose=2
)
print("Training complete.")

# ========= 8) 评估（仅最终在 test 上） =========
print("Evaluating best model on TEST...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy (Best Model): {test_acc:.4f}")

# 保存指标（含 acc；macro-F1 需要 sklearn，可选）
import pandas as pd
pd.DataFrame({'split':['test'],'acc':[float(test_acc)]}).to_csv(
    os.path.join(OUTDIR,'test_metrics.csv'), index=False
)

# ========= 9) 训练曲线 =========
print("Saving training plots...")
acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
epochs_range = range(len(acc))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.plot(epochs_range, acc, label='Train')
plt.plot(epochs_range, val_acc, label='Val'); plt.title('Accuracy'); plt.legend()
plt.subplot(1,2,2); plt.plot(epochs_range, loss, label='Train')
plt.plot(epochs_range, val_loss, label='Val'); plt.title('Loss'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'cifar10_training_plots.png'), dpi=200)
plt.close()
print("Saved plots to", OUTDIR)

print("Model saved to", ckpt_path)
print("All done.")