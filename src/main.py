# main.py
from utils.utils import load_images_from_directory
import os
from layers.conv2d import Conv2D
from utils.utils import im2col  # すでに定義済みのもの
import numpy as np
from layers.MaxPool import MaxPool 
from layers.AvePool import AvePool  

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

train_dir = os.path.join(PARENT_DIR, "train")
test_dir = os.path.join(PARENT_DIR, "test")

X_train, y_train = load_images_from_directory(train_dir)
X_test, y_test = load_images_from_directory(test_dir)
"""
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)




# テスト用ランダム入力
np.random.seed(0)
x = np.random.rand(2, 3, 7, 7)  # (N=2, C_in=3, H=7, W=7)

# 重みとバイアス（out_channels=4, in_channels=3, kernel=3x3）
W = np.random.randn(4, 3, 3, 3) * 0.01
b = np.random.randn(4) * 0.01

# レイヤー定義
conv = Conv2D(W=W, b=b, stride=1, pad=1)

# フォワード実行
y = conv.forward(x)

# 結果確認
print("入力 shape:", x.shape)
print("出力 shape:", y.shape)
print("出力の一部:\n", y[0, 0, :3, :3])


# MaxPoolのテスト
pool = MaxPool(pool_h=2, pool_w=2, stride=2, pad=0)
y_pool = pool.forward(x)      
print("MaxPool 出力 shape:", y_pool.shape)
# AvePoolのテスト
ave_pool = AvePool(pool_h=2, pool_w=2, stride=2, pad=0)
y_ave_pool = ave_pool.forward(x)
print("AvePool 出力 shape:", y_ave_pool.shape)
"""

def main():
    # シンプルな4x4画像を2枚（バッチ2）
    x = np.array([
        [[[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9,10,11,12],
          [13,14,15,16]]],
        
        [[[16,15,14,13],
          [12,11,10, 9],
          [ 8, 7, 6, 5],
          [ 4, 3, 2, 1]]]
    ], dtype=np.float32)  # shape: (2, 1, 4, 4)

    print("入力画像（1枚目）:")
    print(x[0, 0])

    # プーリング層の定義（kernel=2, stride=2）
    avg_pool = AvePool(pool_h=2, pool_w=2, stride=2, pad=0)
    max_pool = MaxPool(pool_h=2, pool_w=2, stride=2, pad=0)

    # フォワードパス
    out_avg = avg_pool.forward(x)
    out_max = max_pool.forward(x)

    print("\n[Average Pooling] 出力（1枚目）:")
    print(out_avg[0, 0])

    print("\n[Max Pooling] 出力（1枚目）:")
    print(out_max[0, 0])

if __name__ == "__main__":
    main()