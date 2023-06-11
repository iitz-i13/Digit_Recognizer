# Digit recognizer
手書きの数字をMNISTデータから学習したモデルを用いて数字判定するアプリ作成
gradioを使用して生成されるURLへ移動し，その数字であると思われる上位4つを表示

## Version
python 3.11.3 
tensorflow 2.12.0 
gradio 3.23.0

## 内容
モデル生成
```
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

## 使用方法
``` 
python main.py
``` 

## 注意
基本的な実装しかしてないため，精度はガバガバです