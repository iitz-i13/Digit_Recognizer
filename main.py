import tensorflow as tf
import gradio as gr

tf.get_logger().setLevel('ERROR')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# モデル生成
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 畳み込み層を追加
  tf.keras.layers.MaxPooling2D((2, 2)),  # プーリング層を追加
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),  # ドロップアウト層を追加
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

def recognize_digit(img):
    img = img.reshape(1, 28, 28)
    prediction = model.predict(img).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

label = gr.outputs.Label(num_top_classes=4)
interface = gr.Interface(fn=recognize_digit, inputs='sketchpad', outputs=label, live=False, title='Digit Recognizer')
interface.launch(share=True)
