import tensorflow as tf, matplotlib.pyplot as plt

(x_train,),(x_test,)=tf.keras.datasets.mnist.load_data()
x_train=x_train[...,None]/255.0
x_test=x_test[...,None]/255.0
blur=lambda x: tf.image.resize(tf.image.resize(x,(14,14)),(28,28))
x_train_blur=blur(x_train)
x_test_blur=blur(x_test)

model=tf.keras.Sequential([
    tf.keras.layers.Input((28,28,1)),
    tf.keras.layers.Conv2D(32,3,activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,3,activation="relu",padding="same"),
    tf.keras.layers.Conv2DTranspose(32,3,2,activation="relu",padding="same"),
    tf.keras.layers.Conv2D(1,1,activation="relu",padding="same")
])

model.compile(optimizer="adam",loss="binary_crossentropy")
model.fit(x_train_blur,x_train,epochs=2,batch_size=64)

preds=model.predict(x_test_blur[:5])
for i in range(5):
  imgs=[x_test_blur[i],x_test[i],preds[i]]
  tit=["Blurred","Original","Predicted/Deblurred"]
  for j in range(3):
    plt.subplot(1,3,j+1)
    plt.imshow(tf.squeeze(imgs[j]))
    plt.title(tit[j])
  plt.show()
