import tensorflow as tf

model = tf.keras.models.load_model("sign_model.h5")

model.summary()

print("\nOutput shape:", model.output_shape)
print("Number of classes:", model.output_shape[-1])