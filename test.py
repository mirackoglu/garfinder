import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import load_img, img_to_array

# Modeli yükle
model = keras.models.load_model("garbage_classifier_transferr2.keras")

# Sınıf isimlerini belirle
class_names = ['cam', 'genelcop', 'kagit', 'karton', 'metal', 'plastik']  # Kendi class isimlerinle değiştir!

# Test edilecek fotoğraf yolu
img_path = "test3.jpg"

# Görseli yükle ve ön işle
img_size = (224, 224)    # eğitimde kullandığın img_size ile aynı!
img = load_img(img_path, target_size=img_size)
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Tahmin yap
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_class_index]
confidence = predictions[0][predicted_class_index]

print(f"Tahmin Edilen Sınıf: {predicted_class} ({confidence * 100:.2f}%)")

# Fotoğrafı göster
plt.imshow(img)
plt.title(f"Tahmin: {predicted_class} ({confidence * 100:.2f}%)")
plt.axis("off")
plt.show()
