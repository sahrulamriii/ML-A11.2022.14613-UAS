# Proyek Machine Learning RockPaperScissors

Proyek ini mengimplementasikan model pembelajaran mesin untuk mengklasifikasikan gambar batu, kertas, dan gunting menggunakan TensorFlow dan Keras. <br/>


## Dokumentasi
<img src="image/Screenshot 2024-07-22 170136.png" align="center" alt="RockPaperScissors Image" width="800" height="425">

<br/>

## Instalasi

1. **Clone repositori:**

    ```sh
    https://github.com/sahrulamriii/project-machine-learning.git
    cd rockpaperscissors
    ```

2. **Instal pustaka yang diperlukan:**

    ```sh
    pip install tensorflow keras matplotlib numpy
    ```

## Deskripsi Proyek

Proyek ini terdiri dari beberapa langkah untuk membangun dan melatih model yang mengklasifikasikan gambar batu, kertas, dan gunting. Untuk mengenali objek pada citra batu, kertas, dan gunting diperlukan langkah klasifikasi. Ada beberapa metode klasifikasi gambar yang populer untuk klasifikasi gambar, termasuk K Nearest Neighbors (KNN), Support Vector Machine (SVM), dan Deep Learning (DL). Salah satu metode yang cukup terkenal dalam pengklasifikasian gambar (imagge) adalah dengan menggunakan Convoluonal neural Network (CNN)
CNN merupakan jaringan saraf tiruan yang dirancang buat memproses foto. Jaringan ini terdiri dari lapisan- lapisan yang silih tersambung, di mana susunan awal mengetahui pola foto serta menciptakan peta fitur. Peta fitur tersebut setelah itu diolah lebih lanjut oleh lapisan- lapisan selanjutnya buat menciptakan prediksi akhir. CNN sangat sesuai buat bermacam tugas pengolahan foto, semacam klasifikasi foto, deteksi objek, serta segmentasi foto.


Berikut penjelasan dari setiap langkahnya.

## Latar Belakang Masalah
Pada zaman modern seperti sekarang ini, permainan tradisional mulai jarang dimainkan oleh anak-anak. Sebuah model klasifikasi gambar dapat memberi sensasi dan pengalaman baru, terutama bagi anak era 2000-an, meskipun ini hanyalah sebuah model pendeteksi dan klasifikasi gambar, tanpa ada permainan adu suit, hal ini cukup baik untuk bahan latihan dan belajar machine learning, khususnya bagi para pemula.

### 1. Impor Pustaka

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from google.colab import files
from keras.preprocessing import image
import matplotlib.pyplot as plt
```
Blok kode tersebut mengimpor pustaka yang diperlukan untuk proyek, termasuk TensorFlow untuk membangun model, os dan numpy untuk manipulasi file dan array, serta matplotlib untuk visualisasi.

# 2. Mengunduh dan Mengekstrak Dataset

# Clone dataset
```python
!wget --no-check-certificate \
  https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip \
  -O /tmp/rockpaperscissors.zip

# Ekstrak file zip

!unzip /tmp/rockpaperscissors.zip -d /tmp
```
Datasets ini berasal dari dicoding .Pada proyek ini akan diterapkan program jaringan saraf tiruan menggunakan
TensorFlow. Program ini harus mampu mengenali bentuk tangan yang membentuk gunting, batu, atau kertas
menggunakan dataset tersebut yang di nantinya terdiri dari data training memiliki 1314 sampel, dan data validasi
sebanyak 874 sampel. Didalam datasets tersebut terdapat 3 atribut data.

```python
#3. Persiapan Data

# Path dataset
base_dir = '/tmp/rockpaperscissors/'

# Membuat direktori untuk data generator

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
```
# Membuat direktori untuk setiap kelas
```python
classes = ['rock', 'paper', 'scissors']
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, cls), exist_ok=True)
```
pada bagian ini membuat direktori untuk data pelatihan dan validasi, serta subdirektori untuk setiap kelas (batu, kertas, gunting).

#4. Memisahkan Data Menjadi Set Pelatihan dan Validasi
```python
# Memindahkan gambar ke direktori masing-masing
for cls in classes:
    source_dir = os.path.join(base_dir, 'rps-cv-images', cls)
    for img in os.listdir(source_dir):
        if np.random.rand(1) < 0.6:
            os.replace(os.path.join(source_dir, img), os.path.join(train_dir, cls, img))
        else:
            os.replace(os.path.join(source_dir, img), os.path.join(validation_dir, cls, img))
```
Blok kode tersebut berperan untuk memisahkan gambar menjadi set pelatihan dan validasi dengan rasio 60:40.

#5. Augmentasi Data
```python
# Augmentasi gambar
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
```
Blok kode tersebut membuat generator augmentasi data untuk meningkatkan variasi gambar pelatihan melalui transformasi seperti rotasi, pergeseran, dan flipping.

#6. Generator Data
```python
# Pembagian data train dan validation
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
```
#7. Membangun Model
```python
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])
```
Proses pembangunan model dimulai
Blok kode ini mendefinisikan arsitektur model CNN dengan beberapa lapisan konvolusi, pooling, dan dense untuk mengklasifikasikan gambar menjadi tiga kelas

#8. Mengompilasi Model
```python
# Kompilasi model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])
```
Blok kode tersebut mengompilasi model dengan loss function categorical crossentropy, optimizer Adam, dan metrik akurasi.

#9. Early Stopping
```python
# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
```
Blok kode tersebut berfungsi mendefinisikan callback early stopping untuk menghentikan pelatihan ketika performa pada data validasi tidak meningkat.

#10. Melatih Model
```python
# Pelatihan model
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      callbacks=[early_stop],
      verbose=1)
```
Blok kode tersebut berperan melatih model menggunakan data pelatihan dan validasi dengan 10 epoch. Callback early stopping akan menghentikan pelatihan jika performa pada data validasi tidak meningkat selama 3 epoch berturut-turut. Hasil pelatihan disimpan dalam variabel history.
#11. Mengevaluasi Model
```python
# Evaluasi model
_, acc = model.evaluate(validation_generator, verbose=0)
print("Akurasi model: {:.2f}%".format(acc * 100))
```
#12. Memvisualisasikan Hasil Pelatihan
```python
# Fungsi untuk memplot loss dan akurasi
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Memanggil fungsi plot_history setelah pelatihan selesai
plot_history(history)
```
Blok kode tersebut mendefinisikan dan memanggil fungsi plot_history untuk memvisualisasikan akurasi dan loss selama pelatihan dan validasi.
#13. Memprediksi Gambar yang Diunggah
```python
# Fungsi untuk memprediksi gambar yang diunggah
def predict_image():
    # Mengunggah gambar
    uploaded = files.upload()

    # Memanggil fungsi untuk memprediksi gambar yang diunggah
    for filename in uploaded.keys():
        img = image.load_img(filename, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)

        # Menampilkan gambar
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        # Mencetak hasil prediksi
        if np.argmax(classes) == 0:
            print('Predicted: rock')
        elif np.argmax(classes) == 1:
            print('Predicted: paper')
        else:
            print('Predicted: scissors')

# Memanggil fungsi untuk memprediksi gambar
predict_image()
```
Blok kode tersebut berfungsi untuk mendefinisikan fungsi predict_image untuk memprediksi kelas gambar yang diunggah oleh pengguna. Gambar akan ditampilkan bersama dengan hasil prediksi.

## Struktur Proyek
 - rockpaperscissors/
      - `train/`
      - `rock/`
      - `paper/`
      - `scissors/`
 - val/
      - `rock/`
      - `paper/`
      - `scissors/`
  
## Kontribusi
Kontribusi sangat diterima. Silakan buka issue atau kirim pull request.
