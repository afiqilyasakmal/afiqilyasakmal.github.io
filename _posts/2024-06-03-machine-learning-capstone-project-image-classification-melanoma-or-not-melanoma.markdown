---
layout: post
title:  "Melanoma atau Bukan Melanoma? Deteksi Kanker Kulit menggunakan Machine Learning"
date:   2024-06-03 20:54:02 +0700
---

**Untuk melihat dokumentasi dataset beserta kode yang digunakan, klik [repository GitHub ini](https://github.com/afiqilyasakmal/skin-cancer-detection).**

* toc
{:toc}

## Topik Klasifikasi
Permasalahan klasifikasi kanker kulit (melanoma) merupakan salah satu aplikasi *machine learning* di bidang kesehatan yang memiliki tujuan untuk membedakan mana citra digital yang merupakan kanker kulit (melanoma), atau bukan kanker kulit (bukan melanoma). Permasalahan klasifikasi citra ini melibatkan penggunaan teknik pada *machine learning* dan pengolahan citra. Tujuan akhir dari klasifikasi melanoma ini adalah untuk mengembangkan model yang dapat dengan secara akurat mengenali mana citra yang merupakan kanker kulit dan bukan.

Pada kesempatan kali ini, saya akan membangun arsitektur *convolutional neural network* menggunakan **PyTorch**.

## Data Preprocessing
Sebelum mendefinisikan arsitektur *neural network*, saya melakukan beberapa *data preprocessing* terhadap dataset yang dilakukan.
### 1. Me-*load* dataset
Pertama, dataset dikelompokkan menjadi 2 kelas: `melanoma` dan `not_melanoma`. Hal ini direpresentasikan dengan folder "melanoma" dan "not_melanoma". Folder tersebut berisi dataset citra yang sesuai dengan nama folder. Sehingga, pada kasus ini, nama folder menjadi nama kelas yang ingin diklasifikasi. Letakkan folder-folder ini pada level *directory* yang sama dengan Jupyter Notebook. Kemudian, setelah siap, *load* dataset.

### 2. Melakukan *data augmentation*, *resizing*, dan *normalization*
<pre>
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
</pre>
Pada tahap ini, citra di-*resize* agar ukurannya seragam menjadi 224 x 224 piksel. Setelah itu, dilakukan flip citra dalam sumbu horizontal dengan probabilitas p = 0,5 secara random. Setelah men-*transform* menjadi tensor, citra dinormalisasi dengan standar deviasi dan mean untuk tiap komponen R, G, dan B sebesar 0,5.

TLDR: Banyak sekali percobaan yang dilakukan pada *data augmentation* ini. Misalnya, saya juga mencoba melakukan *color jitter*, *random rotation*, *random resized crop*, dan sebagainya. Namun, tampaknya *preprocessing* seperti ini sudah cukup mendapatkan F1-score dan akurasi yang tinggi (dikombinasikan dengan beberapa teknik regularisasi dan *trial and error* hyperparameter tentunya).

Berikut adalah contoh citra yang telah di-*preprocess*.

![Gambar contoh citra dataset](/images/melanoma.png)

## Merancang arsitektur model
### 1. Arsitektur *Neural Network*
<pre>
# Mendefinisikan arsitektur convolutional neural network sederhana
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5) 
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 56 * 56, 10),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
</pre>
Model yang digunakan adalah CNN yang terdiri dari dua bagian: CNN layer dan Linear layer.

Untuk **CNN layer**, 
- Layer konvolusi pertama mengambil input gambar 3 channel dan menghasilkan 4 feature maps. Kernel berukuran 3x3 dengan stride 1 dan padding 1.
- Normalisasi batch untuk 4 feature maps.
- Menggunakan ReLU sebagai *activation function*.
- Max pooling layer digunakan untuk mengurangi dimensi feature maps dengan faktor 2
- Layer konvolusi kedua memiliki spesifikasi yang sama dengan layer konvolusi pertama, bedanya dia mengambil 4 channel gambar sebagai input
- Lapisan konvolusi pertama mengkonvolusi gambar dengan 4 feature maps filter 3x3, lalu dilanjutkan dengan batch normalization dan aktivasi ReLU. Lalu, lapisan pooling 2x2 digunakan untuk mengurangi size gambar.
- Lapisan konvolusi kedua mengkonvolusi hasil dari lapisan pertama dengan 4 filter 3x3, sama seperti lapisan konvolusi pertama namun sekarang beda jumlah channelnya.

Untuk **linear layer**,
- Linear layer adalah lapisan *fully connected* yang mengubah tensor dari dimensi 4*56*56 jadi 10.

### 2. Hyperparameter
<pre>
# Model yang digunakan
model = Net()

# Optimizer: pakai Adam
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Loss function: pakai Cross Entropy
criterion = nn.CrossEntropyLoss()

# Scheduler untuk menurunkan learning rate jika tidak ada perbaikan pada validation loss
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
</pre>

<pre>
# Early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
</pre>
1. **Optimizer**: Menggunakan **Adam** dengan *learning rate* = 0,0001 dan *weight decay* = 0,0001.
2. **Loss function**: Menggunakan *Cross-entropy loss*.
3. **Scheduler**: Menggunakan `ReduceLROnPlateau` untuk menurunkan *learning rate* jika *validation loss* tidak membaik dalam *patience* tiap iterasi epoch.
4. **Early stopping**: Di-set `patience` atau jumlah epoch tanpa perbaikan sebelum *training* dilakukan adalah 5 epoch. Jika *validation loss* tidak membaik selama `patience` epoch, maka *training* model akan dihentikan untuk mencegah *overfitting*.
5. **Batch size**: 32
6. **Regularisasi**: 
- Dropout dengan probabilitas 0,5
- Regularisasi L2 dilakukan melalui `weight decay` pada optimizer.

## Evaluasi dan Analisis
### 1. Loss, Accuracy, dan F1-score
![Visualisasi hasil dan evaluasi](/images/evaluasi_melanoma.png)

- Nilai akhir *training loss*: 0,27
- Nilai akhir *validation loss*: 0,27
- Nilai akhir *training accuracy*: 0,95
- Nilai akhir *validation accuracy*: 0,94
- Nilai akhir *training F1-score*: 0,95
- Nilai akhir *validation F1-score*: 0,94

Melihat nilai **training** dan **validation loss** yang makin kecil, menandakan model yang dibangun belajar dengan baik dan menjadi lebih akurat. Nilai metrik yang sangat baik ini, baik *accuracy* dan *F1-score*, menandakan model dapat menggeneralisir dengan baik pada data *validation*. Selain itu, kita bisa katakan bahwa keseimbangan antara *precision* dan *recall* yang bagus mengindikasikan model ini memiliki performa yang *robust*.

Melihat bahwa **training loss** dan **validation loss** serta **accuracy** dan **F1-score** juga menurun secara bersamaan, menunjukkan bahwa model tersebut tidak mengalami *overfitting*, karena performanya baik pada set data pelatihan dan validasi. Akurasi validasi yang tinggi dan skor F1 seiring bertambahnya *epoch* menunjukkan bahwa model tersebut dapat menggeneralisir data dengan baik pada data baru.

### 2. Confusion matrix
![Confusion matrix](/images/confusion-matrix-melanoma.png)

Pada permasalahan deteksi kanker kulit, kita ingin meminimalkan **false negative** dan **false positive** sebab hal kemunculan kasus tersebut bisa berakibat fatal terhadap pasien yang didiagnosis mengalami kanker kulit. Oleh karena itu, metrik yang digunakan untuk mengukur ini adalah **recall** dan **specificity**. Bila dihitung, berdasarkan **confusion matrix** di atas, didapatkan nilai **recall** sebesar **95%** dan nilai **specificity** sebesar **93%**.

## Kesimpulan
Secara keseluruhan, performa model yang sangat baik. Performa yang stabil antara **training** dan **validation** mengindikasikan bahwa model memiliki kemampuan generalisasi yang baik. Model juga dapat memperoleh nilai **recall** dan **specificity** yang sangat baik sehingga model ini dapat meminimalisir kejadian *false negative* dan *false positive*.