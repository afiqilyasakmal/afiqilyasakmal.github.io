---
layout: post
title:  "Analisis Performa Exchange-traded Fund dan Reksa Dana di Kawasan Eropa"
date:   2024-01-05 21:16:02 +0700
---

**Note: Versi slide presentasi (yang jauh lebih singkat) dapat dilihat di link ini.**

Proyek ini merupakan proyek *data science* yang bertujuan menganalisis performa tiap *exchange-traded fund* (ETF) dan reksa dana yang beroperasi di kawasan Eropa. Apa saja yang akan kita lakukan?

- Melakukan eksplorasi dan melihat *insight* dari dataset yang berukuran besar ini (lebih dari 20.000 baris!)
- Melakukan klasifikasi terhadap *rating* tiap ETF dan reksa dana
- Memprediksi proyeksi pertumbuhan laba jangka panjang
- Melakukan *clustering* untuk menganalisis perilaku jenis-jenis manajer investasi

Proyek ini menggunakan bahasa Python beserta pustakanya seperti `pandas`, `NumPy`, `seaborn`, dan `scikit-learn` untuk melakukan EDA dan membangun model *machine learning*. Seluruh dokumentasi proyek ini ada di repository [**GitHub ini**](https://github.com/afiqilyasakmal/european-investment-management){:target="_blank"}.

## Daftar Isi

* toc
{:toc}

## *Data understanding*

Dataset yang kita gunakan kali ini berasal dari Morningstar, Inc., yaitu perusahaan jasa keuangan yang berasal dari Chicago, Amerika Serikat. Morningstar menyediakan layanan riset investasi dan manajemen investasi. Dataset dapat dilihat pada [***link*** **ini**](https://github.com/afiqilyasakmal/european-investment-management/blob/main/data/european-investment-management-train.csv){:target="_blank"}.

**Ukuran dataset** adalah (22420 x 117). Artinya, ada 22420 baris dan 117 fitur. Karena ukuran dataset cukup besar, ringkasan statistika deskriptif tidak akan ditampilkan di sini. Anda bisa melihat *notebook* untuk bagian ini di bagian bawah subbab ini. 

Beberapa fitur penting pada dataset ini adalah:
- `category`: kategori reksa dana
- `rating`: penilaian yang diberikan Morningstar
- `dividend_frequency`: periode pembagian dividen
- `equity_style`: gaya pengelolaan investasi
- `equity_size`: golongan nilai ekuitas
- `equity_size_score`: skor nilai ekuitas
- `price_book_ratio`: rasio harga dengan nilai buku
- `price_sales_ratio`: rasio harga dengan pendapatan
- `price_cash_flow_ratio`: rasio harga dengan arus kas
- `dividend_yield_factor`: % dividen dibandingkan dengan harga
- `long_term_projected_earnings_growth`: % proyeksi pertumbuhan laba jangka panjang
- `historical_earnings_growth`: % pertumbuhan laba secara historis
- `sales_growth`: % pertumbuhan pendapatan
- `book_value_growth`: % pertumbuhan nilai buku
- `roa`: % laba dibandingkan dengan nilai asset
- `roe`: % laba dibandingkan dengan nilai ekuitas
- `roic`: % laba dibandingkan dengan nilai yang diinvestasikan

Untuk masalah **duplikasi data**, tidak ditemukan duplikat pada dataset ini. Bahkan, setelah kolom `ticker` dihapus (yang merupakan kode unik yang merepresentasikan reksa dana), duplikasi data juga tidak ditemukan. Artinya, dataset yang kita miliki benar-benar tidak memiliki data duplikat.

Selanjutnya, akan dicek ***missing values*** pada dataset. Terdapat 20 fitur yang memiliki *missing value* di atas 25%! Bahkan, untuk fitur `modified_duration` dan `effective maturity`, persentase *missing value*-nya lebih dari 90%. Untuk menangani hal ini, kita akan bahas di bagian pembuatan model *machine learning*. 

![Visualisasi missing value pada dataset](/images/01-missing-value-eim.png)
<figcaption style="text-align:center">Visualisasi missing value dataset.</figcaption>
<br>

Terakhir, kita akan cek ***outlier*** yang terdapat pada dataset. Berikut adalah top 10 kolom dengan jumlah outlier terbanyak:

![Visualisasi outlier pada dataset](/images/02-outlier-eim.png)
<figcaption style="text-align:center">Visualisasi outlier dataset.</figcaption>
<br>

Untuk saat ini, kita akan biarkan *outlier* apa adanya. Jika ingin melihat jumlah dan persentase *outlier* tiap kolom, silakan lihat *notebook* di bawah ini.

*Notebook* bagian ***data understanding*** dapat dilihat di [***link*** **ini**](https://github.com/afiqilyasakmal/european-investment-management/blob/main/notebook/01_data_understanding.ipynb){:target="_blank"}.

## *Exploratory data analysis*

Setelah "mengintip" dataset, sekarang kita akan mencoba melakukan eksplorasi sederhana. Kita akan melihat dan mempelajari beberapa *insight* menarik yang bisa kita ambil. Notebook untuk bagian ini bisa dilihat pada [***link*** **ini**](https://github.com/afiqilyasakmal/european-investment-management/blob/main/notebook/02_exploratory_data_analysis.ipynb){:target="_blank"}.

### Manajemen investasi di sektor teknologi memiliki *fund return* paling tinggi dibandingkan sektor lainnya

![Visualisasi beberapa fund return terpilih dari tahun 2015-2020](/images/03-fund-return-historical.png)
<figcaption style="text-align:center">Visualisasi beberapa fund return terpilih dari tahun 2015-2020.</figcaption>
<br>
Visualisasi di atas mengilustrasikan data historis *fund return* beberapa kategori manajemen investasi. Kategori dipilih berdasarkan ***return on investment capital (ROIC)***, ***return on asset (ROA)***, dan ***return on equity (ROE)*** tertinggi, top 25%, top 50%, top 75%, hingga yang terendah. Nampak bahwa semakin tinggi ROIC, ROA, dan ROE, maka terdapat kecenderungan bahwa *fund return* juga akan relatif tinggi juga, **walau tidak selalu demikian!**

Manajemen investasi yang memiliki *fund return* tertinggi secara rata-rata adalah manajemen investasi yang berada dalam sektor teknologi. Hal ini mungkin terjadi karena perkembangan teknologi yang semakin pesat sehingga kinerja perusahaan di bidang teknologi juga makin bagus. Walau demikian, dapat dilihat bahwa pada tahun 2018 kuartal 3, semua sektor mengalami penurunan. Penurunan *fund return* paling dalam justru juga datang dari manajemen investasi berkategori teknologi.

### Top 20 kategori manajemen investasi berdasarkan dana kelolaan

![Visualisasi dana kelolaan top 20 kategori manajemen investasi](/images/04-top-20-manajemen-investasi.png)
<figcaption style="text-align:center">Visualisasi dana kelolaan top 20 kategori manajemen investasi.</figcaption>
<br>

Walaupun sektor teknologi (kategori *Sector Equity Technology*) memiliki *return on investment capital (ROIC)*, *return on asset (ROA)*, *return on equity (ROE)*, dan *fund return* yang tinggi, tidak serta merta menjadikan kategori tersebut memiliki dana kelolaan yang besar juga. Bahkan, kategori tersebut tidak masuk ke top 20 kategori besar dana kelolaan! Dari horizontal bar plot di atas, nampak bahwa dana kelolaan terbesar dikelola oleh manajemen investasi dengan kategori di sektor properti.

### Hubungan antara *management fees* dengan pertumbuhan return investasi

*Management fees* adalah biaya yang dibayarkan kepada para profesional yang dipercaya untuk mengelola investasi atas nama klien. Biasanya, biaya ini ditentukan sebagai persentase dari total aset yang dikelola.

![Heatmap korelasi antara management fee dengan fund return tiap tahunnya](/images/05-heatmap-management-fee-dgn-fund-return.png)
<figcaption style="text-align:center">Heatmap korelasi antara management fee dengan fund return tiap tahunnya.</figcaption>
<br>

Berdasarkan korelasi pearson yang ditampilkan *heatmap* di atas, nampak tidak ada korelasi yang kuat antara *management fee* dan *fund return* per tahun. Nilai korelasi berkisar antara -0.0036 hingga 0.022, yang menunjukkan bahwa hubungan antara kedua variabel ini lemah dan tidak signifikan.

Dengan kata lain, *fund return* tidak tampak dipengaruhi secara signifikan oleh *management fee*. Artinya, investor mungkin tidak perlu terlalu khawatir tentang dampak biaya manajemen terhadap pengembalian investasi mereka.

![Heatmap korelasi antara management fee dengan fund return tiap tahunnya](/images/06-subplot-fundreturn-mgmtfee.png)
<figcaption style="text-align:center">Subplot antara management fee dengan fund return tiap tahunnya.</figcaption>
<br>

Berdasarkan subplot yang telah dibuat, distribusi titik-titik pada plot bersifat terdispersi atau tersebar, dan hal ini dapat mengindikasikan bahwa korelasi antara *management fee* dan *fund return* cenderung rendah. Menunjukkan bahwa **tidak ada hubungan linear** antara *management fee* dan *fund return*. Ini berarti bahwa perubahan dalam Management Fee tidak secara konsisten diikuti oleh perubahan dalam Fund Return dalam arah tertentu.

Ini menegaskan kesimpulan sebelumnya bahwa *management fee* dan *fund return* tampaknya tidak memiliki korelasi yang signifikan. Oleh karena itu, investor mungkin ingin mempertimbangkan faktor lain selain *management fee* ketika membuat keputusan investasi.

### Pilihan sektor investasi dari investment management berdasarkan *equity size*

Berdasarkan visualisasi di bawah ini, nampak bahwa pada equity size **small**, pilihan sektor investasi paling tinggi ada di **sektor industri**, pada size **medium** adalah **sektor real estate**, sedangkan size **large** tertinggi adalah **sektor teknologi**. Untuk lengkapnya, dapat dilihat pada visualisasi berikut.

![Perbandingan sektor investasi dari investment management dengan equity size small](/images/07-equity-size-small.png)
<figcaption style="text-align:center">Perbandingan sektor investasi dari investment management dengan equity size small.</figcaption>
<br>

![Perbandingan sektor investasi dari investment management dengan equity size medium](/images/08-equity-size-medium.png)
<figcaption style="text-align:center">Perbandingan sektor investasi dari investment management dengan equity size medium.</figcaption>
<br>

![Perbandingan sektor investasi dari investment management dengan equity size large](/images/09-equity-size-large.png)
<figcaption style="text-align:center">Perbandingan sektor investasi dari investment management dengan equity size large.</figcaption>
<br>

## Model *machine learning*

Secara konvensional, ada dua paradigma metode pembelajaran mesin: *supervised learning* dan *unsupervised learning*. Untuk *supervised learning*, kita akan mencoba dua hal: mengklasifikasi fitur *rating* dan melakukan prediksi terhadap *long term projected earnings growth*. Sementara itu, untuk *unsupervised learning*, kita akan mencoba melakukan *clustering* data dan menganalisis jenis manajer investasi pada tiap kluster.

### Klasifikasi *rating*

Pertama, kita akan melalukan klasifikasi terhadap fitur `rating`. **Rating** merepresentasikan performa reksadana dengan range nilai 1-5 yang diberikan oleh [Morningstar](https://sg.morningstar.com/sg/news/120375/the-morningstar-rating-for-funds.aspx). Kita akan menerapkan beberapa model, seperti *decision tree*, *K-nearest neighbor*, *random forest*, dan *XGBoost*. Sebelum kita melakukan klasifikasi, data perlu diolah sedemikian rupa sehingga data siap dimasukkan ke dalam model. Tahap ini kita sebut sebagai *data preparation*.

Notebook untuk proses *data preparation* dan pembuatan model untuk mengklasifikasikan *rating* selengkapnya bisa dilihat di [***link*** **ini**](https://github.com/afiqilyasakmal/european-investment-management/blob/main/notebook/03_klasifikasi.ipynb){:target="_blank"}.

#### *Data preparation*

Sebelum data dimasukkan dan di-*fit* ke dalam model, kita perlu mengolahnya agar data tersebut siap pakai. Berikut adalah hal-hal yang dilakukan:
1. **Menangani data duplikat**. Pada dataset, tidak ditemukan duplikat.
2. **Missing value**. Terdapat beberapa fitur yang memiliki banyak *missing value*. Pada konteks ini, kita akan drop fitur dengan missing value lebih dari 50%. 
3. **Melakukan imputasi data**. Fitur dengan tipe data kategorikal akan diimputasi dengan nilai modus, sedangkan fitur dengan tipe data numerik akan diimputasi dengan nilai mean.
4. **Melakukan encoding**. Sebelum dimasukkan ke dalam model, fitur dengan tipe kategorikal perlu dikonveri menjadi numerik terlebih dahulu. Kali ini, kita akan melakukan encoding menggunakan metode *label encoding*.
5. ***Splitting data***. Data dibagi jadi dua untuk training dan testing, dengan proporsi data training : data testing adalah 80 : 20.
6. **Melakukan normalisasi**. Fitur-fitur yang ada dinormalisasi agar range nilainya tidak terlalu lebar sehingga tidak menutupi efek fitur yang lain.
7. **Yeo-Johnson *transform***. Data ditransformasi untuk mengurangi banyaknya outlier pada data.
8. **Feature selection**. *Feature selection* dilakukan menggunakan `SelectKBest` yang memanfaatkan uji statistik ANOVA dan memilih 50 fitur paling penting atau relevan.
9. **Imbalance handling**. Di sini, kita melakukan oversampling agar data minoritas dapat direpresentasikan secara lebih baik. Metode yang digunakan adalah Borderline-SMOTE.

Sampai di sini, data kita sudah siap untuk dimasukkan ke dalam model *machine learning*.  

#### Pembuatan model: Decision Tree, K-Nearest Neighbor, Random Forest, XGBoost

1. **Decision Tree**
<br>
Decision Tree dibuat menggunakan semua fitur. Tree yang dibuat dapat memprediksi target dengan semua metrik: akurasi, f1 score, precision, dan recall, semuanya berada di rentang 70-71%. Berdasarkan hasil ini, kita dapat menyimpulkan bahwa model decison tree memiliki performa yang cukup baik. Dengan akurasi 70% dengan tingkat recall dan precision yang mirip, dapat diasumsikan model ini dapat memprediksi kelas (rating) dengan konsisten. Artinya, tidak terlalu banyak target yang mengalami salah prediksi.
2. **K-Nearest Neighbor**
<br>
Mirip dengan Decision Tree, model K-Nearest Neighbor memiliki hasil yang mirip. Akurasi, f1 score, precision, dan recall berada di rentang 70-74%. Model memiliki kinerja yang konsisten antara presisi, recall, dan F1 score untuk setiap kelas (ditunjukkan oleh perbedaan yang kecil antara nilai macro dan micro). Kemudian, model cenderung memiliki kinerja yang lebih baik dalam hal recall daripada presisi, karena nilai recall macro lebih tinggi daripada nilai presisi macro.
3. **Random Forest** 
<br>
Berdasarkan hasil yang diperoleh, dapat disimpulkan:
- Model memiliki kinerja yang konsisten antara presisi, recall, dan skor F1 untuk setiap kelas (ditunjukkan oleh perbedaan yang kecil antara nilai makro dan mikro). 
- Model cenderung memiliki kinerja yang seimbang antara presisi dan recall, karena nilai-nilai presisi dan recall berada pada jarak yang dekat. 
- Akurasi, skor F1, presisi, dan recall model Random Forest cukup baik, menunjukkan kemampuan model untuk melakukan klasifikasi dengan baik secara keseluruhan.

4. **XGBoost**
<br>
Hasil evaluasi yang didapatkan model XGBoost sangat mirip dengan *random forest*, namun dengan hasil evaluasi yang sedikit lebih tinggi dibandingkan random forest. Artinya, karakteristik model ini mirip dengan *random forest* namun dengan 

### Prediksi nilai *long term projected earnings growth* 

Karena permasalahan memprediksi nilai *long term projected earnings growth* merupakan suatu permasalahan regresi, kita akan menerapkan beberapa model, diantaranya: *linear regression*, *ridge regression*, *lasso regression*, dan *random forest*. Sebelum kita melakukan prediksi, data perlu di-*set* sedemikian rupa sehingga data siap dimasukkan ke dalam model. Tahap ini kita sebut sebagai *data preparation*.

Notebook untuk proses *data preparation* dan pembuatan model untuk prediksi nilai *long term projected earnings growth* selengkapnya bisa dilihat di [***link*** **ini**](https://github.com/afiqilyasakmal/european-investment-management/blob/main/notebook/04_regresi.ipynb){:target="_blank"}.

#### *Data preparation*

Terdapat dua pendekatan untuk bagian ini. **Pendekatan pertama** yaitu dengan melakukan studi literatur untuk menentukan fitur mana saja yang paling baik dapat memprediksi variabel target. Hal-hal yang dilakukan adalah:

1. **Menangani data duplikat**. Pada kasus ini, tidak ditemukan data duplikat. Bagian ini dilakukan di bagian *data understanding*.
2. **Memilih fitur yang relevan**. Setelah mencari banyak referensi, fitur-fitur yang *safe* untuk dikatakan relevan adalah `roa`, `roic`, `roe`, `historical_earnings_growth`, `sales_growth`, `book_value_growth`, `price_prospective_earnings`, `price_book_ratio`, `price_sales_ratio`, `price_cash_flow_ratio`. Kita akan menggunakan semua fitur tersebut untuk di-*fit* ke model. Total fitur yang akan di-*training* ada 10, dengan variabel target adalah `long_term_projected_earnings_growth`.
3. **Membuat KDE Plot dan melihat indeks *skewness***. KDE Plot tiap fitur dibuat untuk melihat distribusinya apakah normal atau tidak. Untuk memastikannya, kita pakai indeks *skewnewss*. *Rule of thumb*-nya, jika indeks berada di antara -1 sampai dengan 1, kita katakan distribusi fitur tersebut normal. Jika begitu, kita akan melakukan imputasi data yang kosong pada fitur tersebut menggunakan *mean*. Jika tidak, artinya fitur *skew* dan akan diimputasi menggunakan nilai median.

Pada tahap ini, *label encoding* tidak diperlukan sebab semua fitur merupakan fitur numerik. Outlier dibiarkan apa adanya.

Sementara itu, pada **pendekatan kedua**, *data preprocessing* yang dilakukan adalah sebagai berikut:

1. **Handling missing values**. Kolom dengan missing value lebih dari 50% di-drop. 
2. **Standarisasi mata uang**. Terdapat kolom dengan mata uang tertentu. Hal ini tentu dapat disalahartikan karena nilai tiap mata uang beda-beda. Untuk mengatasi hal ini, semua mata uang dikonversi menjadi USD. 
3. **Fitur dengan korelasi di bawah 0.1 di-drop**. 
4. **Mengimputasi data**. Pertama, dilihat indeks *skewness*-nya terlebih dahulu. Untuk fitur dengan indeks *skewness* di antara -0.5 sampai +0.5 akan diimputasi dengan menggunakan nilai mean, sedangkan sisanya akan diimputasi dengan nilai median fitur tersebut.
5. **Menghitung korelasi fitur dengan target**. Fitur dipisahkan dengan variabel target, kemudian dicari korelasi antarfitur. Kemudian, dicari juga korelasi antara tiap fitur dengan target. Fitur dengan besar korelasi di atas 0.25 akan dianggap relevan. Tujuan utama tahap ini adalah membuang fitur yang berkorelasi antarsatu sama lain dan memilih fitur-fitur yang memiliki korelasi yang kuat dengan target. Hal ini dilakukan dengan tujuan mengurangi overfitting.
6. **Melakukan standarisasi dan feature selection menggunakan random forest**. Fitur yang diambil adalah top 15 fitur terbaik berdasarkan random forest. Dataset direduksi agar sesuai dengan fitur yang terpilih.
7. **Menerapkan *principal component analysis* (PCA)**. 

#### Pembuatan model: Linear Regression, Ridge Regression, Lasso Regression, Random Forest

Pada **pendekatan pertama**, setelah data di-*preprocess*, dataset dibagi jadi training set dan testing set dengan proporsi 20:80. Berikut adalah ringkasan hasil *fit* ke model-model yang digunakan:

<!-- Membuat grid dengan HTML -->
<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">

  <!-- Baris pertama -->
  <div style="display: flex; flex-direction: column; gap: 10px; text-align: center;">

    <!-- Menampilkan gambar pertama -->
    <div>
      <img src="/images/predict-linreg.png" alt="Gambar 1">
      <figcaption>Linear regression.</figcaption>
    </div>

    <!-- Menampilkan gambar kedua -->
    <div>
      <img src="/images/predict-lasso.png" alt="Gambar 2">
      <figcaption>Lasso regression.</figcaption>
    </div>

  </div>

  <!-- Baris kedua -->
  <div style="display: flex; flex-direction: column; gap: 10px; text-align: center;">

    <!-- Menampilkan gambar ketiga -->
    <div>
      <img src="/images/predict-ridge.png" alt="Gambar 3">
      <figcaption>Ridge regression.</figcaption>
    </div>

    <!-- Menampilkan gambar keempat -->
    <div>
      <img src="/images/predict-forest.png" alt="Gambar 4">
      <figcaption>Random forest.</figcaption>
    </div>

  </div>

</div>
<br>

Model *linear regression*, *ridge regression*, dan *lasso regression* memiliki nilai R-squared yang mirip-mirip, yakni berada di sekitar angka 0.245. Hasil terbaik diraih oleh model *random forest* dengan nilai R-squared 94.5%, *mean squared error* (MSE) 0.81, *root mean squared error* (RMSE) 0.90, *mean absolute error* (MAE) 0.24. 

Pada **pendekatan kedua**, setelah data di-*preprocess*, dataset dibagi jadi training set dan testing set dengan proporsi 20:80. Berikut adalah tabel ringkasan nilai R-squared hasil fit ke model-model yang digunakan:

<table border="1" width="100%" height="100%">
  <thead>
    <tr>
      <th>Model</th>
      <th>Basic feature selection</th>
      <th>Feature selection RF</th>
      <th>Feature selection RF + PCA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear regression</td>
      <td>0.399</td>
      <td>0.344</td>
      <td>0.301</td>
    </tr>
    <tr>
      <td>Lasso regression</td>
      <td>0.161</td>
      <td>0.161</td>
      <td>0.248</td>
    </tr>
    <tr>
      <td>Ridge regression</td>
      <td>0.399</td>
      <td>0.344</td>
      <td>0.301</td>
    </tr>
    <tr>
      <td>Random forest</td>
      <td><b style="color:blue">0.904</b></td>
      <td><b style="color:blue">0.902</b></td>
      <td><b style="color:blue">0.847</b></td>
    </tr>
  </tbody>
</table>
Keterangan:
- ***Basic feature selection*** adalah pemilihan fitur tanpa menggunakan random forest.
- ***Feature selection RF*** adalah pemilihan fitur dengan menggunakan tambahan metode random forest.
- ***Feature selection RF + PCA*** adalah pemilihan fitur dengan menggunakan tambahan metode random forest sekaligus PCA.

Secara garis besar, dapat dilihat bahwa model **random forest** memiliki kehandalan yang paling baik dibandingkan model lainnya. Untuk semua model selain **random forest**, R-squared yang didapat berada di bawah 0.40, sedangkan hasil yang menggunakan **random forest** semuanya berada di atas 0.80. Hal ini mungkin dapat terjadi karena **random forest** memiliki kemampuan yang tahan dengan outlier, di mana dalam data ini ada banyak sekali outlier.

### Clustering: Analisis jenis-jenis manajer investasi

TODO

## Kesimpulan

TODO