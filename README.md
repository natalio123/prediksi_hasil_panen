
# Laporan Proyek Machine Learning - Natalio Michael Tumuahi

## :dart: Domain Proyek
Pertanian merupakan sektor vital dalam perekonomian banyak negara, terutama negara agraris seperti Indonesia. Salah satu tantangan utama yang dihadapi oleh petani dan pelaku agribisnis adalah ketidakpastian hasil panen, yang dapat dipengaruhi oleh banyak faktor, seperti kondisi cuaca, jenis tanah, metode irigasi, penggunaan pupuk, serta waktu tanam dan panen.

Perubahan iklim yang tidak menentu membuat prediksi hasil panen menjadi semakin penting, baik untuk pengambilan keputusan di tingkat petani (misalnya dalam penjadwalan tanam atau alokasi sumber daya) maupun di tingkat pemerintah (misalnya dalam perencanaan ketahanan pangan dan distribusi hasil pertanian).

Dengan kemajuan teknologi dan ketersediaan data pertanian yang semakin melimpah, machine learning kini dapat dimanfaatkan untuk membangun model prediktif yang mampu memperkirakan hasil panen berdasarkan data historis dan kondisi lingkungan. Ini memberikan peluang besar dalam meningkatkan efisiensi dan produktivitas pertanian secara keseluruhan.

## 💹Business Understanding
### 🎗️Problem Statements
1. Bagaimana cara memprediksi hasil panen secara akurat berdasarkan kondisi cuaca dan faktor pertanian lainnya?
2. Apa model terbaik yang dapat digunakan untuk memprediksi hasil panen dengan performa tinggi dan error minimal?
3. Bagaimana membandingkan efektivitas model machine learning seperti Random Forest dan Linear Regression dalam konteks prediksi hasil panen?

### ⛳Goals
1. Mengembangkan model prediksi hasil panen berbasis data cuaca dan agronomis agar dapat memperkirakan hasil dalam satuan ton/hektar
2. Mengevaluasi dan membandingkan performa model regresi menggunakan metrik seperti **Mean Squared Error (MSE)** dan menghindari overfitting atau underfitting
3. Menentukan model yang optimal untuk digunakan dalam skenario nyata di lapangan (misal oleh petani)

#### 💡Solution Statements
1. Membangun dua model regresi:
   - Random Forest Regressor
   - Linear Regression
2. Melakukan evaluasi performa model menggunakan metrik **Mean Squared Error (MSE)** pada data training dan testing untuk memastikan generalisasi yang baik

## 📂 Data Understanding
Dataset berisi data pertanian untuk 1.000.000 sampel yang bertujuan untuk memprediksi hasil panen (dalam ton per hektar) berdasarkan berbagai faktor. Dataset ini dapat digunakan untuk tugas regresi dalam pembelajaran mesin, terutama untuk memprediksi produktivitas tanaman. 
Dataset bisa diakses pada link berikut ini: https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield
<br><br>
**Variabel pada dataset**
* Region: Wilayah geografis tempat tanaman tumbuh (North, East, South, West)
* Soil_Type: Jenis tanah tempat tanaman ditanam (Clay, Sandy, Loam, Silt, Peaty, Chalky)
* Crop: Jenis tanaman yang ditanam (Wheat, Rice, Maize, Barley, Soybean, Cotton)
* Rainfall_mm: Jumlah curah hujan yang diterima dalam milimeter selama periode pertumbuhan tanaman.
* Temperature_Celsius: Suhu rata-rata selama periode pertumbuhan tanaman, diukur dalam derajat Celcius.
* Fertilizer_Used: Menunjukkan apakah pupuk telah diberikan(True, False)
* Irrigation_Used:  Menunjukkan apakah irigasi digunakan selama periode pertumbuhan tanaman (True, False)
* Weather_Condition:  Kondisi cuaca yang dominan selama musim tanam (Sunny, Rainy, Cloudy)
* Days_to_Harvest: Jumlah hari yang dibutuhkan tanaman untuk dipanen setelah penanaman.
* Yield_tons_per_hectare: Total hasil panen yang dihasilkan, diukur dalam ton per hektar. <br>

**Variabel dataset yang digunakan setelah dilakukan proses EDA**
* Rainfall_mm: Jumlah curah hujan yang diterima dalam milimeter selama periode pertumbuhan tanaman.
* Temperature_Celcius: Suhu rata-rata selama periode pertumbuhan tanaman, diukur dalam derajat Celcius.
* Fertilizer_Used: Menunjukkan apakah pupuk telah diberikan(True, False)
* Irrigation_Used: Menunjukkan apakah irigasi digunakan selama periode pertumbuhan tanaman (True, False)
* Yield_tons_per_hectare: Total hasil panen yang dihasilkan, diukur dalam ton per hektar. <br>

**Informasi mengenai dataset <br>**
```
   <class 'pandas.core.frame.DataFrame'>
   RangeIndex: 1000000 entries, 0 to 999999
   Data columns (total columns):
    #   Column                  Non-Null Count    Dtype  
   ---  ------                  --------------    -----  
    0   Region                  1000000 non-null  object 
    1   Soil_Type               1000000 non-null  object 
    2   Crop                    1000000 non-null  object 
    3   Rainfall_mm             1000000 non-null  float64
    4   Temperature_Celsius     1000000 non-null  float64
    5   Fertilizer_Used         1000000 non-null  bool   
    6   Irrigation_Used         1000000 non-null  bool   
    7   Weather_Condition       1000000 non-null  object 
    8   Days_to_Harvest         1000000 non-null  int64  
    9   Yield_tons_per_hectare  1000000 non-null  float64
   dtypes: bool(2), float64(3), int64(1), object(4)
   memory usage: 62.9+ MB
```
**Jumlah missing value**
|                         | 0 |
|:------------------------|--:|
|Region                   |0.0|
|Soil_Type                |0.0|
|Crop                     |0.0|
|Rainfall_mm              |0.0|
|Temperature_Celcius      |0.0|
|Fertilizer_Used          |0.0|
|Irrigation_Used          |0.0|
|Weather_Condition        |0.0|
|Days_to_Harvest          |0.0|
|Yield_tons_per_hectare   |0.0| 

dtype:float64
<br>

**Jumlah duplicate<br>**
```np.int64(0)``` <br>
Artinya tidak ada data duplikat dari file

## 📋 Data Preparation
1. Handling Missing Values <br>
   Alasan: Karena tidak ditemukan missing values pada tahap Data Understanding, maka tidak diperlukan tindakan seperti penghapusan atau imputasi data. <br>
2. Handling Duplicate Data <br>
   Alasan: Karena tidak ditemukan data yang duplikat, maka tidak dilakukan penghapusan data duplicate.
3. Handing Outlier <br>
   Alasan: Outlier pada kolom ```Yield_tons_per_hectare``` dihapus menggunakan metode Interquartile Range (IQR).Hal ini bertujuan untuk mengurangi noise dan meningkatkan kualitas data sebelum modeling. <br>
4. Encoding Fitur Kategori <br>
   Alasan: <br>
   Fitur kategorikal perlu diubah ke format numerik agar dapat digunakan dalam proses pelatihan model machine learning. Namun, setelah dilakukan analisis awal, diketahui bahwa fitur-fitur kategorikal tidak memiliki kontribusi yang signifikan terhadap variabel target (Yield_tons_per_hectare). Oleh karena itu, fitur-fitur ini tidak disertakan dalam proses pelatihan dan evaluasi model, dengan tujuan untuk menyederhanakan model dan menghindari dimensi yang tidak relevan yang dapat mempengaruhi akurasi prediksi. <br>
5. Train-Test Split <br>
   Alasan: <br>
   Data dibagi menjadi data latih(80%) dan data uji (20%) agar performa model dapat dievaluasi pada data yang belum pernah dilihat sebelumnya, sehingga mengurangi risiko overfitting
6. Validasi Jumlah Sampel
   Alasan: <br>
   Untuk memastikan bahwa jumlah sampel pada masing - masing subset (train dan test) telag sesuai dan tidak terjadi kehilangan data secara tidak sengaja.
7. Standarisasi fitur numerik
   Alasan: <br>
   Agar memiliki distribusi seragam(mean = 0 dan standar deviasi = 1). Ini sangat penting terutama ketika menggunakan algoritma sensitif terhadap skala data seperti regresi linear.

##  🤖 Modelling
Algoritma yang digunakan
1. **Linear Regression**
   - Cara Kerja <br>
     Linear Regression bekerja dengan mencari hubungan linear antara fitur-fitur input (misalnya rainfall, temperature, dll) dan target output (Yield_tons_per_hectare). Model ini menghitung koefisien untuk setiap fitur yang menunjukkan seberapa besar pengaruh fitur tersebut terhadap target. <br><br>
     Rumus Dasar Linear Regression <br><br>
        ŷ = β0 + β1*x1 + β2*x2 + ... + βn*xn <br> <br>
     dimana:
     * ŷ : hasil panen yang diprediksi
     * β0 : intercept (konstanta)
     * x1...xn : hasil fitur (contoh rainfall, temperature)
     * β1...βn : koefisien regresi untuk masing-masing fitur
     <br>
   Linear Regression digunakan sebagai baseline model karena model ini mudah diinterpretasikan dan cepat untuk dilatih. Ia membantu melihat apakah ada hubungan linear antara fitur-fitur seperti rainfall_mm, temperature_celcius, region, dll terhadap hasil panen(Yield_tons_per_hectare)<br><br>
   - Parameter:
     - Menggunakan parameter default dari ```LinearRegression()``` dari ```sklearn.linear_model```.
    - Kelebihan
      - Sederhana dan cepat diterapkan
      - Mudah diinterpretasi karena memiliki hubungan linear antar variabel
      - Cocok sebagai baseline model.
    - Kekurangan
      - Kurang efektif untuk data non-linear
      - Sangat sensitif terhadap outlier
      - Tidak mampu menangkap interaksi kompleks antar fitur.
3. Random Forest Regressor
   - Cara Kerja
     Random Forest adalah algoritma ensemble yang terdiri dari banyak decision tree. Setiap pohon dibangun dari subset data dan subset fitur yang dipilih secara acak. Setiap pohon membuat prediksi sendiri, dan hasil akhir adalah rata-rata dari semua prediksi pohon tersebut.<br>
     Proses training model mencakup:
     * Pemisahan data berdasarkan kondisi fitur
     * Pengulangan proses ini sampai mencapai kedalaman maksimum atau data tidak bisa dibagi lagi
     * Membuat banyak pohon seperti ini, lalu menggabungkan hasilnya (averaging)<br>
       <br>
Kaitannya dengan projek saya, Random Forest digunakan dalam proyek prediksi hasil panen karena kemampuannya dalam menangani hubungan non-linear antara variabel, bekerja baik dengan data numerik maupun kategorikal (yang telah diubah melalui proses encoding), serta ketahanannya terhadap outlier dan noise meskipun data telah melalui proses pembersihan. Selain itu, Random Forest tidak memerlukan banyak pra-pemrosesan seperti normalisasi, sehingga cocok untuk data pertanian yang kompleks dan bervariasi.
     
   - Parameter yang digunakan:
     ```
     RandomForestRegressor(
      n_estimators=50,
      max_depth=14,
      random_state=123,
      n_jobs=-1
     )
     ```
   - Kelebihan
     - Mampu menangkap hubungan non-linear
     - Lebih robust terhadap outlier dan noise
     - Tidak terlalu sensitif terhadap multikolinearitas
    - Kekurangan
      - Kurang dapat diinterpretasikan dibandingkan regresi linear
      - Waktu pelatihan lebih lama
      - Ukuran model bisa sangat besar jika estimatorsnya banyak.
        
Model terbaik yaitu Random Forest Regressor
Dikarenakan:
* Lebih baik dalam menagkap relasi kompleks antar fitur input yang tidak linear.
* Lebih robust terhadap outlier dan variasi data, yang umum terjadi dalam data pelatihan
* Dari segi evaluasi model, Random Forest lebih baik

## 📌 Evaluation
**Metrik evaluasi yang digunakan yaitu MSE** <br>
Mean Squared Error (MSE) <br>
MSE mengukur rata-rata kuadrat dari selisih antara nilai prediksi dan nilai aktual. Semakin kecil nilai MSE, maka semakin baik model dalam melakukan prediksi. <br>
Rumus: <br>
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²

**Hasil Evaluasi Model**
|     | train | test   |
|:--- |:-----:|-------:|
|RF   |0.22603|0.252537|
|LN   |0.25041|0.249838|

**Interpretasi Hasil**
* Linear Regression memberikan performa yang cukup stabil antara data training dan testing, namun memiliki MSE yang sedikit lebih tinggi pada data training dibanding Random Forest.
* Random Forest Regressor menunjukkan MSE yang lebih rendah di data training, menandakan model ini lebih baik dalam menangkap pola kompleks pada data.
* Meskipun selisih di MSE testing relatif kecil, Random Forest tetap dipilih sebagai model terbaik karena performanya yang lebih konsisten dan kemampuannya menangani data non-linear dan fitur interaksi.
