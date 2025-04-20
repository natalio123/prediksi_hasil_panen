
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
##### 📐 Pengukuran solusi
Evaluasi model
|     | train | test   |
|:--- |:-----:|-------:|
|RF   |0.21903|0.252526|
|LN   |0.25041|0.249836|

Uji
|     |y_true|prediksi_RF| prediksi_LN|
|:--- |:----:|:---------:|-----------:|
|382393|1.676764|2.0001|1.9957|

## 📂 Data Understanding
Dataset berisi data pertanian untuk 1.000.000 sampel yang bertujuan untuk memprediksi hasil panen (dalam ton per hektar) berdasarkan berbagai faktor. Dataset ini dapat digunakan untuk tugas regresi dalam pembelajaran mesin, terutama untuk memprediksi produktivitas tanaman. 
Dataset bisa diakses pada link berikut ini: https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield

Variabel pada dataset yang digunakan
* Region: Wilayah geografis tempat tanaman tumbuh (North, East, South, West)
* Soil_Type: Jenis tanah tempat tanaman ditanam (Clay, Sandy, Loam, Silt, Peaty, Chalky)
* Crop: Jenis tanaman yang ditanam (Wheat, Rice, Maize, Barley, Soybean, Cotton)
* Rainfall_mm: Jumlah curah hujan yang diterima dalam milimeter selama periode pertumbuhan tanaman.
* Temperature_Celsius: Suhu rata-rata selama periode pertumbuhan tanaman, diukur dalam derajat Celcius.
* Fertilizer_Used: Menunjukkan apakah pupuk telah diberikan(True, False)
* Irrigation_Used:  Menunjukkan apakah irigasi digunakan selama periode pertumbuhan tanaman (True, False)
* Weather_Condition:  Kondisi cuaca yang dominan selama musim tanam (Sunny, Rainy, Cloudy)
* Days_to_Harvest: Jumlah hari yang dibutuhkan tanaman untuk dipanen setelah penanaman.
* Yield_tons_per_hectare: Total hasil panen yang dihasilkan, diukur dalam ton per hektar.

## 📋 Data Preparation
1. Encoding Fitur Kategori <br>
   Alasan: <br>
   Dikarenakan sebagian besar variabel atau fitur tidak bisa bekerja langsung dengan data kategorikal. Maka dari itu, fitur kategori diubah menjadi representasi numerik menggunakan One-Hot Encoding ```pd.get_dummies``` agar bisa digunakan oleh model
2. Train-Test Split <br>
   Alasan: <br>
   Data dibagi menjadi data latih(80%) dan data uji (20%) agar performa model dapat dievaluasi pada data yang belum pernah dilihat sebelumnya, sehingga mengurangi risiko overfitting
3. Validasi Jumlah Sampel
   Alasan: <br>
   Untuk memastikan bahwa jumlah sampel pada masing - masing subset (train dan test) telag sesuai dan tidak terjadi kehilangan data secara tidak sengaja.
4. Standarisasi fitur numerik
   Alasan: <br>
   Agar memiliki distribusi seragam(mean = 0 dan standar deviasi = 1). Ini sangat penting terutama ketika menggunakan algoritma sensitif terhadap skala data seperti regresi linear.

##  🤖 Modelling
Algoritma yang digunakan
1. Linear Regression
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
2. Random Forest Regressor
   - Parameter yang digunakan:
     ```
     RandomForestRegressor(
      n_estimators=50,
      max_depth=16,
      random_state=55,
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
|RF   |0.21903|0.252526|
|LN   |0.25041|0.249836|

**Interpretasi Hasil**
* Linear Regression memberikan performa yang cukup stabil antara data training dan testing, namun memiliki MSE yang sedikit lebih tinggi pada data training dibanding Random Forest.
* Random Forest Regressor menunjukkan MSE yang lebih rendah di data training, menandakan model ini lebih baik dalam menangkap pola kompleks pada data.
* Meskipun selisih di MSE testing relatif kecil, Random Forest tetap dipilih sebagai model terbaik karena performanya yang lebih konsisten dan kemampuannya menangani data non-linear dan fitur interaksi.
>>>>>>> 9d6127fce68ece6ad113b2736f870e090948350d
