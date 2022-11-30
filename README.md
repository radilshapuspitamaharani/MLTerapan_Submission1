# MLTerapan_Submission1
# Laporan Proyek Machine Learning - Radilsha Puspita Maharani

## Domain Proyek

## Latar Belakang
Diamonds atau berlian adalah salah satu mineral karbon alami yang paling langka dan paling keras. intan telah digunakan sebagai batu permata selama berabad-abad karena jika disesuaikan dengan aspeknya, intan memiliki karakteristik 'api' karena indeks biasnya yang tinggi. Di masyarakat, berlian menjadi trend sebagai perhiasan sebagai hadiah pada momen-momen tertentu. Tidak hanya sebagai hadiah, berlian juga dipandang sebagai investasi alternatif selain emas karena nilainya dapat memberikan keuntungan dan peningkatan yang signifikan.
Dalam bidang perdagangan intan, pembeli atau investor mengalami kesulitan dalam memprediksi harga berlian karena adanya perbedaan bentuk, ukuran, dan kemurnian berlian. Banyak model dan aplikasi telah diimplementasikan untuk
memprediksi harga berlian ini di masa depan menggunakan pembelajaran mesin. Pembelajaran mesin dibagi menjadi dua kategori, yaitu diawasi dan tidak diawasi. Algoritma pembelajaran terbimbing menggunakan prinsip umum contoh
praktis untuk prediksi atau peramalan. 
Dalam hal ini, diperlukan sistem <i>predictive analytics</i> untuk menemukan model prediksi harga berlian yang paling efisien.


## Business Understanding

### Problem Statements
- Fitur apa yang paling berpengaruh terhadap harga diamonds
- Berapa harga pasar diamonds dengan karakteristik atau fitur tertentu?

### Goals
- Mengetahui fitur yang paling berkorelasi dengan harga diamonds
- Membuat model <i>machine learning</i> yang dapat memprediksi harga diamonds seakurat mungkin berdasarkan fitur-fitur yang ada

    ### Solution statements
    - Melakukan analisis data, data cleaning untuk mencari <i>missing value</i>, eksplorasi dengan visualisasi data untuk menemukan <i>outliers</i>, dan <i>preprocessing</i> data yang tepat sebelum dilakukan train atau latih
    - menggunakan model logistik regresi untuk memprediksi nilai <i>Boolean</i> (benar atau salah) 

## Data Understanding
Dataset yang digunakan pada proyek ini adalah dataset diamonds yang diundur dari link [berikut ini](https://www.kaggle.com/datasets/shivam2503/diamonds)

Dataset ini memiliki format .csv dengan total 85295 baris dan 10 kolom. Berikut informasi pada masing-masing kolom :
*   carat : Berat berlian (0.2 - 5.01)
*   cut : Kualitas potongan
*   color : Warna berlian
*   clarity : Pengukuran seberapa jelas berlian itu
*   depth : Persentase kedalaman total = z / mean(x, y) = 2 * z / (x + y) (43-79)
*   table : Lebar bagian atas intan relatif terhadap titik terlebar (43-95)
*   price : Harga(dalam dolar AS)
*   x : Panjang dalam mm (0-10,74)
*   y : Lebar dalam mm (0-58,9) 
*   z : Kedalaman dalam mm (0-31,8)

### Exploratory Data Analysis
Proses exploratory data analysis (EDA) merupakan proses investigasi awal pada data untuk melakukan analisis karakteristik, menemukan pola, anomali dan memerika asumsi pada data

## Data Preparation
Hal yang akan dilakukan dalam fase ini adalah menggabungkan data, menyeleksi data yang akan digunakan, melakukan proses transformasi data, dan membagi data menjadi data training dan test.

### Encoding Fitur Kategori
Sebelum masuk ke tahap pembagian dataset, terlebih dahulu dilakukan perubahan untuk merubah setiap nilai di dalam kolom menjadi kolom baru dan mengisinya dengan nilai biner yaitu 0 dan 1. Pada proyek ini, dilakukan perubahan pada variabel dependen(cut, color, clarity) karena fitur pada variabel tersebut merupakan fitur non-numerik yang berarti nilai pada fitur tersebut adalah kategorikal, maka akan dilakukan proses label encoding untuk mengubah fitur tersebut. Label <i>encoding</i> merupakan teknik untuk mengubah jenis data kategorikal menjadi data numerik yang dapat dipahami model. Pada proyek ini, encoding dilakukan dengan menggunakan metode <i>one-hot-encoding</i>.
<ul>Multivariate Analysis
    <li>Categorical Features</li>
    <img width="321" alt="image" src="https://user-images.githubusercontent.com/97927496/204704497-089e759f-9998-4052-8006-296fa73767f6.png">
    <li>Numerical Features</li>
    <img width="464" alt="image" src="https://user-images.githubusercontent.com/97927496/204704548-6fb8487c-82c9-4424-a96d-8ddea9069263.png">    
</ul>
<ul>Outliers dan Down Sampling
| carat | cut       | color | clarity | table | price | x    | y    | z    |
|-------|-----------|-------|---------|-------|-------|------|------|------|
| 0.23  | ideal     | E     | SI2     | 55.0  | 326   | 3.95 | 3.98 | 2.43 |
| 0.21  | Premium   | E     | SI1     | 61.0  | 326   | 3.89 | 3.84 | 2.31 |
| 0.29  | Premium   | I     | VS2     | 58.0  | 334   | 4.20 | 4.23 | 2.63 |
| 0.31  | Good      | J     | SI2     | 58.0  | 335   | 4.34 | 4.35 | 2.75 |
| 0.24  | Very Good | J     | VVS2    | 57.0  | 336   | 3.94 | 3.96 | 2.48 |
</ul>
<ul>Corelation Matrix
<img width="315" alt="image" src="https://user-images.githubusercontent.com/97927496/204705362-262e8030-82f2-422b-a64a-1cd3ed815352.png">


### Reduksi Dimensi Dengan PCA
PCA adalah teknik tanpa pengawasan karena hanya melihat fitur masukan dan tidak memperhitungkan keluaran atau variabel target
PCA dilakukan untuk mengurangi dimensi fitur masukan dari dataset dengan tetap mempertahankan semua informasi penting yang ada dalam data dengan dimensi yang dikurangi
   array([0.998, 0.002, 0.001])
Output diatas adalah 99.8% informasi pada ketiga fitur x, y, z terdapat pada PC pertama. Sedangkan sisanya, sebesar 0.2% dan 0.1% terdapat pada PC kedua dan ketiga. Pada gambar, jumlahnya menjadi >100% dikarenakan proses pembulatan (round) dalam 3 <i>decimal</i>.

### Split Dataset atau Pembagian Dataset
Untuk mengetahui kinerja model ketika dihadapkan pada data yang belum pernah dilihat sebelumnya maka perlu dilakukan pembagian dataset. Pada proyek ini dataset dibagi menjadi data latih dan data uji dengan rasio 90% untuk data latih dan 10% untuk data uji. Data latih merupakan data yang akan kita latih untuk membangun model machine learning, sedangkan data uji merupakan data yang belum pernah dilihat oleh model dan digunakan untuk melihat kinerja atau performa dari model yang dilatih. Pembagian dataset dilakukan dengan modul train_test_split dari scikit-learn. Setelah melakukan pembagian dataset, didapatkan jumlah sample pada data latih yaitu 47524 sampel dan jumlah sampel pada data uji yaitu 4753 sampel dari total jumlah sample pada dataset yaitu 42771 sampel.

### Standarisasi
Standardisasi merupakan teknik transformasi yang paling umum digunakan dalam tahap data preparation. Standarisasi membantu untuk membuat semua fitur numerik berada dalam skala data yang sama dan membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada proyek ini, standarisasi data dilakukan dengan menerapkan teknik StandarScaler dari library Scikitlearn. StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. 
    
## Modeling
Pada proyek ini, model yang dibuat merupakan tugas klasifikasi dengan lebih dari dua kelas atau banyak kelas yang mana menggunakan parameter carat, cut, color, claritym depth, table, price, x, y, dan z.
pada tahap ini, kita membuat model summary yang nantinya akan digunakan untuk membandingkan model/solusi yang akan digunakan. Dimana pada tahap ini akan dilakukan penghitungan nilai dari <i>K-Nearest Neighbor</i>, <i>Random Forest(RF)</i>, dan <i>Boosting Algorithm</i> kemudian nantinya akan dibandingkan performanya.


## Evaluation
Setelah membuat model <i>machine learning</i>, model perlu dievaluasi agar terbukti cocok untuk tujuan yang telah ditentukan. Fase ini bertujuan untuk memastikan bahwa model akan mampu membuat prediksi yang akurat dan tidak mengalami <i>overfitting</i> atau <i>underfitting</i>
Setelah mendapat seluruh performa dari 3 metode yang diterapkan maka hasil yang di dapatkan adalah
|          | train      | test        |
|----------|------------|-------------|
|    KNN   | 226.035885 | 2220.770793 |
|    RF    |  59.264645 | 1341.837668 |
| Boosting | 928.043394 | 2200.663138 |


## Deployment
Setelah melakukan evaluasi pada 3 metode pada train data dan test data maka di dapatkan 
|       | y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|-------|-------:|-------------:|------------:|------------------:|
| 35096 |    886 |       1692.7 |      2154.5 |             798.0 |

Pada proyek ini, terlihat pada program yang dijalankan bahwa prediksi dengan RF memberikan hasil yang paling mendekati. Dengan nilai prediksi KNN 1692.7 dan prediksi RF 2154.5. Perbandingannya dapat dilihat dari gambar diatas.
