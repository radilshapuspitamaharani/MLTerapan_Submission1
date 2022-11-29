# MLTerapan_Submission1
# Laporan Proyek Machine Learning - Radilsha Puspita Maharani

## Domain Proyek

## Latar Belakang
Diamonds atau berlian adalah salah satu mineral karbon alami yang paling langka dan paling keras. intan telah digunakan sebagai batu permata selama berabad-abad karena jika disesuaikan dengan aspeknya, intan memiliki karakteristik 'api' karena indeks biasnya yang tinggi. Di masyarakat, berlian menjadi trend sebagai perhiasan sebagai hadiah pada momen-momen tertentu. Tidak hanya sebagai hadiah, berlian juga dipandang sebagai investasi alternatif selain emas karena nilainya dapat memberikan keuntungan dan peningkatan yang signifikan.
Dalam bidang perdagangan intan, pembeli atau investor mengalami kesulitan dalam memprediksi harga berlian karena adanya perbedaan bentuk, ukuran, dan kemurnian berlian. Banyak model dan aplikasi telah diimplementasikan untuk
memprediksi harga berlian ini di masa depan menggunakan pembelajaran mesin. Pembelajaran mesin dibagi menjadi dua kategori, yaitu diawasi dan tidak diawasi. Algoritma pembelajaran terbimbing menggunakan prinsip umum contoh
praktis untuk prediksi atau peramalan. 
Dalam hal ini, diperlukan sistem predictive analytics untuk menemukan model prediksi harga berlian yang paling efisien.


## Business Understanding

### Problem Statements
- Fitur apa yang paling berpengaruh terhadap harga diamonds
- Berapa harga pasar diamonds dengan karakteristik atau fitur tertentu?

### Goals
- Mengetahui fitur yang paling berkorelasi dengan harga diamonds
- Membuat model machine learning yang dapat memprediksi harga diamonds seakurat mungkin berdasarkan fitur-fitur yang ada

    ### Solution statements
    - Melakukan aalisis data, data cleaning untuk mencari missing value, eksplrasi dengan visualisasi data untuk menemukan outliers, dan preprocessing data yang tepat sebelum dilakukan train atau latih
    - menggunakan model logistik regresi untuk memprediksi nilai Boolean (benar atau salah) 

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
Hal yang akan dilakukan dalam fase ini adalah menggabungkan data, menyeleksi data yang akan digunakan, melakukan proses transformasi dat, dan membagi data menjadi data training dan test.
### Encoding Fitur Kategori
![image](https://user-images.githubusercontent.com/97927496/204614119-3d3bac92-f1a0-46b0-b1c9-1db4acd42b09.png)

### Reduksi Dimensi Dengan PCA
![image](https://user-images.githubusercontent.com/97927496/204614182-ce68d0b9-2419-4ea2-b023-47f330db3975.png)
![image](https://user-images.githubusercontent.com/97927496/204614227-1357ebe2-bc2c-428c-be91-e1f06bdee32f.png)

### Split Dataset atau Pembagian Dataset
![image](https://user-images.githubusercontent.com/97927496/204614261-2680e72b-bcbe-4e15-a465-0523edf978e6.png)

### Standarisasi
![image](https://user-images.githubusercontent.com/97927496/204614315-bef7a618-b6a2-46d8-ad58-a7e1177dac40.png)

## Modeling
Pada pemodelan ini menggunakan regresi machine learning karena nilai harga diamonds yang bersifat kontinyu, menggunakan regresi k-Nearest Neighbors. 
k-Nearest Neighbors adalah metode yang menggunakan algoritma terawasi. Regresi K-NN adalah metode nonparametrik intuitif dengan fitur menggunakan pendekatan k untuk menemukan nilai yang mendekati hasil dengan menghitung nilai kedekatan kasus baru dengan kasus lama, dimana k adalah jumlah nilai terdekat.
<img width="546" alt="image" src="https://user-images.githubusercontent.com/97927496/204616266-db08c93f-9669-40bb-a674-92e73de981de.png">


## Evaluation
Setelah membuat model machine learning, model perlu dievaluasi agar terbukti cocok untuk tujuan yang telah ditentukan. Fase ini bertujuan untuk memastikan bahwa model akan mampu membuat prediksi yang akurat dan tidak mengalami overfitting atau underfitting
<img width="662" alt="image" src="https://user-images.githubusercontent.com/97927496/204618261-224e245d-20ea-4089-b73b-7044a681eb1d.png">

## Deployment
<img width="483" alt="image" src="https://user-images.githubusercontent.com/97927496/204618366-f81cfce1-a57b-4694-b0d1-c12beaa8522f.png">

Pada proyek ini, terlihat pada program yang dijalankan bahwa prediksi dengan RF memberikan hasil yang paling mendekati. Dengan nilai prediksi KNN 1692.7 dan prediksi RF 2154.5. Perbandingannya dapat dilihat dari gambar diatas.
