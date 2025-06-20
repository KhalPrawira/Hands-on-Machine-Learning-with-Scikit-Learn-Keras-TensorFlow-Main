<img src="https://github.com/KhalPrawira/Hands-on-Machine-Learning-with-Scikit-Learn-Keras-TensorFlow-Main/blob/66abfed0c4052497210677bdaf3125f53b6b9c5c/Cover.jpg" width="300" alt="Book Cover">

# Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)

**Penulis:** Aurélien Géron

Buku "Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow" edisi ke-2 ini adalah panduan komprehensif yang dirancang untuk membantu pembaca memahami dan mengimplementasikan konsep-konsep *Machine Learning* dan *Deep Learning* secara praktis. Dengan pendekatan "belajar sambil melakukan", buku ini menggabungkan teori minimal dengan banyak contoh kode Python yang siap produksi, menggunakan *framework* populer seperti Scikit-Learn, Keras, dan TensorFlow 2.

Buku ini cocok untuk programmer yang ingin beralih ke Machine Learning, mahasiswa, atau siapa saja yang ingin memperdalam pemahaman mereka tentang membangun sistem cerdas.

Buku ini dibagi menjadi dua bagian besar:

**Bagian 1: Fondasi Machine Learning (Bab 1-8)**
Bagian ini memperkenalkan konsep-konsep dasar *Machine Learning*, mulai dari jenis-jenis masalah ML, hingga algoritma tradisional seperti Regresi Linier, Support Vector Machines (SVMs), dan Decision Trees. Pembaca akan belajar tentang persiapan data, pemilihan model, evaluasi kinerja, dan teknik regularisasi.

**Bagian 2: Deep Learning (Bab 9-19)**
Bagian ini menyelami dunia *Deep Learning*, dimulai dari pengenalan Jaringan Saraf Tiruan (JST) dasar hingga arsitektur yang lebih kompleks seperti Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Autoencoders, dan Generative Adversarial Networks (GANs).

---

## Bagian I: Fondasi Machine Learning

### Bab 1: The Machine Learning Landscape
Bab ini memberikan pengenalan tingkat tinggi tentang dunia Machine Learning (ML). Ia mendefinisikan apa itu ML, mengapa ML berguna, dan memetakan berbagai jenis sistem ML berdasarkan:
* **Tingkat supervisi**: Supervised, Unsupervised, Semisupervised, Reinforcement Learning.
* **Cara belajar**: Batch vs. Online.
* **Cara generalisasi**: Instance-based vs. Model-based.
Bab ini juga membahas tantangan utama dalam ML seperti kualitas dan kuantitas data, serta masalah *overfitting* dan *underfitting*.

### Bab 2: End-to-End Machine Learning Project
Bab ini memandu pembaca melalui contoh proyek ML lengkap menggunakan dataset California Housing Prices. Langkah-langkah yang dibahas meliputi mendapatkan data, membuat *test set* dengan pentingnya *stratified sampling* untuk mencegah *data snooping bias*, eksplorasi dan visualisasi untuk mencari pola, serta persiapan data (*preprocessing*). Secara khusus, bab ini menekankan pembersihan data, pengisian nilai yang hilang, penanganan fitur kategorikal, dan *feature scaling* menggunakan `Pipeline` dan `ColumnTransformer` dari Scikit-Learn. Setelah itu, dibahas cara melatih beberapa model, menyempurnakan *hyperparameter* dengan `GridSearchCV`, dan mengevaluasi model akhir.

### Bab 3: Classification
Bab ini berfokus pada tugas klasifikasi. Metrik evaluasi yang lebih baik dari sekadar akurasi diperkenalkan, seperti *confusion matrix*, *precision*, *recall*, F1 score, dan kurva ROC. Konsep *multiclass*, *multilabel*, dan *multioutput classification* juga dijelaskan.

### Bab 4: Training Models
Bab ini menyelami beberapa model linier yang paling umum. Dibahas Regresi Linier (termasuk *Normal Equation* dan *Gradient Descent*), Regresi Polinomial, teknik regularisasi (Ridge, Lasso, Elastic Net) untuk mengurangi *overfitting*, dan model klasifikasi seperti Regresi Logistik dan Softmax.

### Bab 5: Support Vector Machines (SVMs)
Bab ini menjelaskan cara kerja SVM, salah satu algoritma ML yang paling kuat. Konsep inti seperti *large margin classification*, *support vectors*, dan *kernel trick* (untuk menangani data non-linier) diuraikan secara detail.

### Bab 6: Decision Trees
Bab ini membahas Decision Trees, algoritma serbaguna yang dapat digunakan untuk klasifikasi dan regresi. Dijelaskan bagaimana pohon keputusan bekerja, bagaimana melatihnya menggunakan algoritma CART, dan pentingnya regularisasi untuk mencegah *overfitting*.

### Bab 7: Ensemble Learning and Random Forests
Bab ini menunjukkan bagaimana menggabungkan beberapa model (*ensemble learning*) dapat meningkatkan kinerja. Berbagai strategi dibahas, termasuk *Voting*, *Bagging*, *Pasting*, *Boosting* (AdaBoost, Gradient Boosting), dan *Stacking*. Random Forests, yang merupakan *ensemble* dari Decision Trees, dijelaskan secara mendalam.

### Bab 8: Dimensionality Reduction
Bab ini membahas "kutukan dimensi" (*curse of dimensionality*) dan menyajikan teknik untuk mengurangi jumlah fitur dalam data. Algoritma utama yang dibahas adalah PCA (Principal Component Analysis), Kernel PCA, dan LLE (Locally Linear Embedding).

---

## Bagian II: Deep Learning

### Bab 9: Unsupervised Learning
Bab ini berfokus pada algoritma *unsupervised learning*, terutama untuk *clustering*. Algoritma yang dibahas meliputi K-Means, DBSCAN (untuk menemukan *cluster* dengan bentuk arbitrer), dan Gaussian Mixture Models (GMM) yang dapat digunakan untuk estimasi kepadatan dan deteksi anomali.

### Bab 10: Introduction to Artificial Neural Networks with Keras
Ini adalah bab pengantar ke *Deep Learning* menggunakan Keras. Dimulai dari inspirasi biologis (neuron), bab ini menjelaskan arsitektur Perceptron, *Multi-Layer Perceptron* (MLP), dan cara melatihnya dengan *backpropagation*. Pembaca akan belajar membangun JST untuk klasifikasi dan regresi menggunakan Sequential dan Functional API di Keras.

### Bab 11: Training Deep Neural Networks
Bab ini membahas tantangan dalam melatih jaringan saraf yang dalam (DNNs). Solusi untuk masalah seperti *vanishing/exploding gradients* disajikan, termasuk inisialisasi bobot (Glorot & He), fungsi aktivasi *nonsaturating* (ReLU, ELU, SELU), *Batch Normalization*, dan *Gradient Clipping*. Teknik untuk mempercepat pelatihan seperti *optimizer* canggih (Adam, Nadam) dan *learning rate scheduling* juga dibahas.

### Bab 12: Custom Models and Training with TensorFlow
Bab ini menyelami API tingkat rendah TensorFlow untuk memberikan fleksibilitas penuh. Pembaca akan belajar cara membuat *loss function*, *layer*, model, dan bahkan *training loop* kustom dari awal menggunakan `tf.GradientTape`.

### Bab 13: Loading and Preprocessing Data with TensorFlow
Bab ini berfokus pada cara memuat dan memproses data dalam skala besar secara efisien. Diperkenalkan Data API (`tf.data`) untuk membangun *pipeline* input yang berperforma tinggi, format file TFRecord untuk penyimpanan data yang efisien, dan *preprocessing layers* di Keras.

### Bab 14: Deep Computer Vision with CNNs
Bab ini membahas Convolutional Neural Networks (CNNs), arsitektur standar untuk tugas *computer vision*. Dijelaskan komponen inti seperti lapisan konvolusional dan *pooling*, serta arsitektur CNN populer (misalnya, ResNet, Xception). Topik lanjutan seperti deteksi objek dan segmentasi juga diperkenalkan.

### Bab 15: Processing Sequences Using RNNs and Attention
Bab ini berfokus pada pemrosesan data sekuensial (seperti deret waktu dan teks) menggunakan Recurrent Neural Networks (RNNs). Dibahas arsitektur dasar RNN, masalah memori jangka pendek, dan solusinya melalui sel LSTM dan GRU. Arsitektur WaveNet yang menggunakan konvolusi juga diperkenalkan.

### Bab 16: Natural Language Processing with RNNs and Attention
Bab ini adalah kelanjutan dari Bab 15, menerapkan RNN dan *Attention* ke Natural Language Processing (NLP). Topik meliputi *word embeddings*, arsitektur *encoder-decoder* untuk penerjemahan mesin, mekanisme *Attention*, dan arsitektur Transformer.

### Bab 17: Autoencoders and GANs
Bab ini membahas dua keluarga model *generatif*: Autoencoders dan Generative Adversarial Networks (GANs). Autoencoders digunakan untuk reduksi dimensi dan pembelajaran representasi (termasuk *denoising* dan *variational* autoencoders), sementara GANs, yang terdiri dari **Generator** dan **Discriminator** yang saling bersaing, mampu menghasilkan data (terutama gambar) yang sangat realistis.

### Bab 18: Reinforcement Learning
Pengenalan ke Reinforcement Learning (RL), di mana *agent* belajar melalui coba-coba. Dibahas konsep-konsep inti seperti *agent*, *environment*, *action*, *reward*, dan *policy*. Algoritma kunci seperti Q-Learning (DQN) dan Policy Gradients (PG) dijelaskan, serta penggunaan OpenAI Gym dan TF-Agents.

### Bab 19: Training and Deploying TensorFlow Models in Scale
Bab ini membahas aspek praktis dari membawa model ke produksi. Topik meliputi melatih model dalam skala besar menggunakan **Distribution Strategies API (`tf.distribute`)**, menyebarkan model untuk inferensi dengan **TensorFlow Serving**, mengoptimalkan model untuk perangkat seluler dan *edge* menggunakan **TensorFlow Lite (TFLite)**, dan menjalankan model di browser dengan **TensorFlow.js**.