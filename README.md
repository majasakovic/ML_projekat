# Klasifikacija bolesti lista paradajza primenom neuronskih mreža

## Članovi tima

- Anđela Nenadović 136/2020
- Maja Saković 1063/2025

## Opis projekta

Cilj projekta je klasifikacija bolesti lista paradajza na osnovu slika. Problem je formulisan kao višeklasna klasifikacija, pri čemu svaka slika pripada jednoj od 10 klasa koje predstavljaju različite bolesti lista paradajza ili zdrav list.

U projektu su implementirana i upoređena tri pristupa:

1. **Baseline CNN** - konvoluciona neuronska mreža trenirana od početka.
2. **Metric Learning + KNN** - model koji uči embedding reprezentacije slika pomoću Triplet Loss funkcije, nakon čega se klasifikacija vrši KNN klasifikatorom.
3. **Transfer Learning ResNet18** - model zasnovan na prethodno istreniranoj ResNet18 arhitekturi.

Modeli su poređeni korišćenjem metrika accuracy, precision, recall i F1-score, kao i pomoću matrica konfuzije.

## Skup podataka

Korišćen je skup podataka **Tomato Leaf Disease Detection**, dostupan na Kaggle platformi:

https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf

Skup sadrži slike listova paradajza organizovane po klasama. U projektu se koristi sledeća struktura:

```text
data/
  tomatoleaf/
    tomato/
      train/
        Tomato___Bacterial_spot/
        Tomato___Early_blight/
        ...
      val/
        Tomato___Bacterial_spot/
        Tomato___Early_blight/
        ...
```

Korišćeno je 10 klasa:

- Bacterial spot
- Early blight
- Late blight
- Leaf Mold
- Septoria leaf spot
- Spider mites Two-spotted spider mite
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato mosaic virus
- healthy

U prvoj svesci izvršena je osnovna analiza skupa podataka: provera strukture foldera, broj slika po klasama, balansiranost klasa, prikaz primera slika i t-SNE vizuelizacija. Na osnovu analize zaključeno je da su originalne slike veličine 256x256 i da su neke klase vizuelno slične, što može otežati klasifikaciju.

## Podela podataka

Originalni skup podataka već sadrži `train` i `val` foldere. Da bi se izbeglo korišćenje istog skupa i za izbor modela i za finalnu evaluaciju, korišćena je sledeća podela:

```text
originalni train skup -> train + validation
originalni val skup   -> test skup
```

Konkretno:

```text
Train size: 8000
Validation size: 2000
Test size: 1000
```

Validacioni skup izdvojen iz originalnog trening skupa koristi se za izbor najboljeg modela tokom treniranja, dok se originalni `val` folder koristi kao test skup za finalnu evaluaciju.

## Struktura projekta

```text
ML_projekat/
  data/
    README.md
    tomatoleaf/
      tomato/
        train/
        val/

  models/
    .gitkeep

  notebooks/
    01_dataset_analysis.ipynb
    02_baseline_cnn.ipynb
    03_metric_learning.ipynb
    04_transfer_learning_resnet18.ipynb
    05_model_comparison.ipynb

  README.md
  .gitignore
```

Folder `models/` postoji u repozitorijumu, ali istrenirani modeli nisu postavljeni direktno na GitHub zbog veličine fajlova. Umesto toga, potrebno ih je preuzeti posebno.

## Preuzimanje istreniranih modela

Istrenirani modeli i prateći fajlovi mogu se preuzeti sa sledećeg Google Drive linka:

https://drive.google.com/drive/folders/1lYGq5tSVeTENDFqQgTuBibUKVNiTlHow?usp=sharing

Nakon preuzimanja, fajlove treba smestiti u folder:

```text
models/
```

Očekivana struktura foldera je:

```text
models/
  baseline_cnn_best.pth
  best_embedding_cnn_triplet.pth
  metric_learning_knn.joblib
  metric_learning_train_embeddings.npy
  metric_learning_train_labels.npy
  transfer_learning_resnet18_best.pth
```

Modeli nisu deo GitHub repozitorijuma jer su ignorisani preko `.gitignore` fajla.

## Podešavanje okruženja

Projekat je rađen u Python okruženju uz korišćenje Jupyter Notebook sveski i PyTorch biblioteke.

Preporučeno je kreirati posebno virtuelno okruženje, na primer pomoću `conda`:

```bash
conda create -n torch_env python=3.11
conda activate torch_env
```

Potrebni paketi:

```bash
pip install torch torchvision 
pip install numpy pandas matplotlib seaborn scikit-learn joblib pillow
pip install jupyter
```

Ukoliko se koristi GPU, potrebno je instalirati odgovarajuću verziju PyTorch-a u skladu sa CUDA verzijom na računaru.

## Redosled pokretanja sveski

Sveske su imenovane redom kojim ih treba pregledati i pokretati.

### 01_dataset_analysis.ipynb

Sveska za osnovnu analizu skupa podataka.

Sadrži:

- učitavanje skupa podataka,
- proveru strukture foldera,
- prikaz primera slika,
- analizu broja slika po klasama,
- proveru balansiranosti klasa,
- t-SNE vizuelizaciju sličnosti slika.

### 02_baseline_cnn.ipynb

Sveska u kojoj je implementiran osnovni CNN model treniran od početka.
Model koristi:

- tri konvoluciona sloja,
- ReLU aktivacije,
- MaxPooling slojeve,
- Adaptive Average Pooling,
- Dropout regularizaciju,
- završni Linear sloj za klasifikaciju.

Model nema eksplicitni `Softmax` sloj, jer se za treniranje koristi `nn.CrossEntropyLoss`, koji očekuje logits izlaze modela i interno primenjuje odgovarajuću softmax/log-softmax transformaciju.

Najbolji model je biran na osnovu najmanje validacione greške (`val_loss`) i dobijen je u 19. epohi:

```text
Best validation loss = 0.2359
Best validation accuracy = 0.9215
```

### 03_metric_learning.ipynb

Sveska u kojoj je implementiran metric learning pristup.
Tok rada:

```text
slika -> Embedding CNN -> embedding vektor -> KNN -> klasa
```

Model se trenira pomoću `TripletMarginLoss` funkcije. Za svaki primer formira se triplet:

- anchor slika,
- positive slika iz iste klase,
- negative slika iz različite klase.

CNN ne predviđa direktno klasu, već uči embedding reprezentacije slika. Nakon treniranja embedding modela, izdvajaju se embedding vektori i trenira se KNN klasifikator. Najbolji KNN model čuva se pomoću `joblib` i koristi se u finalnom poređenju modela.

Tokom treninga kod metric learning modela ne prati se accuracy, jer CNN ne predviđa direktno klase, već uči embeddinge. Prate se `train_loss` i `val_loss`, a klasifikacione metrike računaju se kasnije pomoću KNN modela.
Testirane su sledeće vrednosti parametra `k` za KNN:

```text
k = 1, 3, 5, 7, 9, 11, 15
```

Najbolji rezultat na validacionom skupu postignut je za:

```text
Best k = 9
Best Validation Loss = 0.0295
Best Validation Accuracy = 0.8985
```

### 04_transfer_learning_resnet18.ipynb

Sveska u kojoj je implementiran transfer learning pristup pomoću ResNet18 modela.

Korišćen je prethodno istreniran ResNet18 model, a završni sloj je zamenjen tako da odgovara broju klasa u ovom problemu. Treniranje se vrši nad prilagođenim modelom, a najbolji model se bira na osnovu validacione greške.

Najbolji model je dobijen u 8. epohi:

```text
Best validation loss = 0.0383
Best validation accuracy = 0.9900
```

### 05_model_comparison.ipynb

Finalna sveska za poređenje modela.

Sadrži:

- učitavanje najboljih sačuvanih modela,
- učitavanje sačuvanog KNN klasifikatora za metric learning,
- evaluaciju sva tri pristupa na istom test skupu,
- tabelarno poređenje metrika,
- grafičko poređenje modela,
- classification report za svaki model,
- matrice konfuzije,
- zaključak poređenja.

## Rezultati

Finalno poređenje modela izvršeno je na test skupu od 1000 slika.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Baseline CNN | 0.922 | 0.9225 | 0.922 | 0.9218 |
| Metric Learning + KNN | 0.865 | 0.8670 | 0.865 | 0.8649 |
| Transfer Learning ResNet18 | 0.981 | 0.9812 | 0.981 | 0.9809 |

## Zaključak

Najbolji rezultat postigao je **Transfer Learning ResNet18** model, sa test tačnošću od 98.1%. Ovaj rezultat je očekivan, jer transfer learning koristi već naučene vizuelne karakteristike iz velikog skupa slika, što značajno pomaže kod klasifikacije slika listova.

**Baseline CNN** je postigao dobar rezultat od 92.2% tačnosti i predstavlja stabilan osnovni model za poređenje. Model je uspešno naučio relevantne vizuelne obrasce, ali slabije razlikuje neke vizuelno slične bolesti.

**Metric Learning + KNN** je dao slabiji rezultat od druga dva pristupa, sa tačnošću od 86.5%. Ipak, ovaj pristup je koristan jer pokazuje drugačiji način rešavanja problema: umesto direktne klasifikacije, model uči embedding prostor u kojem se slične slike grupišu.

Matrice konfuzije pokazuju da se najviše grešaka javlja između vizuelno sličnih bolesti lista, kao što su Early blight, Late blight, Leaf Mold, Target Spot i Spider mites.

## Literatura

1. Kaggle dataset: Tomato Leaf Disease Detection  
   https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf

2. FaceNet: A Unified Embedding for Face Recognition and Clustering
   https://arxiv.org/pdf/1503.03832

3. PyTorch Metric Learning
   https://kevinmusgrave.github.io/pytorch-metric-learning/

4. PyTorch Transfer Learning for Computer Vision Tutorial  
   https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

6. He, K., Zhang, X., Ren, S., & Sun, J. Deep Residual Learning for Image Recognition. CVPR, 2016.  
   https://arxiv.org/abs/1512.03385

8. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2011.  
   https://jmlr.org/papers/v12/pedregosa11a.html

9. Materijali sa vežbi kursa Mašinsko učenje asistentkinje Lucije Miličić, Univerzitet u Beogradu, Matematički fakultet
   https://github.com/matf-ml/materijali-sa-vezbi-2026.git

10. What is ResNet-18 - roboflow
    https://blog.roboflow.com/resnet-18/

11. Resnet18 Model With Sequential Layer For Computing Accuracy On Image Classification Dataset
    https://www.researchgate.net/publication/364345322_Resnet18_Model_With_Sequential_Layer_For_Computing_Accuracy_On_Image_Classification_Data
    
12. ResNet18 from Scratch Using PyTorch
    https://www.geeksforgeeks.org/deep-learning/resnet18-from-scratch-using-pytorch/

13. How to implement transfer learning in PyTorch
    https://www.geeksforgeeks.org/deep-learning/how-to-implement-transfer-learning-in-pytorch/

15. Transfer Learning with PyTorch: A Practical Guide
    https://medium.com/@serverwalainfra/transfer-learning-with-pytorch-a-practical-guide-8cc96aaee377

17. PyTorch Transfer Learning
    https://www.learnpytorch.io/06_pytorch_transfer_learning/

18. Metric Learning
    https://www.geeksforgeeks.org/artificial-intelligence/metric-learning/

19. An Introduction to Metric Learning
    https://dida.do/blog/metric-learning

20. Advanced Introduction to Triplet Loss
    https://qdrant.tech/articles/triplet-loss/

21. Triplet Loss
    https://www.researchgate.net/publication/357529033_Triplet_Loss

22. Triplet Loss tutorial
    https://perso.esiee.fr/~chierchg/deep-learning/tutorials/metric/metric-2.html

23. T-distributed Stochastic Neighbor Embedding (t-SNE) Algorithm - ML
    https://www.geeksforgeeks.org/machine-learning/ml-t-distributed-stochastic-neighbor-embedding-t-sne-algorithm/

24. Visualizing Data using t-SNE
    https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

25. Introduction to t-SNE
    https://www.datacamp.com/tutorial/introduction-t-sne

26. Introduction to Convolution Neural Network
    https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/

27. Convolutional Neural Networks: A Comprehensive Guide
    https://medium.com/thedeephub/convolutional-neural-networks-a-comprehensive-guide-5cc0b5eae175
