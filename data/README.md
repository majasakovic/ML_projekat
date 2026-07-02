## Preuzimanje i organizacija skupa podataka

Za projekat se koristi Kaggle skup podataka **Tomato leaf disease detection**:

https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf

Skup podataka se ne čuva u GitHub repozitorijumu zbog veličine. Potrebno ga je preuzeti ručno sa Kaggle stranice i raspakovati lokalno u folder `data/tomatoleaf/`.

Nakon raspakivanja, očekivana struktura foldera je:

ML_projekat/
    data/
        tomatoleaf/
            tomato/
                train/
                val/
                cnn_train.py

Folderi `train` i `val` sadrže podfoldere koji predstavljaju klase bolesti paradajza. Ime svakog podfoldera predstavlja labelu klase.