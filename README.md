# LeNet-5 for Tifinagh Character Classification



## 1. Présentation du projet

Ce projet porte sur la reprogrammation et l'adaptation de l'architecture convolutionnelle **LeNet-5** pour la classification de caractères manuscrits Tifinagh à partir de la base de données AMHCD (*Amazigh Handwritten Character Database*).

L'objectif est d'étudier les performances d'un réseau de neurones convolutif sur une tâche de classification multiclasse, d'analyser ses résultats quantitatifs et qualitatifs, et de comparer plusieurs algorithmes d'optimisation.

Le projet a été développé avec Python et PyTorch dans Google Colab, avec accélération GPU NVIDIA Tesla T4.

## 2. Objectifs

* Adapter l'architecture LeNet-5 à des images RGB et à 33 classes.
* Implémenter des couches paramétrables pour les convolutions, le pooling, les activations et les couches entièrement connectées.
* Séparer les données en ensembles d'entraînement, de validation et de test.
* Étudier l'évolution des fonctions de perte et des précisions.
* Évaluer le modèle à l'aide de métriques de classification.
* Visualiser la matrice de confusion, les cartes de caractéristiques et des exemples de prédictions.
* Comparer SGD, SGD avec Momentum, RMSprop et Adam.

## 3. Base de données

La base utilisée est **AMHCD — Amazigh Handwritten Character Database**.

* **Nombre d'images :** 25 740
* **Nombre de classes :** 33 caractères Tifinagh
* **Images originales :** 64 × 64 pixels, RGB
* **Nombre d'images par classe :** 780

Source : [AMHCD — Kaggle](https://www.kaggle.com/datasets/benaddym/amazigh-handwritten-character-database-amhcd)

### Prétraitement

Les images sont redimensionnées en 32 × 32 pixels, converties dans l'intervalle [0, 1], puis normalisées par canal à partir des moyennes et écarts-types calculés sur l'ensemble d'entraînement.

La répartition stratifiée, effectuée avec la graine aléatoire 42, est la suivante :

| Ensemble     | Proportion | Nombre d'images |
| ------------ | ---------: | --------------: |
| Entraînement |       70 % |          18 018 |
| Validation   |       15 % |           3 861 |
| Test         |       15 % |           3 861 |
| **Total**    |  **100 %** |      **25 740** |

Les données sont chargées par lots de 64 images.

## 4. Architecture du modèle

Le modèle reprend les principales étapes de LeNet-5, avec une entrée RGB et une couche de sortie adaptée aux 33 classes.

| Couche | Opération                                        | Dimensions de sortie |
| ------ | ------------------------------------------------ | -------------------- |
| Entrée | Image RGB                                        | 32 × 32 × 3          |
| C1     | Convolution, 6 filtres 5 × 5 + tanh              | 28 × 28 × 6          |
| S2     | Average Pooling 2 × 2                            | 14 × 14 × 6          |
| C3     | Convolution, 16 filtres 5 × 5 + tanh             | 10 × 10 × 16         |
| S4     | Average Pooling 2 × 2                            | 5 × 5 × 16           |
| C5     | Convolution, 120 filtres 5 × 5 + tanh            | 1 × 1 × 120          |
| F6     | Couche entièrement connectée, 84 neurones + tanh | 84                   |
| Sortie | Couche entièrement connectée, 33 neurones        | 33                   |

Le modèle comporte **63 961 paramètres entraînables**. Le dropout est paramétrable, avec une probabilité fixée à 0 dans l'expérience principale.

La fonction de perte utilisée est `CrossEntropyLoss`. Le modèle retenu est celui qui obtient la meilleure précision sur l'ensemble de validation.

## 5. Environnement technique

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* kagglehub
* Google Colab avec GPU NVIDIA Tesla T4

Les dépendances nécessaires à l'exécution doivent être précisées dans le fichier `requirements.txt`.

## 6. Résultats expérimentaux

### 6.1. Modèle principal

Le modèle principal utilise Adam avec un taux d'apprentissage de \(10^{-3}\), pendant 30 époques.

| Métrique                                      | Résultat |
| --------------------------------------------- | -------: |
| Précision d'entraînement à la dernière époque |  99,99 % |
| Précision de validation à la dernière époque  |  98,81 % |
| Accuracy sur le test                          |  98,29 % |
| Précision macro sur le test                   |   98,3 % |
| Rappel macro sur le test                      |   98,3 % |
| F1-score macro sur le test                    |   98,3 % |
| Temps d'entraînement rapporté                 |   32,5 s |

Le temps d'exécution correspond à l'environnement Google Colab utilisé et peut varier selon les ressources disponibles.

### 6.2. Comparaison des optimiseurs

Les optimiseurs ont été comparés sur 15 époques, avec une seule exécution par configuration.

| Optimiseur           | Taux d'apprentissage | Accuracy validation | Accuracy test |
| -------------------- | -------------------: | ------------------: | ------------: |
| SGD                  |                 0,01 |             88,45 % |       89,69 % |
| SGD + Momentum (0,9) |                 0,01 |             97,93 % |       98,11 % |
| RMSprop              |                0,001 |             98,32 % |       97,80 % |
| Adam                 |                0,001 |             98,63 % |       97,93 % |

Ces résultats montrent que Momentum, RMSprop et Adam obtiennent des performances proches dans les configurations testées. Une seule exécution par configuration ne permet cependant pas de conclure à la supériorité générale d'un optimiseur.

### 6.3. Analyse qualitative

La matrice de confusion est globalement proche d'une diagonale, ce qui indique une bonne classification de la majorité des caractères.

Les cartes de caractéristiques montrent des motifs visuels différents selon la profondeur du réseau : les premières couches détectent notamment des contours et des orientations, tandis que les couches suivantes produisent des représentations plus abstraites.

Les exemples de prédictions visualisés illustrent le comportement du modèle sur des images individuelles.

## 7. Figures et résultats visuels

Les figures générées par le notebook peuvent être regroupées dans le dossier `results/`.

| Fichier                       | Description                                |
| ----------------------------- | ------------------------------------------ |
| `courbes.png`                 | Évolution de la perte et de l'accuracy     |
| `confusion.png`               | Matrice de confusion sur le test           |
| `fmaps_c1.png`                | Cartes de caractéristiques de la couche C1 |
| `fmaps_c3.png`                | Cartes de caractéristiques de la couche C3 |
| `predictions.png`             | Exemples de prédictions                    |
| `comparaison_optimiseurs.png` | Comparaison graphique des optimiseurs      |
| `comparaison_optimiseurs.csv` | Résultats numériques de la comparaison     |

## 8. Installation et utilisation

### Option 1 : Google Colab 
1. Ouvrir le notebook `LeNet5_Tifinagh.ipynb` dans Google Colab.
2. Activer le GPU 
3. Exécuter les cellules dans l'ordre. La base AMHCD est téléchargée automatiquement avec `kagglehub`.

### Option 2 : installation locale
```bash
git clone https://github.com/ikramelghidouni59/Tp_Impl-mentation_LeNet5.git
cd Tp_Impl-mentation_LeNet5
pip install -r requirements.txt
jupyter notebook
```

Le téléchargement de la base peut demander un compte Kaggle configuré (`kaggle.json`).




