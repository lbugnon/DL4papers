# DL4papers - README

---

Version: 1.1.

Update:  2019-12-15.

URL: http://sinc.unl.edu.ar/web-demo/dl4papers/.

---

### Description

DL4papers is a new method based on deep learning that is capable of analyzing and interpreting papers in order to automatically extract relevant relations from the published literature, and between specific keywords. DL4papers receives as input a set of desired keywords, and it returns a ranked list of papers that contain meaningful associations between them. Our model was tested in a corpus of oncology papers and could be used as an accurate tool in rapidly identifying relationships between genes and their mutations, drug responses and treatments in the case of cancer patients.

The script provided will evaluate DL4papers on a previously embedded corpus of papers. Data is composed by 108 full PubMed manuscripts ([BRONCO dataset](http://infos.korea.ac.kr/bronco/)). For each text, biomedical keywords were identified: genes and its mutations, drugs, diseases and cell-lines. The texts were embedded with the FastText model. A 10-fold cross-validation is performed. DL4papers is evaluated using different pairs of mutation-gene and mutation-drug keywords. For each keyword pair, the model should say if there is or there is not a relation between the terms on every manuscript.

This is a distribution of the source code used in:

>  Bugnon L., C. Yones, J. Raad, M. Gerard, M. Rubiolo, G. Merino, M. Pividori, L. Di Persia, D.H. Milone, and G. Stegmayer, "DL4papers: a deep learning model for  interpretation of scientific articles", (to appear in) Oxford Bioinformatics (2020).

---

### Contents

1. [Requirements and installation](#1.-Requirements and installation)
2. [Source code organization](#2.-Source code organization)
3. [Running the model in a Colab](#3.-Running the model in a Colab)
4. [Running the model locally](#4.-Running the model locally)

---

#### 1. Requirements and installation

DL4papers was writen and tested in Python 3.5.

The deep learning model was implemented using:

* torch (1.1)
* numpy (1.18)
* pandas (0.25.3)
* matplotlib (3.1.2)
* sklearn (0.21.3)

Installation only requires to unzip the "<data>.zip" file.
After unzipping, required Python packages can be installed with:

```bash
python -m pip install --user -r ./requirements.txt
```

The *requirements.txt* file is already provided with the source code to easily install dependencies and run DL4papers code.

---

#### 2. Source code organization

The following structure is available after unzipping the source code:

```
.
+-- data/
|   +-- articles.pk
|   +-- cases_drug_mutation.pk
|   +-- cases_gene_mutation.pk
|   +-- embedding_drug_mutation.pk
|   +-- embedding_gene_mutation.pk
|
+-- src/
|   +-- dataloader.py
|   +-- logger.py
|   +-- model.py
|   +-- network.py
|   +-- sampler.py
|
+-- boxplot.py
+-- dl4papers.ipynb
+-- main.py
+-- requirements.txt
```

While **data** folder contains the preprocessed data used for DL4papers, **src** folder includes all classes and functions used to build the model.

The *requirements.txt* file include the list of python libraries required to run DL4papers. 

[main.py](#3.-RUNNING-MODEL-FROM-SOURCE-CODE) and [dl4papers.ipynb](#4.-RUNNING-MODEL-IN-COLAB) allow to run DL4papers in a desktop PC or in the cloud, respectively. Both options are described in Sections 3 and 4, respectively.

#### 2.1 *data* folder

The data was preprocessed as described in our article, and then stored using the Pickle module of Python's standard library.

* articles.pk: Contains the identifiers of papers analyzed in the *BRONCO dataset*.


* cases_*\<name\>*.pk:


>       | Field name   |  Description                                                                  |
>       | :----------- |  :--------------------------------------------------------------------------- |
>       | article      |  PubMed identifier for the article (i.e. PMC3386822).                         |
>       | target1      |  Type of entity (i.e. DRUG).                                                  |
>       | term1        |  First entity to search (i.e. METYLCELLULOSE).                                |
>       | pos1         |  Positions where "term1" was found in "article".                              |
>       | target2      |  Type of entity (i.e. MUTATION).                                              |
>       | term2        |  Second entity to search (i.e. V600E).                                        |
>       | pos2         |  Positions where "term2" was found in "article".                              |
>       | label        |  Label for classification ("0" are negative cases and "1" are positive ones). |

* embeddings_*\<name\>*.pk: These files contain the FastText embedding for each article in *BRONCO dataset*. Each store 108 Torch tensors, one per article, with 306 rows (300 rows corresponding to word embedding, 5 rows indicating the entity type, and 1 row to fill with the entities involved in the search) and 2^14 columns (maximum number of words in an article).

#### 2.2 *src* folder

This folder stores the files that comprise the core of DL4papers.
Below is available a short description of each one.

* *dataloader.py*: Class and methods to work with pre-procesed BRONCO data.

* *logger.py*: Class and methods to keep track of the different stages for the training and inference.

* *model.py*: Class and methods to perform training and testing of DL4papers model.

* *network.py*: Classes and methods that define the DL4papers neural model.

* *sampler.py*: Class and methods for managing data sampling (train-test partitions).

---

#### 3. Running the model in a Colab

Using a [Colaboratory](https://colab.research.google.com/) notebook makes it easier to run the scripts, since the code is executed on a server with the required computational power and software prerequisites.
The source folder needs to be uploaded to a Google Drive account. Then, the "dl4papers.ipynb" file can be opened from the Google Drive webpage, using the option "Open with Google Colaboratory".
A notebook will be opened in a new tab, with further instructions to run the experiment.

---

#### 4. Running the model locally

The models used in this code requires a GPU for training. Thus you should have the related firmware and CUDA toolkit installed. For more information, please check on https://www.pytorch.org/.

The main script can be run by executing the following command:

```bash
python main.py
```

During training, the model *loss* and *accuracy* are displayed on the console and saved to "results/" folder. Results for each testing fold are also saved in the same folder. After finishing, a boxplot of the results can be seen by executing:

```bash
python boxplot.py
```

*Note: It may take about 4 hours using an Nvidia TitanXP GPU.*

---

Licence: This project is licensed under the GNU General Public Licence.

Acknowledgments: Thanks to NVIDIA Corporation for the donation of several Titan Xp GPUs used for this research.
