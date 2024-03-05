# 3D-LEX Semi-Automatic Annotation of Phonemes in Sign Language

This repository contains the code to perform semi-autmatic annotations of phonetic properties based on the 3D-LEX dataset, and to reproduce the results of the paper: _3D-LEX v1.0: 3D Lexicons for American Sign Language and Sign Language of the Netherlands_.

Experiments currently available for the following phonetic classes:

    Handshapes

## Setup (Linux)

1. Download 3D-LEX dataset
2. Download the WLASL/SemLEX metadata files 
3. setup environment: 

    bash setup/setup.sh

    conda activate saa



## Datasets
#### 3D-LEX
_Data will be made available upon publication of the associated paper._


#### Evaluation: WLASL & SemLEX
Evaluations of new phonetic labels are currently available with WLASL 2000 ISLR benchmark [(Li et al., 2020)](https://arxiv.org/abs/1910.11006), which has been combinet with the phonological annotations in ASL-LEX 2.0 [(Sehyr et al., 2021)](https://academic.oup.com/jdsde/article/26/2/263/6142509) in Kezar et al., 2023. Please obtain the metadata for the merged ASL-LEX and WLASL benchmark from the original repository [(Kezar et al., 2023)](https://github.com/leekezar/Modeling-ASL-Phonology/tree/main/training_data) and WLASL pose data from the [openhands repositroy](https://openhands.ai4bharat.org/en/latest/instructions/datasets.html) for evaluations or 


mkdir data/wlasl/
wget https://zenodo.org/record/6674324/files/WLASL.zip?download=1
mv WLASL.zip?download=1 data/wlasl/
cd data/wlasl/
unzip WLASL.zip?download=1
rm WLASL.zip?download=1

_Evaluations on SemLEX will be available soon._

## Demos
### Phonetic Labeling with StretchSense Gloves
To produce new handshape labels run either

    python main_ED.py --mode annotate
    
    python main_KMeans.py --mode annotate


### Evaluation
To perform evaluations using openhands on the wlasl benchmark, please run:

    cd evaluation/isolated_sign_recognition/
    
    python train.py

    # Edit model path in evaluation/isolated_sign_recognition/configs/gcn_test.yaml
    
    python test.py



## Citation
_Coming soon_