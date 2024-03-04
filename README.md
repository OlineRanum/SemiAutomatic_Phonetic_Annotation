# 3D-LEX Semi-Automatic Annotation of Phonemes in Sign Language

This repository contains the code to perform semi-autmatic annotations of phonetic properties based on the 3D-LEX dataset, and to reproduce the results of the paper: _3D-LEX v1.0: 3D Lexicons for American Sign Language and Sign Language of the Netherlands_.

Currently available phoneme classes: 

    - Handshapes

## Setup (Linux)

    1. Download 3D-LEX dataset
    2. Download the WLASL/SemLEX metadata files 
    3. setup environment: 

    bash setup/setup.sh



## Datasets
#### 3D-LEX

#### Evaluation: WLASL & SemLEX
Evaluations of new phonetic labels are currently available with WLASL 2000 ISLR benchmark [(Li et al., 2020)](https://arxiv.org/abs/1910.11006), which has been combinet with the phonological annotations in ASL-LEX 2.0 [(Sehyr et al., 2021)](https://academic.oup.com/jdsde/article/26/2/263/6142509) in Kezar et al., 2023. Please obtain the metadata for the merged ASL-LEX and WLASL benchmark from the original repository [(Kezar et al., 2023)](https://github.com/leekezar/Modeling-ASL-Phonology/tree/main/training_data).

Evaluations on SemLEX will be available soon. 

## Demos
### Phonetic Labeling with StretchSense Gloves

This repository contains code to perform semi-automatic annotation of handshapes based upon the StretchSense Gloves for sign language lexicons



1. clean dir
2. make new wlasl
3. least squares
4. k means