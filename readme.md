# Atomic Design of Alkyne Semi-Hydrogenation Catalysts via Active Learning

This repository contains the data and code for the active learning searching in the paper "Atomic Design of Alkyne Semi-Hydrogenation Catalysts via Active Learning".

![image](image.png)

## Data
All Ni-based alloy within the scope of this study are listed in [this directory](Cat_Data/Raw_Crystal_Data_All).
The entire searching space of alloy surfaces are listed in [this directory](Cat_Data/Slab_All).

The DFT calculated surface in round 1 are listed in [this directory](Cat_Data/R1).
The DFT calculated surface in round 2 are listed in [this directory](Cat_Data/R2).

The model prediction and recommendation in round 1 are provided in [here](Results/Round_1/R1_predictions.xlsx).
The model prediction and recommendation in round 2 are provided in [here](Results/Round_2/R2_predictions.xlsx).

## Model
The model used in this work is provided in [here](model), and trained via [this script](model/run.py).
The trained model for [round 1](model/R1_model) and [round 2](model/R2_model) are also provided.
The model prediction can be obtained via [this script](model/run_predict.py).

