## Code repo for the NeurIPS 2022 OGB challenge of the dataset PCQM4Mv2 ##

### 1. Dataset ###
Make sure that you have downloaded the dataset PCQM4Mv2 in advance before you begin to process the data (Quick).
```bash
cd data
python dataset.py
```
Hopefully this will generate the files within the processed folder under pcqm4m-v2 folder :) The processing time depends on your CPU. For example, the Intel 9 10980HK will spend approxmiately 30:00 processing the data.
### 2. Training process ###
main train.py file is located in the model folder.
```bash
cd model
python train.py
```
### 3. Visualization ###
To be continue
