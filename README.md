# ISOLATOR
This repository is the official implementation of "Unveiling the Impact of Multi-modal Content in Multi-modal Recommender Systems".
## Environment
- python == 3.10.13
- pytorch == 2.0.0
- numpy == 1.23.5
- pandas == 1.5.3
- pyyaml == 6.0
- scipy == 1.11.4
## Quik Start
1. Download the data under './data/' and obtain the embeddings required by the algorithm.
2. Change model inference startegy(D-ISOLATOR/U-ISOLATOR) under './src/models/vbpr.py' on 'full_sort_predict(..)' function
3. Run the following command to train and evaluate the model.
 ```python
   # run the code 
   python -u main.py --model=VBPR --dataset=baby --gpu_id=0
 ```
