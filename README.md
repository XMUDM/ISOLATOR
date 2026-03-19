# ISOLATOR

This repository is the official implementation of the paper:  
**"Unveiling the Impact of Multi-modal Content in Multi-modal Recommender Systems"**  
Accepted at **ACM Multimedia 2025**.

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{10.1145/3746027.3755300,
  author    = {Xv, Guipeng and Li, Xinyu and Liu, Yi and Lin, Chen and Wang, Xiaoli},
  title     = {Unveiling the Impact of Multi-modal Content in Multi-modal Recommender Systems},
  year      = {2025},
  isbn      = {9798400720352},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3746027.3755300},
  doi       = {10.1145/3746027.3755300},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages     = {6093–6102},
  numpages  = {10},
  keywords  = {causal inference, multi-modal recommender system, user-side content bias},
  location  = {Dublin, Ireland},
  series    = {MM '25}
}
```

## Environment

The code has been tested with the following dependencies:

- Python == 3.10.13
- PyTorch == 2.0.0
- NumPy == 1.23.5
- Pandas == 1.5.3
- PyYAML == 6.0
- SciPy == 1.11.4

## Quick Start

1. **Prepare the data**  
   Download the datasets and place them under the `./data/` directory. Then generate the required multi-modal embeddings as described in the paper.

2. **Choose the variant**  
   - `src_D/` corresponds to **D-ISOLATOR** (Debiasing Item-side Content)  
   - `src_U/` corresponds to **U-ISOLATOR** (Debiasing User-side Content)  
   Navigate to the appropriate directory and modify the hyperparameter search space in the configuration file (`config/*.yaml`) if needed.

3. **Run training and evaluation**  
   Execute the following command:
   ```bash
   python -u main.py --model=VBPR --dataset=baby --gpu_id=0
   ```
   Replace `VBPR` with the desired backbone model and `baby` with your dataset name. The `-u` flag ensures unbuffered output.

## Notes

- For more details on the method, datasets, and hyperparameter settings, please refer to our paper.
- If you encounter any issues or have questions, feel free to open an issue on GitHub.
