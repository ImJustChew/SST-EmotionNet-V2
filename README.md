# SST-EmotionNet-V2 
SST-EmotionNet: Spatial-Spectral-Temporal based Attention 3D Dense Network for EEG Emotion Recognition

Rewritten to support newer version of Tensorflow 2.5.0 and Keras.

Based off [https://github.com/ziyujia/SST-EmotionNet](https://github.com/ziyujia/SST-EmotionNet)

## Requirements
- Python 3.6-3.9
- CUDA 11.2
- cuDNN 8.1


> Note: Notes were taken in notes.md, they might not be accurate or properly documented.

## Preprocessing
Place `Preprocessed_EEG` folder from SEED dataset in root folder, and run `preprocess_seed.ipynb` to generate the processed data. 

Resulting data should be processed to `SEED_input_data` folder.


## Config Usage

- Configuration

  We provide a sample configuration file `SEED.ini` for SEED dataset. 

  - `input_width` denotes the width of 2D map.
  - `specInput_length` and `temInput_length` denote how many 2D maps are stacked in the 3D spatial-spectral representation and 3D spatial-temporal representation, respectively.
  -  `depth_spec` and `depth_tem` denote the number of layers in spatial-spectral stream and spatial-temporal stream.
  -  `nb_dense_block` denotes the number of A3DBs to add to end. 
  - `gr_spec ` and `gr_tem ` denote the growth rate of spatial-spectral stream and spatial-temporal stream. 

## Running the model
For Subject Independent model, run `run_independent.py`.
For Subject Dependent model, run `run_dependent.py`.

Run with -c parameter, which refers to the path of the configuration file for training. For instance, the model is trained by:
  ```
    python run_independent.py -c ./config/SEED.ini
  ```

# References

```latex
@inproceedings{jia2020sst,
  title={SST-EmotionNet: Spatial-spectral-temporal based attention 3D dense network for EEG emotion recognition},
  author={Jia, Ziyu and Lin, Youfang and Cai, Xiyang and Chen, Haobin and Gou, Haijun and Wang, Jing},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2909--2917},
  year={2020}
}
```


