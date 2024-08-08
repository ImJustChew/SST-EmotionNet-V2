# Preprocessing
## Expected Directories
### Training Data Directories

1. **train\_specInput\_root\_path**: This should contain files named `subject_{i}/section_{j}_data.npy` where `i` is the subject number (0-14) and `j`is the session number (0-2).
    - Example: `subject_0/section_0_data.npy`
2. **train\_tempInput\_root\_path**: Similar structure to `train_specInput_root_path`, containing files named `subject_{i}/section_{j}_data.npy`.
    - Example: `subject_0/section_0_data.npy`
3. **train\_label\_root\_path**: This should contain files named `subject_{i}/section_{j}_label.npy`.
    - Example: `subject_0/section_0_label.npy`

### Testing Data Directories

1. **test\_specInput\_root\_path**: This should contain files named `subject_{i}/section_{j}_data.npy` where `i` is the subject number (0-14) and `j`is the session number (0-2).
    - Example: `subject_0/section_0_data.npy`
2. **test\_tempInput\_root\_path**: Similar structure to `test_specInput_root_path`, containing files named `subject_{i}/section_{j}_data.npy`.
    - Example: `subject_0/section_0_data.npy`
3. **test\_label\_root\_path**: This should contain files named `subject_{i}/section_{j}_label.npy`.
    - Example: `subject_0/section_0_label.npy`

### Result Directories

1. **result\_path**: Directory where the results and logs are stored.
    - Example files:
        - `all_result.txt`
        - `Sub_{i}_Session_{j}.txt`
2. **model\_save\_path**: Directory where the trained model files are saved.
    - Example files:
        - `Sub_{i}_Session_{j}.h5`

1. Extract 5 bands for spectral data and and 25 bands for temporal data.
*25 Bands later changed to 40 bands for temporal data.

To pick 25 bands, we first used frequency downsampling,
Then later tried with averaging the bands.
Finally with PCA, yields highest improvement in accuracy, from 0.1 to 0.5

val split was 0.4, later changed to 0.1. Using 0.2 now

processed data is in wrong shape, needs to reshape to (trials, size, size, depth, 1) 

# Run

Was based on per subject per session model.
Later changed to per subject model. `run.py` is the per subject model.
Finally `run_independent.py` is a subject independent model.

# Results
run_independent.py (without normalization): 
PCA=40, T=5min, Chn=all
| Conditions | Accuracy |
|------------|----------|
| Dropout 0.5 | 0.5 |
| Dropout 0.3 | 0.8 |
| Dropout 0.1 | 0.5666666626930237 |
| Dropout 0.1 | 0.8777777552604675 |
| Dropout 0.1, no internal dropout | 0.8444444537162781 |
| Dropout 0.1, no internal dropout | 0.5333333611488342 |
| Dropout 0.3 | 0.7777777910232544 |