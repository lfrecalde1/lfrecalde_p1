
## Running the Madgwick Filter

You can execute the Madgwick filter using the following command:

```bash
python3 Wrapper.py --data_root ../Data --split Train --imu_params ../IMUParams.mat --exp_num 5
```

By default:

* `--data_root` points to the root folder containing the Train/Test splits.
* `--split` selects the dataset split (`Train` or `Test`).
* `--imu_params` specifies the path to the IMU parameters file.
* `--exp_num` selects the experiment number.

The generated results will be automatically saved as **PDF files** for later inspection.