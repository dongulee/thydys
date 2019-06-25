# Thydys: Thyroid dysfunction predicting Deep w/ wearable device bio-sequences

## Requirements
* An anaconda environment is recommended (refer to environments.yml)

    or
### Versions
    * Python 3.6
    * Tensorflow 1.13.1
    * cudnn 7.3.1
    * CUDA 10.0

## Data preparation

    ```bash
    python prep_data.py -t raw -d data -s 15T --duration 10 -o data/10D15m

    ```


## Train

    ```bash
    python train.py --input-dir=data/10D15m --timesteps=960 --num-input=2 -o ./model_10D_1

    ```

## Infer

    ```bash
    python train.py --input-dir=data/10D15m --timesteps=960 --num-input=2 -o ./model_10D_1
    ```
