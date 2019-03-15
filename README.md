# RGNN

This repository contains code for the paper _Retrieval Guided Neural Networks for Essay Generation_.

## Requirements

- Python2.7
- Tensoflow1.7
- Texar

## Data Set

Composition Data Set: [Download](https://pan.baidu.com/s/1_JPh5-g2rry2QmbjQ3pZ6w)

## Useage

- You can use _train.sh_ and _gen.sh_  to train RGNN and generate essays respectively. The defalut config is for RGNNv2.  

- For RGNNv1, use the following command:

    ```sh
    python main.py --mode=train --config=config_retrieval --exp_name=log_dir --method_name=EncRetrievalAttDecoderWordEnc
    ```
  
- For RGNNv3, use the following command:

    ```sh
    python main.py --mode=train --config=config_retrieval_test --exp_name=log_dir --method_name=EncRetrievalAttDecoderWordEncCopy
    ```

- This code also contains two baselines: PNN and TAT-LSTM. The two commands are as follows:

    ```sh
    python main.py --mode=train --config=config --exp_name=log_dir --method_name=PNN
    python main.py --mode=train --config=config --exp_name=log_dir --method_name=EncAttentionDecoder
    ```

### Training

```sh
./train.sh
```

### Generation


```sh
./gen.sh
```
