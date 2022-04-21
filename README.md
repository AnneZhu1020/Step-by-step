# Step-by-step
Step by step: a hierarchical framework for multi-hop knowledge graph reasoning with reinforcement learning
https://doi.org/10.1016/j.knosys.2022.108843

### MindSpore Version
https://github.com/CC1st/step-by-step-mindspore by Luo Xuewei

## Dependencies
tqdm==4.9.0\
matplotlib==2.1.2

## Process data
First, unpack the data files
```
tar xvzf data-release.tgz
```
and run the following command to preprocess the datasets.

```
./experiment.sh configs/<dataset>.sh --process_data <gpu-ID>
```
<dataset> is the name of any dataset folder in the ./data directory. In our experiments, the five datasets used are: umls, kinship, fb15k-237, wn18rr and nell-995. <gpu-ID> is a non-negative integer number representing the GPU index.
  
## Train models
Then the following commands can be used to train the proposed models and baselines in the paper. By default, dev set evaluation results will be printed when training terminates.

* Train embedding-based models
```
./experiment-emb.sh configs/<dataset>-<emb_model>.sh --train <gpu-ID>
```
The following embedding-based models are implemented: `distmult`, `complex` and `conve`.

* Train HRL models (policy gradient + reward shaping)
```
./experiment-rs.sh configs/<dataset>-rs.sh --train <gpu-ID>
```

* Note: To train the HRL models, make sure 1) you have pre-trained the embedding-based models and 2) set the file path pointers to the pre-trained embedding-based models correctly ([example configuration file](configs/umls-rs.sh)).

## Evaluate models
To generate the evaluation results of a pre-trained model, simply change the `--train` flag in the commands above to `--inference`. 

For example, the following command performs inference with the HRL models and prints the evaluation results (on both dev and test sets).
```
./experiment-rs.sh configs/<dataset>-rs.sh --inference <gpu-ID>
```

* Note for the NELL-995 dataset: 

  On this dataset we split the original training data into `train.triples` and `dev.triples`, and the final model to test has to be trained with these two files combined. 
  1. To obtain the correct test set results, you need to add the `--test` flag to all data pre-processing, training and inference commands.  
    ```
    # You may need to adjust the number of training epochs based on the dev set development.

    ./experiment.sh configs/nell-995.sh --process_data <gpu-ID> --test
    ./experiment-emb.sh configs/nell-995-conve.sh --train <gpu-ID> --test
    ./experiment-rs.sh configs/NELL-995-rs.sh --train <gpu-ID> --test
    ./experiment-rs.sh configs/NELL-995-rs.sh --inference <gpu-ID> --test
    ```    
  2. Leave out the `--test` flag during development.
  
### Change the hyperparameters
To change the hyperparameters and other experiment set up, start from the configuration files.
