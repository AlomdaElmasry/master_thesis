# Video Inpainting Thesis - 2020
The code included in this repository uses Skeltorch as structuring framework. This framework is based on the creation of
experiments, which allow the easy distribution of checkpoints and replication. When working with Skeltorch, the 
experiments are saved inside `/experiments`. Remember to place in that folder the experiments that you may download.

With respect other packages and folders available in this repository:
- `datasets`: contains files related to data handling.
- `copy_paste [Skeltorch]`: includes our replication of the paper "Copy-and-Paste Networks for Deep Video Inpainting".
Based on the [official checkpoint of the paper](https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting).
- `scripts`: includes scripts to perform visualization tasks.
- `utils`: includes auxiliary classes which help with data handling and other non-modeling tasks.

Every Skeltorch project has one configuration file associated. This configuration file is only required when creating
the experiment. No further changes on configuration parameters can be made once the experiment has been created. In 
order them, you are required to create another different experiment.

- `config.cp.json`: configuration file used to create experiments related to the `copy_paste` package.

## Running copy_paste Package

This package includes the training and testing of the model published in "Copy-and-Paste Networks for Deep Video 
Inpainting". The has been adapted to work with with Skeltorch. To use it, download our adapted 
["cp_official_release" experiment]().

+ **Creating a new experiment**

```
python -m copy_paste --data-path <data_path> --exp-name <exp_name> init --config-path <config_path>
```

+ **Train the model (coming soon)**

```
python -m copy_paste --data-path <data_path> --exp-name <exp_name> train
```

+ **Infer the inpainted version of test sequences**

```
python -m copy_paste --data-path <data_path> --exp-name <exp_name> test --epoch <epoch_n> --data-output <data_output>
```

+ **Infer the aligned versions of test sequences with respect to the first frames**

```
python -m copy_paste --data-path <data_path> --exp-name <exp_name> test_alignment --epoch <epoch_n> --data-output <data_output>
```