# Appling OpenDelta to GLUE/SuperGLUE tasks using Seq2Seq Paradigm


## install the repo
```bash
cd ../
python setup_seq2seq.py develop
```
This will add `examples_seq2seq` to the environment path of the python lib.

## Generating the json configuration file

```
python config_gen.py --job $job_name

```
The available job configuration (e.g., `--job lora_t5-base`) can be seen from `config_gen.py`. You can also
create your only configuration.


## Run the code

```
python run_seq2seq.py configs/$job_name/$dataset.json
```

## Link to the original training scripts
This example repo is based on the [compacter training scripts](https://github.com/rabeehk/compacter), with compacter-related lines removed. Thanks to the authors of the original repo. In addition, in private correspondence with the authors, they shared the codes to create the json configs. Thanks again for their efforts. 
