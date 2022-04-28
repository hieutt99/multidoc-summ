# Summarization 


## Run 

train BERT+Transformer

```bash 
! python train.py -mode train -encoder transformer -dropout 0.1\
    -bert_data_path ./bert_data/cnndm -model_path ./saved_models/bert_transformer\
    -lr 2e-3 -visible_gpus 0  -gpu_ranks 0 -world_size 3 -report_every 50\
    -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000\
    -accum_count 2 -log_file ./logs/bert_transformer -use_interval true\
    -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8
```