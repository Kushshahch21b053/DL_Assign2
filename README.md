# DL Assignment 2 - Part A

### Github Link for Part A

https://github.com/Kushshahch21b053/DL_Assign2_TaskA

### Wandb Report Link (Part A and B)

https://api.wandb.ai/links/ch21b053-indian-institute-of-technology-madras/usnef6k8

### Code Organisation for Part A

- config.py
- dataset.py
- model.py
- train.py
- main.py
- sweep.py
- test_model.py

### How to run code

- Firstly, if needed do:
```
pip install -r requirements.txt
```

**Part A Question 1-3:**
- To run the sweep, login into wandb, and then run the following
```
python sweep.py --create \
               --project (your_project_name) \
               --entity (your_entity_name)
```
- This will give a sweep_id, then use it as follows to start the agent.
- Then, 
```
python sweep.py --sweep_id (your_sweep_id) \
               --count 20 \
               --project (your_project_name) \
               --entity (your_entity_name)
```
- To run without wand for specific hyperparameter configuration, see the following example on the best configuration:
```
python main.py \
  --data_dir nature_12K\inaturalist_12K \
  --filters_per_layer 32 \
  --filter_organization same \
  --activation relu \
  --learning_rate 0.001 \
  --batch_size 32 \
  --epochs 10 \
  --weight_decay 0.0 \
  --use_batchnorm \
  --dropout_rate 0.0 \
  --optimizer adam
```

**Part A Question 4**
- To test on the best configuration:
```
python test_model.py --data_dir (your data directory)
```



