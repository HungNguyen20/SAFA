# Inorder to use this dataset, pull the repository
``` bash
git clone https://github.com/AIoT-Lab-BKAI/datasets.git
```

The seed for random split MUST BE THE SAME with the seed with which the data was divided
```python
training_data, testing_data = random_split(total_data, [80000, 20000], generator=torch.Generator().manual_seed(20022000))
```