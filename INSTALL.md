## GPU

```
conda create --name SESNN python=3.7
conda activate SESNN
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
pip install -r requirements.txt --no-cache-dir
```

## CPU

```
conda create --name SESNN python=3.7
conda activate SESNN
conda install pytorch torchvision cpuonly -c pytorch
pip install -r requirements.txt --no-cache-dir
```
