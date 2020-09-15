## GPU

```
conda create --name A2SNN python=3.7
conda activate A2SNN
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
pip install -r requirements.txt --no-cache-dir
```

## CPU

```
conda create --name A2SNN python=3.7
conda activate A2SNN
conda install pytorch torchvision cpuonly -c pytorch
pip install -r requirements.txt --no-cache-dir
```

## Installing Foolbox

Extended with EoT, required for adversarial testing.

```
git clone https://github.com/peustr/foolbox.git
cd foolbox
pip install -e .
```
