conda create --name SESNN python=3.7
conda activate SESNN
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
pip install -r requirements.txt --no-cache-dir
