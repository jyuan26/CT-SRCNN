yes | conda create -n srcnn2 python=3.7
eval "$(conda shell.bash hook)"
conda activate srcnn2
yes | conda histpytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirement.txt
