conda create -n bumble_env python=3.10 -y

conda activate bumble_env

pip install -e .

可选 pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -e .[flash-attn] --no-build-isolation