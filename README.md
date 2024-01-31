# ASR project barebones

## Installation guide

On your fist step run installations of requirements (I used python3.8)
```shell
pip install -r ./requirements.txt
```
Second step is to load dataset

```shell
bash loader.sh
```
After it you can run train on your config by command:
```shell
python3 train.py -c PATH/TO/YOUR/CONFIG/CONFIG_NAME.json 
```
I ran with tran.json config:
```shell
%%python3 train.py -c hw_nv/configs/train.json
```

To test my model you may run:
```shell
%%python3 test.py -r model_info/checkpoint.pth  
```
Before it you should save in model_info my checkpoint: https://disk.yandex.ru/d/94DTgWd55LWW1Q. -r is a path to directory with config and checkpoint. -i is a path to directory with wavs ans audios.
