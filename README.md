## Environment Setup

To set up the enviroment you can easily run the following command:
```buildoutcfg
conda create -n TopoDiff python=3.8
conda activate TopoDiff
pip install -r requirements.txt
```

## SD pre-train model

Then you need to decide which Stable Diffusion Model you want to control. In this example, we will just use standard SD1.5. You can download it from the [official page of Stability](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). You want the file ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).


## Inference 
We provide an example inference script. The example outputs, including log file, generated images, config file,  are saved to the specified path `./example_output`.  Detail configuration can be found in the `./conf/base_config.yaml` and `inference.py`.
```buildoutcfg
python inference.py general.save_path=./example_output 
python inference_above_below.py general.save_path=./example_output
```
