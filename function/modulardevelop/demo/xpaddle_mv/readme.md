run the following code to avoid the input shape check of X2paddle and install the tools to convert models.

```
pip install -r requirements.txt
cp -r ./onnx_decoder.py your_env/site-packages/x2paddle/decoder/onnx_decoder.py
cp -r ./common.py your_env/site-packages/onnx2torch/utils/common.py
```


please replace the `source activate tf2.3` in this the function `paddle_convert` of [file](../develop.py) to `source activate YOUR_CONDA_ENVS_NAME`.