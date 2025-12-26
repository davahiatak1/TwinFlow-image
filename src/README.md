
## Quick Start

To run TwinFlow on OpenUni, first you need to configure the environment as [OpenUni](https://github.com/wusize/OpenUni) repo does.

Secondly, you need to download the models to your path:

```
https://huggingface.co/wusize/openuni/blob/main/openuni_l_internvl3_2b_sana_1_6b_512_hf_blip3o60k.pth
https://huggingface.co/OpenGVLab/InternVL3-2B
https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_diffusers
https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers
```

After configure the environment and models, run the command to train TwinFlow on OpenUni:

```sh
src/scripts/openuni/train_ddp.sh src/configs/openuni_task/openuni_full.yaml
```

To run data free training (which does not need text-image pairs):
```sh
src/scripts/openuni/train_ddp.sh src/configs/openuni_task/openuni_full_imgfree.yaml
```