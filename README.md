# ComfyUI_ChronoEdit_SM
[ChronoEdit](https://github.com/nv-tlabs/ChronoEdit): Towards Temporal Reasoning for Image Editing and World Simulation,you can use this node in comfyUI,and Vram >12G

# Update
* 推荐5帧8步lora/ use 5 frames 8 steps lora
* 如果跑29帧要跑50步/ if use 29 frames need 50 steps


1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_ChronoEdit_SM

```

2.requirements  
----

```
pip install -r requirements.txt
```

3.checkpoints 
----

1.gguf [links](https://huggingface.co/QuantStack/ChronoEdit-14B-GGUF/tree/main)   
2.lora [links](https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers/tree/main/lora)   
3.wan T5 clipvison vae [ links](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) 
```
├── ComfyUI/models/gguf
|     ├── ChronoEdit-14B-Q6_K.gguf # or Q8
├── ComfyUI/models/vae
|        ├──Wan2.1_VAE.pth
├── ComfyUI/models/clip
|        ├──umt5_xxl_fp8_e4m3fn_scaled.safetensors
|        ├──qwen_2.5_vl_7b.safetensors # OPTIONAL if use vl 可选 太慢
├── ComfyUI/models/clip_vision 
|        ├──clip_vision_h.safetensors
├── ComfyUI/models/loras 
|        ├──chronoedit_distill_lora.safetensors

```

# 4 Example
![](https://github.com/smthemex/ComfyUI_ChronoEdit_SM/blob/main/example_workflows/example.png)

# 5 Citation
```
@article{wu2025chronoedit,
    title={ChronoEdit: Towards Temporal Reasoning for Image Editing and World Simulation},
    author={Wu, Jay Zhangjie and Ren, Xuanchi and Shen, Tianchang and Cao, Tianshi and He, Kai and Lu, Yifan and Gao, Ruiyuan and Xie, Enze and Lan, Shiyi and Alvarez, Jose M. and Gao, Jun and Fidler, Sanja and Wang, Zian and Ling, Huan},
    journal={arXiv preprint arXiv:2510.04290},
    year={2025}
}
```
