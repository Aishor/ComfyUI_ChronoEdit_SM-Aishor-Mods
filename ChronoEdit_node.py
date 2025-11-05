 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from .Chrono.scripts.run_inference_diffusers import load_dit_model,load_prompt_enhancer,prompt_enhance,inference_chrono,load_prompt_enhancer_cf
from .model_loader_utils import tensor_upscale,tensor2image
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
from pathlib import PureWindowsPath
from transformers import AutoProcessor
import comfy.model_management as mm
MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)

folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir


class ChronoEdit_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Model",
            display_name="ChronoEdit_SM_Model",
            category="ChronoEdit_SM",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
            ],
            outputs=[
                io.Custom("ChronoEdit_SM_Model").Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, dit,gguf) -> io.NodeOutput:
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        model=load_dit_model(dit_path, gguf_path,node_cr_path)
        return io.NodeOutput(model)
    
class ChronoEdit_SM_Lora(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Lora",
            display_name="ChronoEdit_SM_Lora",
            category="ChronoEdit_SM",
            inputs=[
                io.Custom("ChronoEdit_SM_Model").Input("model"),
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras") ),
                io.Float.Input("lora_scale", default=1, min=0.1, max=1.0,step=0.1, round=0.01,),
            ],
            outputs=[
                io.Custom("ChronoEdit_SM_Model").Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, model,lora,lora_scale) -> io.NodeOutput:
        lora_path=folder_paths.get_full_path("loras", lora) if lora != "none" else None
        if lora_path is not None:
            print(f"Loading LoRA weights from {lora_path}...")
            all_adapters = model.get_list_adapters()
            dit_list=[]
            if all_adapters:
                dit_list= all_adapters['transformer']

                if dit_list:
                    if os.path.splitext(os.path.basename(lora_path))[0] in dit_list:
                        pass
                    else:
                        model.load_lora_weights(lora_path, adapter_name=os.path.splitext(os.path.basename(lora_path))[0])
                else:
                    model.load_lora_weights(lora_path,adapter_name= os.path.splitext(os.path.basename(lora_path))[0])
            else:
                model.load_lora_weights(lora_path,adapter_name= os.path.splitext(os.path.basename(lora_path))[0])
            model.set_adapters([os.path.splitext(os.path.basename(lora_path))[0]], adapter_weights=[lora_scale])
        try:
            dit_list= model.get_list_adapters()['transformer']
            for name in dit_list:
                if lora_path is not None:
                    if name!=os.path.splitext(os.path.basename(lora_path))[0]:
                        model.delete_adapters(name)
                        print(f"当前激活的适配器: {os.path.splitext(os.path.basename(lora_path))[0]}")  
                else:
                    model.delete_adapters(name)
        except:pass

        return io.NodeOutput(model)
    
class ChronoEdit_SM_Vae(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Vae",
            display_name="ChronoEdit_SM_Vae",
            category="ChronoEdit_SM",
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Combo.Input("vae_decoder",options= ["none"] + folder_paths.get_filename_list("vae") ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                ],
            )
    @classmethod
    def execute(cls, vae,latent,vae_decoder) -> io.NodeOutput:
        samples=latent["samples"]
        if samples.shape[2]>2: #torch.Size([1, 16, 8, 60, 104])
            if vae_decoder!="none":
                video_edit = vae.decode(samples[:, :, [0, -1]])
                video_reason =vae.decode(samples[:, :, :-1])
                # print(f"Video edit shape: {video_edit.shape}") #Video edit shape: torch.Size([1, 5, 480, 832, 3])
                # print(f"video_reason edit shape: {video_reason.shape}") #Video edit shape: torch.Size([1, 25, 480, 832, 3])
                #images = torch.cat([video_reason, video_edit[:, :, 1:]], dim=2)
                images = torch.cat([video_reason, video_edit[:, 1:, :, :]], dim=1)
            else:
                images = vae.decode(samples)
        else:
            images = vae.decode(samples)
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return io.NodeOutput(images)


class ChronoEdit_SM_Enhance_Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Enhance_Loader",
            display_name="ChronoEdit_SM_Enhance_Loader",
            category="ChronoEdit_SM",
            inputs=[
                io.String.Input("repo", multiline=False, default="Qwen/Qwen2.5-VL-7B-Instruct"),
                io.Combo.Input("clip",options= ["none"] + folder_paths.get_filename_list("clip")),
            ],
            outputs=[
                io.Custom("ChronoEdit_SM_Model_En").Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, repo,clip) -> io.NodeOutput:
        clip_path=folder_paths.get_full_path("clip", clip) if clip != "none" else None

        if clip_path is not None:
            if "2.5" in clip_path.lower() or "qwen2" in clip_path.lower(): 
                model_name = os.path.join(node_cr_path,"Qwen/Qwen2.5-VL-7B-Instruct") 
            elif "qwen3" in clip_path.lower(): 
                model_name = os.path.join(node_cr_path,"Qwen/Qwen3-VL-8B-Instruct")
            else:
                raise ValueError(f"Unsupported model: {clip_path}")
            model=load_prompt_enhancer_cf(clip_path,model_name)
            processor = AutoProcessor.from_pretrained(model_name)
        elif repo:
            repo=PureWindowsPath(repo)
            model,processor=load_prompt_enhancer(repo)
        else:
            raise ValueError(f"Unsupported model: {repo} or {clip}")
        pipe={"model":model,"processor":processor,}
        return io.NodeOutput(pipe)

class ChronoEdit_SM_Enhance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Enhance",
            display_name="ChronoEdit_SM_Enhance",
            category="ChronoEdit_SM",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("prompt", multiline=True, default="Transform the image so that inside the floral teacup of steaming tea, a small, cute mouse is sitting and taking a bath; the mouse should look relaxed and cheerful, with a tiny white bath towel draped over its head as if enjoying a spa moment, while the steam rises gently around it, blending seamlessly with the warm and cozy atmosphere."),
                io.Custom("ChronoEdit_SM_Model_En").Input("model"),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                ],
            )
    @classmethod
    def execute(cls, image,prompt,model) -> io.NodeOutput:
        image=tensor2image(image)
        prompt=prompt_enhance(image,prompt,model["model"],model["processor"])

        return io.NodeOutput(prompt)


class ChronoEdit_SM_Latent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="ChronoEdit_SM_Latent",
            display_name="ChronoEdit_SM_Latent",
            category="ChronoEdit_SM",
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Int.Input("width", default=832, min=128, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=480, min=128, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.Combo.Input("num_frames",options= [5,29]),
            ],
            outputs=[
                io.Latent.Output(display_name="Latent"),
                io.Int.Output(display_name="num_frames"),
                ],
            )
    @classmethod
    def execute(cls, vae,image,width,height,num_frames) -> io.NodeOutput:
        image = tensor_upscale(image, width, height)
        image = torch.cat([image, image.new_zeros(int(num_frames) - 1, image.shape[1],  image.shape[2], image.shape[3])], dim=0)
        Latent=vae.encode(image)
        return io.NodeOutput(Latent,num_frames)

class ChronoEdit_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ChronoEdit_SM_KSampler",
            display_name="ChronoEdit_SM_KSampler",
            category="ChronoEdit_SM",
            inputs=[
                io.Custom("ChronoEdit_SM_Model").Input("model"),
                io.Latent.Input("image_latent"), #1, 16, 2, 60, 104
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Int.Input("num_frames",force_input=True),   
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Float.Input("flow_shift", default=2.0, min=0.1, max=10.0, step=0.1, round=0.01,),
                io.Int.Input("steps", default=8, min=1, max=10000),
                io.Float.Input("guidance_scale", default=1.0, min=0.1, max=10.0, step=0.1, round=0.01,),
                io.Int.Input("num_temporal_reasoning_steps", default=50, min=0, max=MAX_SEED,step=1,),
                io.Boolean.Input("offload_model", default=True),
                io.Int.Input("block_num", default=1, min=1, max=48,step=1),
          
            ],
            outputs=[
                io.Latent.Output(display_name="Latent"),
            ],
        )
    @classmethod
    def execute(cls, model,image_latent,positive,negative,num_frames,seed,flow_shift, steps, 
                guidance_scale,num_temporal_reasoning_steps,offload_model,block_num) -> io.NodeOutput:
        cf_models=mm.loaded_models()
        try:
            for pipe in cf_models:   
                pipe.unpatch_model(device_to=torch.device("cpu"))
                print(f"Unpatching models.{pipe}")
        except: pass
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        max_gpu_memory = torch.cuda.max_memory_allocated()
        print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")
        width,height=image_latent.shape[3]*8,image_latent.shape[4]*8
        Latent=inference_chrono (model,image_latent,positive[0][0],positive[0][1]["unclip_conditioning"][0]["clip_vision_output"]["penultimate_hidden_states"],negative[0][0],flow_shift,seed,steps,
                                 width,height,num_frames,num_temporal_reasoning_steps,guidance_scale,offload_model,block_num,device)
        out={}
        out["samples"]=Latent #torch.Size([1, 16, 2, 60, 104])
        return io.NodeOutput(out)


from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/ChronoEdit_SM_Extension")
async def get_hello(request):
    return web.json_response("ChronoEdit_SM_Extension")

class ChronoEdit_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ChronoEdit_SM_Model,
            ChronoEdit_SM_Lora,
            ChronoEdit_SM_Enhance_Loader,
            ChronoEdit_SM_Enhance,
            ChronoEdit_SM_Vae,
            ChronoEdit_SM_Latent,
            ChronoEdit_SM_KSampler,
        ]
async def comfy_entrypoint() -> ChronoEdit_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return ChronoEdit_SM_Extension()



