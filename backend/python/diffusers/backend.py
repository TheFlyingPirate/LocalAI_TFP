#!/usr/bin/env python3
from concurrent import futures
import traceback
import argparse
from collections import defaultdict
from enum import Enum
import signal
import sys
import time
import os

from PIL import Image
import torch

import backend_pb2
import backend_pb2_grpc

import grpc

from diffusers import (
    SanaPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
    StableDiffusionDepth2ImgPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
    StableDiffusionImg2ImgPipeline,
    AutoPipelineForText2Image,
    ControlNetModel,
    StableVideoDiffusionPipeline,
    Lumina2Text2ImgPipeline,
    GGUFQuantizationConfig
)
from diffusers.pipelines.stable_diffusion import safety_checker
from diffusers.utils import load_image, export_to_video
from compel import Compel, ReturnedEmbeddingsType
from optimum.quanto import freeze, qfloat8, quantize
from transformers import CLIPTextModel, T5EncoderModel
from safetensors.torch import load_file

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
COMPEL = os.environ.get("COMPEL", "0") == "1"
XPU = os.environ.get("XPU", "0") == "1"
CLIPSKIP = os.environ.get("CLIPSKIP", "1") == "1"
SAFETENSORS = os.environ.get("SAFETENSORS", "1") == "1"
CHUNK_SIZE = os.environ.get("CHUNK_SIZE", "8")
FPS = os.environ.get("FPS", "7")
DISABLE_CPU_OFFLOAD = os.environ.get("DISABLE_CPU_OFFLOAD", "0") == "1"
FRAMES = os.environ.get("FRAMES", "64")

if XPU:
    import intel_extension_for_pytorch as ipex
    print(ipex.xpu.get_device_name(0))

# If MAX_WORKERS are specified in the environment use it, otherwise default to 1
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))

# Disable safety checker (always allow outputs)
# https://github.com/CompVis/stable-diffusion/issues/239#issuecomment-1627615287
def sc(self, clip_input, images):
    return images, [False for _ in images]
safety_checker.StableDiffusionSafetyChecker.forward = sc

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)

class DiffusionScheduler(str, Enum):
    ddim = "ddim"  # DDIM
    pndm = "pndm"  # PNDM
    heun = "heun"  # Heun
    unipc = "unipc"  # UniPC
    euler = "euler"  # Euler
    euler_a = "euler_a"  # Euler a
    lms = "lms"  # LMS
    k_lms = "k_lms"  # LMS Karras
    dpm_2 = "dpm_2"  # DPM2
    k_dpm_2 = "k_dpm_2"  # DPM2 Karras
    dpm_2_a = "dpm_2_a"  # DPM2 a
    k_dpm_2_a = "k_dpm_2_a"  # DPM2 a Karras
    dpmpp_2m = "dpmpp_2m"  # DPM++ 2M
    k_dpmpp_2m = "k_dpmpp_2m"  # DPM++ 2M Karras
    dpmpp_sde = "dpmpp_sde"  # DPM++ SDE
    k_dpmpp_sde = "k_dpmpp_sde"  # DPM++ SDE Karras
    dpmpp_2m_sde = "dpmpp_2m_sde"  # DPM++ 2M SDE
    k_dpmpp_2m_sde = "k_dpmpp_2m_sde"  # DPM++ 2M SDE Karras

def get_scheduler(name: str, config: dict = {}):
    is_karras = name.startswith("k_")
    if is_karras:
        name = name.lstrip("k_")
        config["use_karras_sigmas"] = True

    mapping = {
        DiffusionScheduler.ddim: DDIMScheduler,
        DiffusionScheduler.pndm: PNDMScheduler,
        DiffusionScheduler.heun: HeunDiscreteScheduler,
        DiffusionScheduler.unipc: UniPCMultistepScheduler,
        DiffusionScheduler.euler: EulerDiscreteScheduler,
        DiffusionScheduler.euler_a: EulerAncestralDiscreteScheduler,
        DiffusionScheduler.lms: LMSDiscreteScheduler,
        DiffusionScheduler.dpm_2: KDPM2DiscreteScheduler,
        DiffusionScheduler.dpm_2_a: KDPM2AncestralDiscreteScheduler,
        DiffusionScheduler.dpmpp_2m: (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 2}),
        DiffusionScheduler.dpmpp_sde: DPMSolverSinglestepScheduler,
        DiffusionScheduler.dpmpp_2m_sde: (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
    }

    entry = mapping.get(name)
    if entry is None:
        raise ValueError(f"Invalid scheduler '{name}'")

    if isinstance(entry, tuple):
        cls, extra = entry
        config.update(extra)
    else:
        cls = entry

    return cls.from_config(config)

class BackendServicer(backend_pb2_grpc.BackendServicer):
    def Health(self, request, context):
        return backend_pb2.Reply(message=b"OK")

    def LoadModel(self, request, context):
        try:
            print(f"Loading model {request.Model}...", file=sys.stderr)
            print(f"Request: {request}", file=sys.stderr)

            # Determine dtype & single-file flag
            torch_type = torch.float16 if request.F16Memory else torch.float32
            local_file = bool(request.ModelFile and os.path.exists(request.ModelFile))
            model_path = request.ModelFile if local_file else request.Model
            from_single = model_path.startswith("/") or model_path.startswith("http") or local_file

            # Parse custom options
            self.options = {}
            for opt in request.Options:
                if ":" in opt:
                    k, v = opt.split(":",1)
                    self.options[k] = v

            self.cfg_scale = request.CFGScale or 7
            self.PipelineType = request.PipelineType
            self.clip_skip = request.CLIPSkip if CLIPSKIP and request.CLIPSkip else 0

            # Helper for GGUF vs non-GGUF
            def sf_load(cls, **kwargs):
                if model_path.endswith(".gguf"):
                    gguf_cfg = GGUFQuantizationConfig(compute_dtype=torch_type)
                    return cls.from_single_file(model_path, quantization_config=gguf_cfg, **kwargs)
                return cls.from_single_file(model_path, **kwargs)

            # Pipeline dispatch
            if self.PipelineType == "StableDiffusionImg2ImgPipeline" or (request.IMG2IMG and not self.PipelineType):
                self.pipe = sf_load(StableDiffusionImg2ImgPipeline, torch_dtype=torch_type) if from_single else StableDiffusionImg2ImgPipeline.from_pretrained(request.Model, torch_dtype=torch_type)

            elif self.PipelineType == "StableDiffusionDepth2ImgPipeline":
                self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(request.Model, torch_dtype=torch_type)

            elif self.PipelineType == "StableVideoDiffusionPipeline":
                self.img2vid = True
                self.pipe = StableVideoDiffusionPipeline.from_pretrained(request.Model, torch_dtype=torch_type)
                if not DISABLE_CPU_OFFLOAD:
                    self.pipe.enable_model_cpu_offload()

            elif self.PipelineType in ["AutoPipelineForText2Image", ""]:
                self.pipe = AutoPipelineForText2Image.from_pretrained(request.Model, torch_dtype=torch_type, use_safetensors=SAFETENSORS)

            elif self.PipelineType == "StableDiffusionPipeline":
                self.pipe = sf_load(StableDiffusionPipeline, torch_dtype=torch_type) if from_single else StableDiffusionPipeline.from_pretrained(request.Model, torch_dtype=torch_type)

            elif self.PipelineType == "DiffusionPipeline":
                self.pipe = DiffusionPipeline.from_pretrained(request.Model, torch_dtype=torch_type)

            elif self.PipelineType == "StableDiffusionXLPipeline":
                self.pipe = sf_load(StableDiffusionXLPipeline, torch_dtype=torch_type, use_safetensors=True) if from_single else StableDiffusionXLPipeline.from_pretrained(request.Model, torch_dtype=torch_type, use_safetensors=True)

            elif self.PipelineType == "StableDiffusion3Pipeline":
                self.pipe = sf_load(StableDiffusion3Pipeline, torch_dtype=torch_type, use_safetensors=True) if from_single else StableDiffusion3Pipeline.from_pretrained(request.Model, torch_dtype=torch_type, use_safetensors=True)

            elif self.PipelineType == "FluxPipeline":
                self.pipe = sf_load(FluxPipeline, torch_dtype=torch_type, use_safetensors=True) if from_single else FluxPipeline.from_pretrained(request.Model, torch_dtype=torch.bfloat16)
                if request.LowVRAM:
                    self.pipe.enable_model_cpu_offload()

            elif self.PipelineType == "FluxTransformer2DModel":
                dtype = torch.bfloat16
                transformer = FluxTransformer2DModel.from_single_file(model_path, torch_dtype=dtype)
                quantize(transformer, weights=qfloat8); freeze(transformer)
                te2 = T5EncoderModel.from_pretrained(os.environ.get("BFL_REPO", "ChuckMcSneed/FLUX.1-dev"), subfolder="text_encoder_2", torch_dtype=dtype)
                quantize(te2, weights=qfloat8); freeze(te2)
                self.pipe = FluxPipeline.from_pretrained(os.environ.get("BFL_REPO", "ChuckMcSneed/FLUX.1-dev"), transformer=None, text_encoder_2=None, torch_dtype=dtype)
                self.pipe.transformer = transformer
                self.pipe.text_encoder_2 = te2
                if request.LowVRAM:
                    self.pipe.enable_model_cpu_offload()

            elif self.PipelineType == "Lumina2Text2ImgPipeline":
                self.pipe = Lumina2Text2ImgPipeline.from_pretrained(request.Model, torch_dtype=torch.bfloat16)
                if request.LowVRAM:
                    self.pipe.enable_model_cpu_offload()

            elif self.PipelineType == "SanaPipeline":
                self.pipe = SanaPipeline.from_pretrained(request.Model, variant="bf16", torch_dtype=torch.bfloat16)
                self.pipe.vae.to(torch.bfloat16)
                self.pipe.text_encoder.to(torch.bfloat16)

            else:
                raise ValueError(f"Unsupported pipeline: {self.PipelineType}")

            # Scheduler override
            if request.SchedulerType:
                self.pipe.scheduler = get_scheduler(request.SchedulerType, self.pipe.scheduler.config)

            # Compel integration
            if COMPEL:
                self.compel = Compel(
                    tokenizer=[self.pipe.tokenizer, getattr(self.pipe, "tokenizer_2", None)],
                    text_encoder=[self.pipe.text_encoder, getattr(self.pipe, "text_encoder_2", None)],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True]
                )

            # ControlNet
            if request.ControlNet:
                cn = ControlNetModel.from_pretrained(request.ControlNet, torch_dtype=torch_type)
                self.pipe.controlnet = cn

            # LoRA adapters
            if request.LoraAdapter:
                la = request.LoraAdapter
                if not os.path.isabs(la):
                    la = os.path.join(request.ModelPath, la)
                if os.path.exists(la) and not os.path.isdir(la):
                    self.pipe.load_lora_weights(la)
                else:
                    self.pipe.unet.load_attn_procs(la)

            for idx, la in enumerate(request.LoraAdapters):
                path = la if os.path.isabs(la) else os.path.join(request.ModelPath, la)
                self.pipe.load_lora_weights(path, adapter_name=f"adapter_{idx}")
            if request.LoraScales:
                self.pipe.set_adapters([f"adapter_{i}" for i in range(len(request.LoraScales))], adapter_weights=request.LoraScales)

            # Move to device
            if request.CUDA:
                self.pipe.to('cuda')
                if request.ControlNet:
                    self.pipe.controlnet.to('cuda')
            if XPU:
                self.pipe = self.pipe.to("xpu")

        except Exception as err:
            return backend_pb2.Result(success=False, message=f"Unexpected {err}, {type(err)}")
        return backend_pb2.Result(message="Model loaded successfully", success=True)

    # https://github.com/huggingface/diffusers/issues/3064
    def load_lora_weights(self, checkpoint_path, multiplier, device, dtype):
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        state_dict = load_file(checkpoint_path, device=device)
        updates = defaultdict(dict)
        for key, value in state_dict.items():
            layer, elem = key.split('.', 1)
            updates[layer][elem] = value
        for layer, elems in updates.items():
            if "text" in layer:
                base = self.pipe.text_encoder; prefix="lora_te"
            else:
                base = self.pipe.unet; prefix="lora_unet"
            parts = layer.split(f"{prefix}_",1)[-1].split("_")
            curr = base
            temp = parts.pop(0)
            while True:
                try:
                    curr = getattr(curr, temp)
                    if not parts: break
                    temp = parts.pop(0)
                except:
                    temp = parts.pop(0)
            up = elems['lora_up.weight'].to(dtype)
            down = elems['lora_down.weight'].to(dtype)
            alpha = elems.get('alpha', torch.tensor( up.shape[1], dtype=torch.float32 )) 
            alpha = alpha.item() / up.shape[1]
            delta = (torch.mm(up.squeeze(-1).squeeze(-1), down.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
                     if up.ndim==4 else torch.mm(up, down))
            curr.weight.data += multiplier * alpha * delta

    def GenerateImage(self, request, context):
        prompt = request.positive_prompt
        steps = request.step or 1
        options = {"negative_prompt": request.negative_prompt, "num_inference_steps": steps}

        if request.src and not hasattr(self, "controlnet") and not getattr(self, "img2vid", False):
            options["image"] = Image.open(request.src)
        if getattr(self, "controlnet", None) and request.src:
            options["image"] = load_image(request.src)

        if CLIPSKIP and self.clip_skip:
            options["clip_skip"] = self.clip_skip

        keys = [k.strip() for k in (request.EnableParameters or "").split(",")] if request.EnableParameters else options.keys()
        if request.EnableParameters == "none":
            keys = []

        kwargs = {k: options[k] for k in keys if k in options}
        kwargs.update(self.options)

        if request.seed > 0:
            gen_device = 'cuda' if request.CUDA else 'cpu'
            kwargs["generator"] = torch.Generator(device=gen_device).manual_seed(request.seed)

        if self.PipelineType == "FluxPipeline":
            kwargs["max_sequence_length"] = 256

        if request.width: kwargs["width"] = request.width
        if request.height: kwargs["height"] = request.height

        if self.PipelineType == "FluxTransformer2DModel":
            kwargs["output_type"] = "pil"
            kwargs["generator"] = torch.Generator("cpu").manual_seed(0)

        if getattr(self, "img2vid", False):
            frames = self.pipe(options["image"], guidance_scale=self.cfg_scale, decode_chunk_size=CHUNK_SIZE, generator=torch.manual_seed(request.seed)).frames[0]
            export_to_video(frames, request.dst, fps=FPS)
            return backend_pb2.Result(message="Media generated successfully", success=True)

        if getattr(self, "txt2vid", False):
            vf = self.pipe(prompt, guidance_scale=self.cfg_scale, num_inference_steps=steps, num_frames=int(FRAMES)).frames
            export_to_video(vf, request.dst)
            return backend_pb2.Result(message="Media generated successfully", success=True)

        image = self.pipe(prompt, guidance_scale=self.cfg_scale, **kwargs).images[0]
        image.save(request.dst)
        return backend_pb2.Result(message="Media generated", success=True)

def serve(address):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ('grpc.max_message_length', 50 * 1024 * 1024),
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    print("Server started. Listening on: " + address, file=sys.stderr)

    def signal_handler(sig, frame):
        print("Received termination signal. Shutting down...", file=sys.stderr)
        server.stop(0)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument("--addr", default="localhost:50051", help="The address to bind the server to.")
    args = parser.parse_args()
    serve(args.addr)
# end of file
