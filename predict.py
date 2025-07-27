import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from PIL import Image
import os

class Predictor:
    def setup(self):
        # Load the base Stable Diffusion 1.5 model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-1-5",
            torch_dtype=torch.float16
        ).to("cuda")

        # Download LoRA file from Hugging Face
        self.lora_path = hf_hub_download(
            repo_id="Arunkalyan/apks",
            filename="lora.safetensors"
        )

        # Apply LoRA (requires diffusers 0.17.0+)
        self.pipe.load_lora_weights(self.lora_path)
        self.pipe.fuse_lora()

    def predict(self, prompt: str = "A futuristic cityscape at sunset") -> str:
        # Generate image
        image = self.pipe(prompt).images[0]

        # Save and return the path
        output_path = "/tmp/output.png"
        image.save(output_path)
        return output_path
