import runpod
import base64
import io
from PIL import Image
import app

def generate_video(job):
    job_input = job["input"]
    
    prompt = job_input.get("prompt", "make this image come alive, cinematic motion, smooth animation")
    negative_prompt = job_input.get("negative_prompt", app.default_negative_prompt)
    image_base64 = job_input.get("image_base64")
    duration_seconds = job_input.get("duration_seconds", 3.5)
    steps = job_input.get("steps", 6)
    
    if not image_base64:
        return {"error": "image_base64 is required for Image-to-Video."}
        
    try:
        # Decode image
        image_data = base64.b64decode(image_base64)
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Call the generate_video function from app.py
        # returns video_path, raw_video_path, current_seed
        video_component, video_path, current_seed = app.generate_video(
            input_image=init_image,
            last_image=None,
            prompt=prompt,
            steps=steps,
            negative_prompt=negative_prompt,
            duration_seconds=duration_seconds,
            guidance_scale=1.0,
            guidance_scale_2=1.0,
            seed=42,
            randomize_seed=True,
            quality=6,
            scheduler="UniPCMultistep",
            flow_shift=3.0,
            frame_multiplier=16,
            video_component=False,
            safe_mode=False,
            enable_safety_checker=False,
            progress=None # Mock or None, hopefully it accepts None
        )
        
        if not video_path:
            return {"error": "Failed to generate video (could be NSFW filter or internal error)."}
            
        # Read file and encode to base64
        with open(video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode("utf-8")
            
        import os
        os.remove(video_path)
        
        return {
            "status": "success",
            "video_base64": video_base64,
            "seed": current_seed
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": generate_video})
