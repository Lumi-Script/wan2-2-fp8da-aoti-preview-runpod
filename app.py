import spaces
import torch
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
import gc
import copy

from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization import Int8WeightOnlyConfig

import aoti

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SASolverScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepInverseScheduler,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
)


MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 160

MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

SCHEDULER_MAP = {
    "FlowMatchEulerDiscrete": FlowMatchEulerDiscreteScheduler,
    "SASolver": SASolverScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "DPMSolverMultistepInverse": DPMSolverMultistepInverseScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
}

pipe = WanImageToVideoPipeline.from_pretrained(
    "TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING",
    torch_dtype=torch.bfloat16,
).to('cuda')
original_scheduler = copy.deepcopy(pipe.scheduler)
print(original_scheduler)

quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())

aoti.aoti_blocks_load(pipe.transformer, 'zerogpu-aoti/Wan2', variant='fp8da')
aoti.aoti_blocks_load(pipe.transformer_2, 'zerogpu-aoti/Wan2', variant='fp8da')


default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"


def resize_image(image: Image.Image) -> Image.Image:
    """
    Resizes an image to fit within the model's constraints, preserving aspect ratio as much as possible.
    """
    width, height = image.size

    # Handle square case
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)

    aspect_ratio = width / height

    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM

    image_to_resize = image

    if aspect_ratio > MAX_ASPECT_RATIO:
        # Very wide image -> crop width to fit 832x480 aspect ratio
        target_w, target_h = MAX_DIM, MIN_DIM
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        # Very tall image -> crop height to fit 480x832 aspect ratio
        target_w, target_h = MIN_DIM, MAX_DIM
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))
    else:
        if width > height:  # Landscape
            target_w = MAX_DIM
            target_h = int(round(target_w / aspect_ratio))
        else:  # Portrait
            target_h = MAX_DIM
            target_w = int(round(target_h * aspect_ratio))

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF

    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))

    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)


def resize_and_crop_to_match(target_image, reference_image):
    """Resizes and center-crops the target image to match the reference image's dimensions."""
    ref_width, ref_height = reference_image.size
    target_width, target_height = target_image.size
    scale = max(ref_width / target_width, ref_height / target_height)
    new_width, new_height = int(target_width * scale), int(target_height * scale)
    resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left, top = (new_width - ref_width) // 2, (new_height - ref_height) // 2
    return resized.crop((left, top, left + ref_width, top + ref_height))


def get_num_frames(duration_seconds: float):
    return 1 + int(np.clip(
        int(round(duration_seconds * FIXED_FPS)),
        MIN_FRAMES_MODEL,
        MAX_FRAMES_MODEL,
    ))


def get_inference_duration(
    resized_image,
    processed_last_image,
    prompt,
    steps,
    negative_prompt,
    num_frames,
    guidance_scale,
    guidance_scale_2,
    current_seed,
    scheduler_name,
    flow_shift,
    progress
):
    BASE_FRAMES_HEIGHT_WIDTH = 81 * 832 * 624
    BASE_STEP_DURATION = 15
    width, height = resized_image.size
    factor = num_frames * width * height / BASE_FRAMES_HEIGHT_WIDTH
    step_duration = BASE_STEP_DURATION * factor ** 1.5
    return 5 + int(steps) * step_duration


@spaces.GPU(duration=get_inference_duration)
def run_inference(
    resized_image,
    processed_last_image,
    prompt,
    steps,
    negative_prompt,
    num_frames,
    guidance_scale,
    guidance_scale_2,
    current_seed,
    scheduler_name,
    flow_shift,
    progress=gr.Progress(track_tqdm=True),
):

    scheduler_class = SCHEDULER_MAP.get(scheduler_name)
    if scheduler_class.__name__ != pipe.scheduler.config._class_name or flow_shift != pipe.scheduler.config.get("flow_shift", "shift"):
        config = copy.deepcopy(original_scheduler.config)
        if scheduler_class == FlowMatchEulerDiscreteScheduler:
            config['shift'] = flow_shift
        else:
            config['flow_shift'] = flow_shift
        pipe.scheduler = scheduler_class.from_config(config)

    result = pipe(
        image=resized_image,
        last_image=processed_last_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
    ).frames[0]

    pipe.scheduler = original_scheduler
    return result


def generate_video(
    input_image,
    last_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=MAX_DURATION,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    quality=5,
    scheduler="UniPCMultistep",
    flow_shift=6.0,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate a video from an input image using the Wan 2.2 14B I2V model with Lightning LoRA.

    This function takes an input image and generates a video animation based on the provided
    prompt and parameters. It uses an FP8 qunatized Wan 2.2 14B Image-to-Video model in with Lightning LoRA
    for fast generation in 4-8 steps.

    Args:
        input_image (PIL.Image): The input image to animate. Will be resized to target dimensions.
        last_image (PIL.Image, optional): The optional last image for the video.
        prompt (str): Text prompt describing the desired animation or motion.
        steps (int, optional): Number of inference steps. More steps = higher quality but slower.
            Defaults to 4. Range: 1-30.
        negative_prompt (str, optional): Negative prompt to avoid unwanted elements.
            Defaults to default_negative_prompt (contains unwanted visual artifacts).
        duration_seconds (float, optional): Duration of the generated video in seconds.
            Defaults to 2. Clamped between MIN_FRAMES_MODEL/FIXED_FPS and MAX_FRAMES_MODEL/FIXED_FPS.
        guidance_scale (float, optional): Controls adherence to the prompt. Higher values = more adherence.
            Defaults to 1.0. Range: 0.0-20.0.
        guidance_scale_2 (float, optional): Controls adherence to the prompt. Higher values = more adherence.
            Defaults to 1.0. Range: 0.0-20.0.
        seed (int, optional): Random seed for reproducible results. Defaults to 42.
            Range: 0 to MAX_SEED (2147483647).
        randomize_seed (bool, optional): Whether to use a random seed instead of the provided seed.
            Defaults to False.
        quality (float, optional): Video output quality. Default is 5. Uses variable bit rate.
            Highest quality is 10, lowest is 1.
        scheduler (str, optional): The name of the scheduler to use for inference. Defaults to "UniPCMultistep".
        flow_shift (float, optional): The flow shift value for compatible schedulers. Defaults to 6.0.
        progress (gr.Progress, optional): Gradio progress tracker. Defaults to gr.Progress(track_tqdm=True).

    Returns:
        tuple: A tuple containing:
            - video_path (str): Path for the video component.
            - video_path (str): Path for the file download component. Attempt to avoid reconversion in video component.
            - current_seed (int): The seed used for generation.

    Raises:
        gr.Error: If input_image is None (no image uploaded).

    Note:
        - Frame count is calculated as duration_seconds * FIXED_FPS (24)
        - Output dimensions are adjusted to be multiples of MOD_VALUE (32)
        - The function uses GPU acceleration via the @spaces.GPU decorator
        - Generation time varies based on steps and duration (see get_duration function)
    """
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    processed_last_image = None
    if last_image:
        processed_last_image = resize_and_crop_to_match(last_image, resized_image)

    output_frames_list = run_inference(
        resized_image,
        processed_last_image,
        prompt,
        steps,
        negative_prompt,
        num_frames,
        guidance_scale,
        guidance_scale_2,
        current_seed,
        scheduler,
        flow_shift,
        progress,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name

    export_to_video(output_frames_list, video_path, fps=FIXED_FPS, quality=quality)

    return video_path, video_path, current_seed


with gr.Blocks(theme=gr.themes.Soft(), delete_cache=(12800, 12800)) as demo:
    gr.Markdown("# WAMU V2 - Wan 2.2 I2V (14B) 🐢")
    gr.Markdown("## ℹ️ **A Note on Performance:** This version prioritizes a straightforward setup over maximum speed, so performance may vary.")
    gr.Markdown("run Wan 2.2 in just 4-8 steps, fp8 quantization & AoT compilation - compatible with 🧨 diffusers and ZeroGPU")
    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Input Image")
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            duration_seconds_input = gr.Slider(minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=3.5, label="Duration (seconds)", info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps.")
            steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=6, label="Inference Steps")

            with gr.Accordion("Advanced Settings", open=False):
                last_image_component = gr.Image(type="pil", label="Last Image (Optional)")
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, info="Used if any Guidance Scale > 1.", lines=3)
                quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=6, label="Video Quality")
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale - high noise stage")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 2 - low noise stage")
                scheduler_dropdown = gr.Dropdown(
                    label="Scheduler",
                    choices=list(SCHEDULER_MAP.keys()),
                    value="UniPCMultistep",
                    info="Select a custom scheduler."
                )
                flow_shift_slider = gr.Slider(minimum=0.5, maximum=15.0, step=0.1, value=3.0, label="Flow Shift")

            generate_button = gr.Button("Generate Video", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Generated Video", autoplay=True, interactive=False)
            file_output = gr.File(label="Download Video")

    ui_inputs = [
        input_image_component, last_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input, seed_input, randomize_seed_checkbox,
        quality_slider, scheduler_dropdown, flow_shift_slider,
    ]
    generate_button.click(fn=generate_video, inputs=ui_inputs, outputs=[video_output, file_output, seed_input])


if __name__ == "__main__":
    demo.queue().launch(
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
    )