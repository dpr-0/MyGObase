import io

import numpy as np
import torch
from janus.models import VLChatProcessor
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM

model_path = "deepseek-ai/Janus-1.3B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = "eager"
vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path, language_config=language_config, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda()

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer  # type: ignore
cuda_device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def multimodal_understanding(
    image_data: bytes, question: str, seed=42, top_p=0.95, temperature=0.1
) -> str:
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            "images": [image_data],
        },
        {"role": "Assistant", "content": ""},
    ]

    pil_images = [Image.open(io.BytesIO(image_data))]
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,  # type: ignore
        force_batchify=True,
    ).to(  # type: ignore
        cuda_device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer
