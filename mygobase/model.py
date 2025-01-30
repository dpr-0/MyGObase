import io

import numpy as np
import torch
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
from transformers import AutoModelForCausalLM

model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)  # type: ignore
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).eval()  # type: ignore
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)  # type: ignore
tokenizer = vl_chat_processor.tokenizer  # type: ignore
cuda_device = "cpu"


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
        {
            "role": "Assistant",
            "content": """
                任務描述：

                你是一個多模態理解模型，能夠分析電影分鏡圖像與對應的台詞文本。請根據提供的分鏡圖像和台詞，完成以下任務：

                1. 場景描述：詳細描述分鏡中的場景，包括環境、人物動作、情緒氛圍等。

                2. 台詞分析：識別台詞的說話者，並分析該台詞與場景的關聯性，以及說話者的情緒或意圖。

                主要角色外貌描述：

                1. 高松燈：紫色短髮。主唱。故事設定為羽丘女子學園高中一年級生，感受性跟普通人不同，被稱為「羽丘的怪女生」。

                2. 千早愛音：粉色長頭髮。吉他手。故事設定為羽丘女子學園高中一年級生，成績優秀。使用樂器為ESP ULTRATONE，在加入MyGO!!!!!後才開始學電吉他。

                3. 要樂奈：異色瞳孔，淺色短髮。主音吉他手。故事設定為花咲川女子學園中學三年級生，不時在Live House「RiNG」出沒。

                4. 長崎爽世：金褐色長髮，貝斯手。故事設定為月之森女子學園高中一年級生。

                5. 椎名立希：深色長髮。故事設定為花咲川女子學園高中一年級生，在Live House「RiNG」打工。使用樂器為爵士鼓。

                6. 豊川祥子：淺藍色雙馬尾。羽丘女子學園的高一學生。前CRYCHIC的成員同時也是樂團創始人，負責作詞和鍵盤手。經常在學校的音樂教室彈鋼琴。

                輸入資料：

                <image_placeholder>\n台詞:[台詞]

                輸出要求：

                1. 場景描述：

                * 描述分鏡中的環境細節（如光線、背景、物件等）。

                * 描述人物的動作、表情、服裝等特徵。

                * 推測場景的情緒氛圍（如緊張、悲傷、歡樂等）。

                2. 台詞分析：

                * 推測說話者的身份（需推測角色名稱）。

                * 分析台詞的語氣、情緒（如憤怒、疑惑、威脅等）。

                * 解釋台詞與場景的關聯性，以及說話者的潛在動機。
""",
        },
    ]

    pil_images = [Image.open(io.BytesIO(image_data))]
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,  # type: ignore
        force_batchify=True,
    ).to(vl_gpt.device)
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
