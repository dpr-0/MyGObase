from typing import List

from openai import OpenAI

from mygobase import BASE_URL, EMBEDD_MODEL


class EmbeddingExtractor:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=BASE_URL, api_key="lm-studio")

    def extract(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            input=text, model=EMBEDD_MODEL, encoding_format="float"
        )
        embedding = resp.data[0].embedding
        return embedding
