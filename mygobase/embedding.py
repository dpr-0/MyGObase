from openai import OpenAI


class EmbeddingExtractor:
    def __init__(self) -> None:
        self.MODEL = "text-embedding-nomic-embed-text-v1.5@f32"
        self.client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

    def extract(self, text: str):
        resp = self.client.embeddings.create(
            input=text, model=self.MODEL, encoding_format="float"
        )
        embedding = resp.data[0].embedding

        # print(len(embedding))
        return embedding
