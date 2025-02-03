from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from pydantic import BaseModel


class Finding(BaseModel):
    summary: str
    explanation: str


class Report(BaseModel):
    title: str
    summary: str
    rating: float
    rating_explanation: str
    findings: List[Finding]


json_schema = ResponseFormatJSONSchema(
    json_schema=JSONSchema(
        name="cr", description="", schema=Report.model_json_schema()
    ),
    type="json_schema",
)

COMMUNITY_REPORT_PROMPT = """
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Input Format
-----------
Text:

<Source Entity> -> <Relation> -> <Target Entity>
<Source Entity> -> <Relation> -> <Target Entity>
<Source Entity> -> <Relation> -> <Target Entity>



# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
{input_text}

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

Output:"""


class CommunityReporter:
    def __init__(self) -> None:
        self.MODEL = "gemma-2-9b-it"
        self.client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

    def extract(self, text: str):
        chat_completion: ChatCompletion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": COMMUNITY_REPORT_PROMPT.format(input_text=text),
                }
            ],  # type: ignore
            model=self.MODEL,
            stream=False,
            max_tokens=-1,
            temperature=0.3,
            response_format=json_schema,
        )  # type: ignore

        content: str = chat_completion.choices[0].message.content  # type: ignore
        return Report.model_validate_json(content)


class Summarizer:
    def __init__(self) -> None:
        self.MODEL = "gemma-2-9b-it"
        self.client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

    def summarize(self, query, contexnt: str) -> str:
        chat_completion: ChatCompletion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"根據此問題:{query}, 摘要以下內容可能可以回答問題的重點: {contexnt}",
                }
            ],  # type: ignore
            model=self.MODEL,
            stream=False,
            max_tokens=-1,
            temperature=0.7,
        )  # type: ignore

        content: str = chat_completion.choices[0].message.content  # type: ignore
        return content
