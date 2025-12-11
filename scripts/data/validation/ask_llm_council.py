# Taken from https://huggingface.co/spaces/burtenshaw/karpathy-llm-council
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".." / ".." / ".." / ".env.example")

import llm_council.backend.config as llm_council_config
from llm_council.backend.council import run_full_council
import asyncio

def change_model_in_council(previous_model, new_model):
    if  previous_model in llm_council_config.COUNCIL_MODELS:
        llm_council_config.COUNCIL_MODELS.remove(previous_model)
    if new_model not in llm_council_config.COUNCIL_MODELS:
        llm_council_config.COUNCIL_MODELS.append(new_model)

def ask_llm_council(question: str, response_format=None) -> str:
    # 1. Update Anthropic model of council to Claude Opus 4.5:
    change_model_in_council("anthropic/claude-sonnet-4.5", "anthropic/claude-opus-4.5")
    print(f"LLM council : {llm_council_config.COUNCIL_MODELS}")
    print(f"Chairman model: {llm_council_config.CHAIRMAN_MODEL}")

    """
    Ask the LLM Council a question.

    The council consists of multiple advanced LLMs (currently: {models}) that:
    1. Individually answer the question
    2. Rank each other's answers
    3. Synthesize a final best answer (Chairman: {chairman})

    Args:
        question: The user's question to be discussed by the council.

    Returns:
        The final synthesized answer from the Council Chairman.
    """.format(models=", ".join([m.split("/")[-1] for m in llm_council_config.COUNCIL_MODELS]),
               chairman=llm_council_config.CHAIRMAN_MODEL.split("/")[-1])

    try:
        # Run the council
        # run_full_council returns (stage1, stage2, stage3, metadata)
        _, _, stage3_result, _ = asyncio.run(run_full_council(question, response_format))

        response = stage3_result.get("response")
        if not response:
            return "The council failed to generate a response."

        return response

    except Exception as e:
        return f"Error consulting the council: {str(e)}"

def get_council():
    # Change Claude to Opus if not changed yet:
    change_model_in_council("anthropic/claude-sonnet-4.5", "anthropic/claude-opus-4.5")

    return {"LLM council": llm_council_config.COUNCIL_MODELS,
            "Chairman": llm_council_config.CHAIRMAN_MODEL}
