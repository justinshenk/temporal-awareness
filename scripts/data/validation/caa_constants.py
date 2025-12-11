no_explicit_temporal_words_rule = "- temporal_keywords: No explicit temporal words (now, immediate, future, lasting, years, soon, quick, urgent, permanent, decade)"

allowed_explicit_temporal_words_rule = " - temporal_keywords: No explicit temporal words (now, immediate, future, lasting, years, soon, quick, urgent, permanent, decade). However, make an exception for a given example and rate this factor with 5 for it everywhere."

validation_prompt = """You are validating contrastive pairs for a temporal scope dataset used to train steering vectors.

For each pair, score these factors from 1-5:
  1 = Poor (significant issue, needs rewrite)
  2 = Weak (notable issue, needs revision)
  3 = Acceptable (minor issues, usable)
  4 = Good (no significant issues)
  5 = Excellent (ideal example)

PAIR:
Question: {question}
(A) {option_a}
(B) {option_b}
Category: {category}
Stated labels: (A) = immediate, (B) = long-term

FACTORS TO SCORE:

1. LEXICAL
   {no_explicit_temporal_words_rule}
   - vocabulary_balance: Both options use similarly common/complex words
   - ngram_leakage: No idiomatic phrases that give away temporal scope

2. SURFACE
   - length_match: Options within 50% character length
   - structure: Grammatically parallel

3. SEMANTIC
   - formality: Same register
   - hedging: Same certainty level
   - specificity: Same abstraction level
   - sentiment: Both neutral or matched valence

4. VALIDITY
   - clear_temporal: Unambiguous immediate vs long-term distinction
   - both_valid: Both options are sensible responses
   - single_dimension: Options differ ONLY on temporal scope
   - labels_correct: The immediate/long-term labels are accurate
"""

validation_prompt_return_hint = """\n\nReturn JSON:
{
  "scores": {
    "temporal_keywords": 1-5,
    "vocabulary_balance": 1-5,
    "ngram_leakage": 1-5,
    "length_match": 1-5,
    "structure": 1-5,
    "formality": 1-5,
    "hedging": 1-5,
    "specificity": 1-5,
    "sentiment": 1-5,
    "clear_temporal": 1-5,
    "both_valid": 1-5,
    "single_dimension": 1-5,
    "labels_correct": 1-5
  },
  "average": <mean of scores>,
  "issues": ["list of specific problems"],
  "suggestion": "how to improve"
}
"""

validation_response_format = {
"type": "json_schema",
"json_schema": {
    "name": "dataset_validation_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
        "scores": {
            "type": "object",
            "description": "Scores ranging from 1 to 5 per each category",
            "properties": {
                "temporal_keywords": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "No explicit temporal words (now, immediate, future, lasting, years, soon, quick, urgent, permanent, decade)"
                },
                "vocabulary_balance": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "Both options use similarly common/complex words"
                },
                "ngram_leakage": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "No idiomatic phrases that give away temporal scope"
                },
                "length_match": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "Options within 50% character length"
                },
                "structure": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "Grammatically parallel"
                },
                "formality": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "Same register"
                },
                "hedging": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "Same certainty level"
                },
                "specificity": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "Same abstraction level"
                },
                "sentiment": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "Both neutral or matched valence"
                },
                "clear_temporal": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "Unambiguous immediate vs long-term distinction"
                },
                "both_valid": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "Both options are sensible responses"
                },
                "single_dimension": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "Options differ ONLY on temporal scope"
                },
                "labels_correct": {
                    "type": "integer",
                    "minimum" : 1,
                    "maximum" : 5,
                    "description": "The immediate/long-term labels are accurate"
                }
            },
            "required": ["temporal_keywords", "vocabulary_balance", "ngram_leakage", "length_match", "structure", "formality", "hedging", "specificity", "sentiment", "clear_temporal", "both_valid", "single_dimension", "labels_correct"]
        },
        "average": {
            "type": "number",
            "description": "Average score over all categories"
        },
        "issues": {
            "type": "array",
            "items": { "type": "string" },
            "description": "list of specific problems"
        },
        "suggestion": {
            "type": "string",
            "description": "how to improve"
        },
        },
        "required": ["scores", "average", "issues", "suggestion"],
        "additionalProperties": False
    }
}
}
