import ask_llm_council
import caa_constants
import json
from pathlib import Path
import argparse
import time

def get_parser():
    parser = argparse.ArgumentParser()
    root_dir = Path(__file__).parent / ".." / ".." / ".."
    implicit_dataset = str(root_dir / "data" / "raw" / "temporal_scope_implicit.json")
    parser.add_argument("--dataset", type=str, default=implicit_dataset)
    parser.add_argument("--agent", choices=['council', 'claude'], default='council')
    return parser

def validate_dataset(dataset_path, agent):
    pairs = None
    with open(dataset_path) as f:
        json_content = json.load(f)
        assert 'pairs' in json_content
        pairs = json_content['pairs']

    results = []
    ask_agent = None
    if agent == "council":
        ask_agent = ask_llm_council.ask_llm_council
        get_agent_info = ask_llm_council.get_council
    elif agent == "claude":
        raise Exception("Claude is not fully supported yet")
        # ask_agent = ask_claude_sonnet.ask_claude_sonnet
        # ask_agent = ask_claude_sonnet.get_info
    else:
        raise("Shouldn't reach that point")

    no_explicit_temporal_words_rule = \
        caa_constants.no_explicit_temporal_words_rule \
            if 'implicit' in dataset_path else \
        caa_constants.allowed_explicit_temporal_words_rule
    print(f"No temporal words rule for given dataset: {no_explicit_temporal_words_rule}")

    for i, pair in enumerate(pairs):
        response = ask_agent(
            caa_constants.validation_prompt.format(
                no_explicit_temporal_words_rule=no_explicit_temporal_words_rule,
                question=pair['question'],
                # Using [5:] here as both the predefined prompt and options
                # in datasets pairs contain item letters (A), (B):
                option_a=f"{pair['immediate'][5:]}",
                option_b=f"{pair['long_term'][5:]}",
                category=pair.get('category', 'unknown')) + \
            caa_constants.validation_prompt_return_hint,
            caa_constants.validation_response_format)
        result = json.loads(response)

        result['pair_index'] = i
        result["pair"] = pair    
        
        avg = result['average']
        status = "✓" if float(avg) >= 3.5 else ("⚠" if float(avg) >= 2.5 else "✗")
        print()
        print(f"{status} Pair {i}: {avg:.1f}/5 - \n{result["scores"]}")
        print(f"Issues: {result.get('issues', [])}")

        results.append(result)

    # Summary
    avgs = [r['average'] for r in results]
    mean = sum(avgs)/len(avgs)
    excellent = sum(1 for a in avgs if a >= 4.5)
    good = sum(1 for a in avgs if 3.5 <= a < 4.5)
    marginal = sum(1 for a in avgs if 2.5 <= a < 3.5)
    poor = sum(1 for a in avgs if a < 2.5)

    summary = {
        "Mean" : mean,
        "Excellent (4.5+)" : excellent,
        "Good (3.5-4.4)" : good,
        "Marginal (2.5-3.4)" : marginal,
        "Poor (<2.5)" : poor,
        "Assessed by" : get_agent_info()
    }

    print(f"\n=== SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    output = {"summary" : summary, "results" : results}
    return output

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    results = validate_dataset(args.dataset, args.agent)

    results_dir = Path(__file__).parent / "results"
    result_file_name = f"validation_{Path(args.dataset).stem}_{time.strftime("%Y%m%d-%H%M%S")}.json"
    with open(results_dir / result_file_name, "w") as f:
        json.dump(results, f, indent=2)
