import pandas as pd
from pathlib import Path
import os
import argparse
import json
import re

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_results", type=str, help="Path to .json file with results")
    return parser

def style_html_table(tdf_styler, title, color="darkblue"):
    def __get_html_style_for_tables(color="darkblue"):
        # HTML styles
        table_width_via_font = {
            'selector': '',
            'props': ' font-size: 0.99vw !important;'
        }
        cell_hover = {  # for row hover use <tr> instead of <td>
            'selector': 'td:hover',
            'props': [('background-color', '#87cefa')]
        }
        index_names = {
            'selector': '.index_name',
            'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
        }
        headers = {
            'selector': 'th:not(.index_name)',
            'props': f'background-color: {color}; color: white;'
        }
        caption = {
            'selector': 'caption',
            'props': 'text-align: center; font-size: 120%; margin: 10px;'
        }
        output = {
            'selector': '',
            'props': 'margin-left: auto !important; margin-right: auto !important;'
        }

        return [table_width_via_font, cell_hover, index_names, headers, caption, output]

    def __set_main_header(html, header):
        # HTML styles
        # FIXME: Should I change <h2> and </style>?
        return re.sub('</style>', f'</style><h2 style="text-align: center;">{header}</h2>', html, 1)

    tdf_styler.set_table_styles(__get_html_style_for_tables(color))
    headed_styler_html = __set_main_header(tdf_styler._repr_html_(), title)
    return headed_styler_html

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    table_case_headers = ["Test case"]
    table_score_headers = ["temporal_keywords", "vocabulary_balance", "ngram_leakage",
                            "length_match", "structure", "formality", "hedging", "specificity",
                            "sentiment", "clear_temporal", "both_valid", "single_dimension",
                            "labels_correct"]
    table_all_results_headers = ["average"] + table_score_headers + ["issues", "suggestion"]
    table_content = []

    raw_results_path = args.raw_results
    assert(os.path.isfile(raw_results_path))
    with open(raw_results_path, 'r') as f:
        raw_dict = json.load(f)
        validated_cases = raw_dict["results"]
        for validated in validated_cases:
            table_content.append([validated["pair"], validated["average"],
                *[validated["scores"][header] for header in table_score_headers],
                 validated["issues"], validated["suggestion"]])

        df = pd.DataFrame(table_content, columns=table_case_headers + table_all_results_headers)
        df.loc[-1] = df.mean(numeric_only=True)
        df.loc[-1, "Test case"] = "Average over all dataset"
        df.loc[-1, "issues"] = "N/A"
        df.loc[-1, "suggestion"] = "N/A"
        df.index = df.index + 1
        df.sort_index(inplace=True)

        pivot_content = raw_dict["summary"].copy()
        pivot_df = pd.DataFrame([v for v in pivot_content.values()], index=pivot_content.keys(), columns=["Summary"])
        pivot_df.loc["Assessed by", :] = \
        f"""LLM Council: {pivot_content["Assessed by"]["LLM council"]},
            Chairman: {pivot_content["Assessed by"]["Chairman"]}"""

        dataset_and_timestamp = re.search(r"validation_([a-zA-Z_]+_20[\d-]+).json", raw_results_path).group(1)
        with open((Path(raw_results_path).parent / f"report_{dataset_and_timestamp}.html"), "w") as f:
            def gradient_paint_score_rows(styler):
                styler.background_gradient(axis=1, vmin=1, vmax=5, cmap="RdYlGn", low=0.4, high=0.8,
                                           subset=["average"] + table_score_headers)
                return styler
            df_styler = gradient_paint_score_rows(df.style)
            html_table = style_html_table(df_styler, f"Validation of dataset with LLM council")
            html_pivot_table = style_html_table(pivot_df.style, f"Pivot table", "darkgreen")
            f.write(html_table + html_pivot_table)
