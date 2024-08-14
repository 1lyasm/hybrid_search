import json
import os
import argparse
import termcolor
from search import Searcher


def main():
    parser = argparse.ArgumentParser(
        prog="show_performances",
        description="Displays performance results of different search methods",
    )
    parser.add_argument(
        "--results_directory",
        help="Path of the directory that includes result files of search methods",
        default=os.path.join("output"),
    )
    arguments = parser.parse_args()

    performances = dict()
    for search_method in Searcher.methods:
        performances[search_method] = []
        with open(
            os.path.join(
                arguments.results_directory,
                f"search_{search_method}.json",
            ),
            "r",
        ) as result_file:
            results = json.load(result_file)

        total_score = 0
        evaluation_count = 0
        for result in results:
            for document in result["retrieved_documents"]:
                score = int(document["evaluation"]["score"])
                total_score += score
                evaluation_count += 1
        average_score = total_score / evaluation_count
        performances[search_method] = round(average_score, 2)

    termcolor.cprint(f"Average scores:", "magenta")
    for method in Searcher.methods:
        termcolor.cprint(
            f"{method} search: {performances[method]}", "blue"
        )


if __name__ == "__main__":
    main()
