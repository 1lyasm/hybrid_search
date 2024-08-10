import argparse


class KeywordSearcher:
    def __init__(self):
        pass

    def search(self, query_string):
        pass


def main():
    parser = argparse.ArgumentParser(
        prog="keyword_searcher",
        description="Performs keyword-based search with the given query",
    )
    parser.add_argument(
        "-q", "--query_string", help="Query string", required=True
    )
    arguments = parser.parse_args()

    searcher = KeywordSearcher()
    searcher.search(arguments.query_string)


if __name__ == "__main__":
    main()
