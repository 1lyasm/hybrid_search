import argparse
import whoosh
from whoosh.fields import Schema, TEXT
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
import pandas
import os


class LexicalSearcher:
    def __init__(self):
        documents = (
            pandas.read_csv("data/imdb_dataset.csv")
            .drop("label", axis=1)
            .to_numpy()
            .reshape([-1])
        )

        schema = Schema(
            review=TEXT(stored=True),
        )
        directory_name = "keyword_search_index"
        if os.path.exists(directory_name):
            index = open_dir(directory_name)
        else:
            os.mkdir(directory_name)
            index = create_in(directory_name, schema)
            writer = index.writer()
            for document in documents:
                writer.add_document(review=document)
            writer.commit()

        self.parser = QueryParser("review", schema)
        self.searcher = index.searcher()

    def search(self, query_string):
        query = self.parser.parse(query_string)
        result_list = []
        results = self.searcher.search(query)
        for result in results:
            result_list.append(result["review"])
        return result_list

    def close(self):
        self.searcher.close()


def main():
    parser = argparse.ArgumentParser(
        prog="keyword_searcher",
        description="Performs keyword-based search with the given query",
    )
    parser.add_argument(
        "-q", "--query_string", help="Query string", required=True
    )
    arguments = parser.parse_args()

    searcher = LexicalSearcher()
    search_result = searcher.search(arguments.query_string)
    searcher.close()


if __name__ == "__main__":
    main()
