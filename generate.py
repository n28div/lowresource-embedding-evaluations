"""
Generate the test set using one of the evaluation.categories modules.
"""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out", type=str, default="stdout")

subparsers = parser.add_subparsers(dest="source")

wikidata_parser = subparsers.add_parser("wikidata", help="Generate data using Wikidata.")
wikidata_parser.add_argument("-c", "--categories", type=str, nargs="+")
wikidata_parser.add_argument("-l", "--language", type=str)

wikidata_parser = subparsers.add_parser("sparql", help="Generate data using Wikidata.")
wikidata_parser.add_argument("-c", "--config", type=str)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.source == "wikidata":
        from evaluation.categories.wikidata import generate_test_set
        categories = generate_test_set(args.categories, args.language)
    elif args.source == "sparql":
        from evaluation.categories.sparql import generate_test_set

        with open(args.config, "r") as f:
            config = json.load(f)

        categories = generate_test_set(config["queries"], 
                                       config["endpoint"], 
                                       config["variable_name"], 
                                       user_agent=config["user_agent"])

    categories_json = json.dumps(categories)

    if args.out == "stdout":
        print(categories)
    else:
        with open(args.out, "w") as f:
            json.dump(categories, f)
        