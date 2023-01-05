"""Automatic generation of test sets for OddOneOut
and Topk methods using Wikidata"""
from typing import Tuple, List, Dict
from SPARQLWrapper import SPARQLWrapper, JSON
import re


def sparql_query(endpoint_url: str, query: str, variable_name: str, user_agent: str = None) -> List[str]:
    """
    Make a request to a SPARQL endpoint and convert the result into categories.
    The response should contain only one variable.

    Parameters
    ----------
    endpoint_url : str
        The url for the appropriate SPARQL endpoint.
    query : str
        SPARQL query
    variable_name : str
        Variable containing the category string.
    user_agent: str
        User agent for the SPARQL query.

    Returns
    -------
    List[str]
        Elements of the category.

    Raises
    ------
    AssertionError
        A ValueError exception is raised if more than one variable is found
        in the response of the SPARQL endpoint.
    """
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    res = sparql.query().convert()

    assert len(res["head"]["vars"]) == 1, "Only one variable must be selected using the SPARQL query."

    return [result[variable_name]["value"] for result in res["results"]["bindings"]]


# create a separate test set for each language from queries.
# test set formatted to be compatible with OddOneOut and Topk
def generate_test_set(queries: Dict[str, str], 
                      endpoint_url: str, 
                      variable_name: str, 
                      user_agent: str = None) -> Dict[str, str]:
    """
    Generate the set of categories for each query.

    Parameters
    ----------
    queries : Dict[str, str]
        Set of query from which each category is extracted.
    endpoint_url : str
        Endpoint on which the queries are run.
    variable_name : str
        Variable from which the category is extracted.
    user_agent : str, optional
        User agent for the SPARQL query, by default None.

    Returns
    -------
    Dict[str, str]
        Mapping of words to the corresponding categories.
    """
    return {
        cat: sparql_query(endpoint_url, query, variable_name, user_agent)
        for cat, query in queries.items()
    }
