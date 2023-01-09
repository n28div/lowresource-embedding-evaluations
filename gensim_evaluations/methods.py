"""
Implementation of the Topk and OddOneOut methods for evaluating word embeddings
"""
from typing import Dict, Tuple, List, Set
import re
from itertools import combinations
import random
import logging


def top_k(categories: Dict[str, str], 
          model, 
          k: int = 3, 
          allow_oov: bool = False,
          logger: logging.Logger = logging.getLogger(__name__)) -> Tuple[float, Dict[str, float], List[str], int, Dict[str, float]]:
    """
    Implements the topk evaluation on an embedding model.

    Parameters
    ----------
    categories : Dict[str, str]
        Dictionary containing the categories to be tested.
        The keys of the dictionary are the categories and the values are
        the element that are part of such category.
    model : gensim.KeyedVectors
        Embedding model used to compute the embeddings. Needs to a gensim
        KeyedVector model.
    k : int, optional
        Number of words to return from similarity query. Default to 3.
    allow_oov : bool, optional
        Allows comparisons with words not in the model. If set to True,
        comparisons with oov words will be marked as wrong.
        Defaults to False.
    logger: logging.Logger
        Logger used to comunicate to the user. Defaults to logging.getLogger(__name__)

    Returns
    -------
    Tuple[float, Dict[str, float], List[str], int, Dict[str, float]]
        A tuple containing respectively:
            - overall accuracy
            - category accuracy (accuracy for each category)
            - list of skipped categories
            - overall raw score (total number of correct comparisons)
            - category raw score (number of correct comparisons for each category)

    Raises
    ------
    KeyError
        If OOV is disabled and some words are not in the vocabulary then KeyError
        exception is raised.
    """
    logger.info('Performing Topk Evaluation')

    # for tracking scores for each category
    category_acc = {}
    category_raw = {}
    # total raw number of correct answers
    raw_correct = 0
    # total number of categories
    m = len(categories.items())

    # find the topk most similar for each word in all categories
    skipped_categories = []
    for cat, words in categories.items():
        if len(words) > 2:
            score = 0
            cat_oov = 0

            for word in words:
                # Evaluating Comparisons
                if model.has_index_for(word):
                    # find top_k similar words for a given entry
                    topk = model.most_similar(positive=word, topn=k)
                    # items in each category
                    for x in topk:
                        # x is a tuple and we want first element
                        score += x[0] in words
                else:
                    cat_oov += 1

            n = len(words) if allow_oov else len(words) - cat_oov

            category_acc.update({cat: score / (n * k) if n > 0 else 0})
            category_raw.update({cat: score})
            # update total raw number correct
            raw_correct += score
        else:
            skipped_categories.append(cat)

    if len(skipped_categories) > 0:
        logger.warn('%d categories do not have enough words and has been skipped' % len(skipped_categories.keys()))

    # Total Score
    accuracy = sum(category_acc.values()) / m

    return accuracy, category_acc, skipped_categories, raw_correct, category_raw


def odd_one_out(categories: Dict[str, str], 
                model, 
                k_in: int = 3, 
                sample_size: int = 1000,
                restrict_vocab: bool = False,
                vocab: List[str] = None,
                allow_oov: bool = False,
                random_seed: int = 42,
                logger: logging.Logger = logging.getLogger(__name__)) -> Tuple[float, Dict[str, float], List[str], int, Dict[str, float]]:
    """
    Performs OddOneOut Evaluation on an embedding model

    allow_oov : bool, optional
        Allows comparisons with words not in the model. If set to True,
        comparisons with oov words will be marked as wrong.
        Defaults to False.
    logger: logging.Logger
        Logger used to comunicate to the user. Defaults to logging.getLogger(__name__)

    Parameters
    ----------
    categories : Dict[str, str]
        Dictionary containing the categories to be tested.
        The keys of the dictionary are the categories and the values are
        the element that are part of such category.
    model : gensim.KeyedVectors
        Embedding model used to compute the embeddings. Needs to a gensim
        KeyedVector model.
    k_in : int
        Size of group of words from the same category
    sample_size : int
        Number of OddOneOut comparisons to evaluate for each category. 
        Defaults to 1000.
    restrict_vocab : int, optional
        The size of the model vocabulary to sample the out-word from.
        Defaults to None.
    vocab: list, optional
        Vocabulary of tokens in the corpora. If None the index_to_key method
        of KeyedVectors is used. Defaults to None.
    allow_oov : bool, optional
        Allows comparisons with words not in the model. If set to True,
        comparisons with oov words will be marked as wrong.
        Defaults to False.
    random_seed: int, optional
        Random seed used to sample from categories.
    logger: logging.Logger
        Logger used to comunicate to the user. Defaults to logging.getLogger(__name__)

    Returns
    -------
    Tuple[float, Dict[str, float], List[str], int, Dict[str, float]]
        A tuple containing respectively:
            - overall accuracy
            - category accuracy (accuracy for each category)
            - list of skipped categories
            - overall raw score (total number of correct comparisons)
            - category raw score (number of correct comparisons for each category)
    """
    original_state = random.getstate()
    random.seed(random_seed)
    
    # test set info
    logger.info('OddOneOut Evaluation')
    
        # for tracking scores for each category
    category_acc = {}
    category_raw = {}
    # total raw number of correct answers
    raw_correct = 0
    # total number of categories
    m = len(categories.items())
    logger.info('Will calculate the %d th order OddOneOut score for %d categories' % (k_in, m))

    not_skipped = 0
    skipped_categories = []
    for cat, words in categories.items():
        if len(words) > 2:
            # sample sample_size subsets from words
            s_sampled = random.choices(list(combinations(words, k_in)), k=sample_size)
            
            c_i = categories[cat]
            # sample OddOneOut from model vocabulary
            w_sampled = []
            vocabulary = vocab if vocab is not None else model.index_to_key[:(restrict_vocab or None)]
            while len(w_sampled) < sample_size:
                word = random.choice(vocabulary)
                if word not in c_i:
                    w_sampled.append(word)

            # compute OddOneOut for each comparison
            cat_score = 0
            cat_oov = 0
            for in_words, out_word in zip(s_sampled, w_sampled):
                if model.has_index_for(word):
                    cat_score += int(model.doesnt_match([*in_words, out_word]) == out_word)
                else:
                    cat_oov += 1

            n = sample_size if allow_oov else sample_size - cat_oov

            category_acc.update({cat: cat_score / n})
            category_raw.update({cat: cat_score})
            # update total raw number correct answers
            raw_correct += cat_score
        else:
            skipped_categories.append(cat)

    if len(skipped_categories) > 0:
        logger.warn('%d categories do not have enough words and has been skipped' % len(skipped_categories))

    # Calculate Total Score
    accuracy = sum(category_acc.values()) / m
    random.setstate(original_state)

    return accuracy, category_acc, skipped_categories, raw_correct, category_raw
