"""
Implementation of the Topk and OddOneOut methods for evaluating word embeddings
"""
from tyiping import Dict, Tuple
import re
from itertools import combinations
import random
import logging
from gensim.models import KeyedVectors


def top_k(categories: Dict[str, str], 
          model: KeyedVectors, 
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
    model : KeyedVectors
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
    # skip those categories that has fewer than 2 words
    skipped_cats = [k for k, cats in categories.items() if len(cats) < 2]

    oov_list = set()
    words_in_test = 0
    # verify that all words are in the vocabulary
    for value in categories.values():
        for word in value:
            words_in_test += 1
            # keep track of oov words
            if allow_oov and not model.has_index_for(word):
                oov_list.add(word)
            elif not allow_oov and not model.has_index_for(word):
                raise KeyError('word '+word+' is not in vocabulary')

    # test set info
    logger.info('Performing Topk Evaluataion')
    
    if len(skipped_cats) > 0:
        logger.warn('%d categories do not have enough words and will be skipped' % len(skipped_cats))
    
    if len(oov_list) > 0:
        logger.warn('%d words have been identified as out out of vocabulary' % len(oov_list))
        ratio = len(oov_list)/words_in_test
        logger.warn('out of vocab ratio is ', '{0:.2f}'.format(ratio))

    logger.info('%d total words in test set' % words_in_test)
    
    # for tracking scores for each category
    category_acc = {}
    category_raw = {}
    # total raw number of correct answers
    raw_correct = 0
    # total number of categories
    m = len(categories.items())

    # remove dupes
    oov_list = set(oov_list)

    # find the topk most similar for each word in all categories
    for cat, words in cats.items():
        # zero out for each new category
        cat_score = 0

        for word in words:
            # number of the words
            n = len(words)
            category = words
            # Evaluating Comparisons
            if word not in oov_list:
                # find top_k similar words for a given entry
                topk = model.most_similar(positive=word, topn=k)
                # items in each category
                for x in topk:
                    # x is a tuple and we want first element
                    cat_score += x[0] in category

        # Update category score
        category_acc.update({cat: cat_score/(n*k)})
        category_raw.update({cat: cat_score})
        # update total raw number correct
        raw_correct += cat_score

    # Total Score
    accuracy = sum(category_acc.values()) / m

    return accuracy, category_acc, skipped_cats, raw_correct, category_raw


def odd_one_out(categories: Dict[str, str], 
                model: KeyedVectors, 
                k_in: int = 3, 
                sample_size: int = 100,
                restrict_vocab: bool = False,
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
    model : KeyedVectors
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

    Raises
    ------
    KeyError
        If OOV is disabled and some words are not in the vocabulary then KeyError
        exception is raised.
    """
    original_state = random.getstate()
    random.seed(random_seed)
    
    # skip those categories that has fewer than k_in words
    skipped_cats = [k for k, cats in categories.items() if len(cats) < k_in]

    # Verify all words are in the vocabulary
    oov_list = set()
    words_in_test = 0
    # verify that all words are in the vocabulary
    for value in categories.values():
        for word in value:
            words_in_test += 1
            # keep track of oov words
            if allow_oov and not model.has_index_for(word):
                oov_list.add(word)
            elif not allow_oov and not model.has_index_for(word):
                raise KeyError('word '+word+' is not in vocabulary')

    # test set info
    logger.info('OddOneOut Evaluation')
    
    if len(skipped_cats) > 0:
        logger.warn(' %d categories have fewer than k_in entries and will be skipped' % len(skipped_cats))
    
    if len(oov_list) > 0:
        logger.warn(' %d words have been identified as out out of vocabulary' % len(oov_list))
        ratio = len(oov_list)/words_in_test
        logger.warn('out of vocab ratio is ', '{0:.2f}'.format(ratio))

    # for storing accuracies
    category_acc = {}
    category_raw = {}

    # to return raw # correct preds instead of accuracy
    raw_correct = 0

    # total number of categories
    m = len(cats.items())
    logger.info('Will calculate the %d th order OddOneOut score for %d categories' % (k_in, m))
    not_skipped = 0
    for cat in cats.keys():
        s = list(combinations(cats[cat], k_in))
        c_i = cats[cat]
        # sample k-combos
        s_sampled = random.choices(s, k=sample_size)
        # sample OddOneOut from model vocabulary
        w_sampled = []
        while len(w_sampled) < sample_size:
            word = random.choice(model.index_to_key[:restrict_vocab])
            # don't add word if it's a dupe
            if word not in c_i:
                w_sampled.append(word)
        # kth order OddOneOut score for category i
        cat_score = 0
        # compute OddOneOut for each comparison
        for in_words, odd_one_out in zip(s_sampled, w_sampled):
            comparison = in_words + (odd_one_out)

            # By default don't ignore any comparisons
            ignore_comparison = False
            if allow_oov:
                # check for oov word
                for w in comparison:
                    if w in oov_list:
                        ignore_comparison = True
            if not ignore_comparison:
                cat_score += int(model.doesnt_match(comparison) == odd_one_out)

        category_acc.update({cat: cat_score/sample_size})
        category_raw.update({cat: cat_score})
        # update total raw number correct answers
        raw_correct += cat_score
    # Calculate Total Score
    accuracy = sum(category_acc.values())/m
    random.setstate(original_state)

    return accuracy, category_acc, skipped_cats, raw_correct, category_raw
