"""Names shared by experiment output, plotting, and support summaries."""

import re

PSEUDO_METHODS = {"pseudo_lasso", "pseudo_lasso_centered"}


def pseudo_method_info(method):
    if method in PSEUDO_METHODS:
        return method, None
    match = re.fullmatch(r"(pseudo_lasso(?:_centered)?)_ranked_k(\d+)", method)
    return (match[1], int(match[2])) if match else None


def expand_ranked_methods(methods, counts):
    if counts is None:
        return list(methods)
    if not counts or any(k < 1 for k in counts) or len(set(counts)) != len(counts):
        raise ValueError("Ranked feature counts must be distinct positive integers")
    if not PSEUDO_METHODS.intersection(methods):
        raise ValueError("Ranked feature counts require at least one pseudo-lasso method")
    return [variant for method in methods
            for variant in ([f"{method}_ranked_k{k}" for k in sorted(counts)]
                            if method in PSEUDO_METHODS else [method])]
