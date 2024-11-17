from typing import Dict, List


def get_word_frequencies(doc: str) -> Dict[str, int]:
    """Extracts word frequencies from a document.

    Args:
        doc: Document content given as a string.

    Returns:
        Dictionary with words as keys and their frequencies as values.
    """

    try:
        replacements = ["\n", "\t", ",", ".", ":", ";", "?", "!", " "]
        data = {}

        for replacement in replacements:
            doc = doc.replace(replacement, " ")
        
        tokens = doc.split(" ")
        for token in tokens:
            if token == "":
                continue

            if token in data:
                data[token] += 1
            else:
                data[token] = 1

        print("data: ", data)
        return data

    except:
        return {}


def get_word_feature_vector(
    word_frequencies: Dict[str, int], vocabulary: List[str]
) -> List[int]:
    """Creates a feature vector for a document, comprising word frequencies
        over a vocabulary.

    Args:
        word_frequencies: Dictionary with words as keys and frequencies as
            values.
        vocabulary: List of words.

    Returns:
        List of length `len(vocabulary)` with respective frequencies as values.
    """
    try:
        datalist = []
        for v in vocabulary:
            if v in word_frequencies:
                datalist.append(word_frequencies[v])
            else:
                datalist.append(0)

        return datalist
    except:
        return []

