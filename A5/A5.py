import abc
from collections import Counter
from collections import UserDict as DictClass
from collections import defaultdict
from typing import Dict, List
import math

CollectionType = Dict[str, Dict[str, List[str]]]


class DocumentCollection(DictClass):
    """Document dictionary class with helper functions."""

    def total_field_length(self, field: str) -> int:
        """Total number of terms in a field for all documents."""
        return sum(len(fields[field]) for fields in self.values())

    def avg_field_length(self, field: str) -> float:
        """Average number of terms in a field across all documents."""
        return self.total_field_length(field) / len(self)

    def get_field_documents(self, field: str) -> Dict[str, List[str]]:
        """Dictionary of documents for a single field."""
        return {
            doc_id: doc[field] for (doc_id, doc) in self.items() if field in doc
        }
    
    def get_number_of_documents_containing_term(self, field: str, term: str) -> int:
        counter = 0
        docs = self.get_field_documents(field)
        for doc_id in docs:
            if term in docs[doc_id]:
                counter += 1
        return counter
    
    def total_occurrences_of_term_in_field(self, field: str, term: str) -> int:
        counter = 0
        docs = self.get_field_documents(field)
        for doc_id in docs:
            if term in docs[doc_id]:
                counter += docs[doc_id].count(term)
        return counter
    



class Scorer(abc.ABC):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = None,
        fields: List[str] = None,
    ):
        """Interface for the scorer class.

        Args:
            collection: Collection of documents. Needed to calculate document
                statistical information.
            index: Index to use for calculating scores.
            field (optional): Single field to use in scoring.. Defaults to None.
            fields (optional): List of fields to use in scoring. Defaults to
                None.

        Raises:
            ValueError: Either field or fields need to be specified.
        """
        self.collection = collection
        self.index = index

        if not (field or fields):
            raise ValueError("Either field or fields have to be defined.")

        self.field = field
        self.fields = fields

        # Score accumulator for the query that is currently being scored.
        self.scores = None

    def score_collection(self, query_terms: List[str]):
        """Scores all documents in the collection using term-at-a-time query
        processing.

        Params:
            query_term: Sequence (list) of query terms.

        Returns:
            Dict with doc_ids as keys and retrieval scores as values.
            (It may be assumed that documents that are not present in this dict
            have a retrival score of 0.)
        """
        self.scores = defaultdict(float)  # Reset scores.
        query_term_freqs = Counter(query_terms)

        for term, query_freq in query_term_freqs.items():
            self.score_term(term, query_freq)

        return self.scores

    @abc.abstractmethod
    def score_term(self, term: str, query_freq: int):
        """Scores one query term and updates the accumulated document retrieval
        scores (`self.scores`).

        Params:
            term: Query term
            query_freq: Frequency (count) of the term in the query.
        """
        raise NotImplementedError


class SimpleScorer(Scorer):
    def score_term(self, term: str, query_freq: int) -> None:
        for doc_id in self.collection:
            try:
                score = self.collection[doc_id][self.field].count(term) * query_freq
                self.scores[doc_id] += score
            except:
                continue
        


class ScorerBM25(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        b: float = 0.75,
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25, self).__init__(collection, index, field)
        self.b = b
        self.k1 = k1
        self.avg_doc_length = self.collection.avg_field_length(field)  # Average document length for the specified field
        self.num_docs = len(self.collection.keys())


    def score_term(self, term: str, query_freq: int) -> None:
        for doc_id in self.collection:
            try:
                doc_length = len(self.collection[doc_id][self.field])
                term_freq = self.collection[doc_id][self.field].count(term)
                num_doc_contain_term = self.collection.get_number_of_documents_containing_term(self.field, term)
                
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                idf = math.log(self.num_docs / num_doc_contain_term)
                bm25_score = idf * (numerator / denominator) # Final score of the document using BM25
                
                self.scores[doc_id] += bm25_score * query_freq
            except:
                continue



class ScorerLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        smoothing_param: float = 0.1,
    ):
        super(ScorerLM, self).__init__(collection, index, field)
        self.smoothing_param = smoothing_param
        self.total_terms_in_field = self.collection.total_field_length(self.field)

    def score_term(self, term: str, query_freq: int) -> None:
        for doc_id in self.collection:
            try:
                doc_length = len(self.collection[doc_id][self.field])
                total_term_occurr = self.collection.total_occurrences_of_term_in_field(self.field, term)
                term_freq = self.collection[doc_id][self.field].count(term)
                collection_term_prob = total_term_occurr / self.total_terms_in_field

                doc_term_prob = ((1 - self.smoothing_param) * (term_freq / doc_length)) + (self.smoothing_param * collection_term_prob)
                self.scores[doc_id] += query_freq * math.log(doc_term_prob)
            except:
                continue



class ScorerBM25F(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: List[str] = ["title", "body"],
        field_weights: List[float] = [0.2, 0.8],
        bi: List[float] = [0.75, 0.75],
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25F, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.bi = bi
        self.k1 = k1
        self.num_docs = len(self.collection.keys())

    def score_term(self, term: str, query_freq: int) -> None:
        for doc_id in self.collection:
            try:
                num_doc_contain_term = self.collection.get_number_of_documents_containing_term("body", term)
                idf = math.log(self.num_docs / num_doc_contain_term)

                normalized_term_freq = 0 # C~t,d

                for i, field in enumerate(self.fields):
                    term_freq = self.collection[doc_id][field].count(term) # Ct,d
                    wi = self.field_weights[i]

                    doc_length = len(self.collection[doc_id][field])
                    avg_doc_length = self.collection.avg_field_length(field)
                    soft_normal = (1 - self.bi[i] + self.bi[i] * (doc_length / avg_doc_length)) #Bi

                    normalized_term_freq += wi * (term_freq / soft_normal)

                bm25_score = idf * (normalized_term_freq / (self.k1 + normalized_term_freq))
                self.scores[doc_id] += bm25_score * query_freq
            except:
                continue



class ScorerMLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: List[str] = ["title", "body"],
        field_weights: List[float] = [0.2, 0.8],
        smoothing_param: float = 0.1,
    ):
        super(ScorerMLM, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.smoothing_param = smoothing_param

    def score_term(self, term: str, query_freq: float) -> None:
        for doc_id in self.collection:
            try:
                doc_term_prob = 0 

                for i, field in enumerate(self.fields):
                        total_terms_in_field = self.collection.total_field_length(field)
                        doc_length = len(self.collection[doc_id][field])

                        total_term_occurr = self.collection.total_occurrences_of_term_in_field(field, term)
                        term_freq = self.collection[doc_id][field].count(term)
                        collection_term_prob = total_term_occurr / total_terms_in_field
                        doc_term_prob += self.field_weights[i] * (((1 - self.smoothing_param) * (term_freq / doc_length)) + (self.smoothing_param * collection_term_prob))

                self.scores[doc_id] += query_freq * math.log(doc_term_prob)
            except:
                continue

