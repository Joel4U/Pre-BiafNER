from typing import List, Dict, Tuple, TypeVar, Optional, Callable
from src.data import Instance
import numpy as np


B_PREF="B-"
I_PREF = "I-"
S_PREF = "S-"
E_PREF = "E-"
O = "O"

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"
root_dep_label = "root"
self_label = "self"

FLAT = 0       # Flat entities
NESTED = 1     # Nested entities
ARBITRARY = 2  # Arbitrarily overlapping entities

import logging

logger = logging.getLogger(__name__)

def convert_iobes(labels: List[str]) -> List[str]:
	"""
	Use IOBES tagging schema to replace the IOB tagging schema in the instance
	:param insts:
	:return:
	"""
	for pos in range(len(labels)):
		curr_entity = labels[pos]
		if pos == len(labels) - 1:
			if curr_entity.startswith(B_PREF):
				labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				labels[pos] = curr_entity.replace(I_PREF, E_PREF)
		else:
			next_entity = labels[pos + 1]
			if curr_entity.startswith(B_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(I_PREF, E_PREF)
	return labels


def build_label_idx(insts: List[Instance]) -> Tuple[List[str], Dict[str, int]]:
	"""
	Build the mapping from label to index and index to labels.
	:param insts: list of instances.
	:return:
	"""
	label2idx = {}
	idx2labels = []
	label2idx[PAD] = len(label2idx)
	idx2labels.append(PAD)
	for inst in insts:
		for label in inst.labels:
			if label not in label2idx:
				idx2labels.append(label)
				label2idx[label] = len(label2idx)

	label2idx[START_TAG] = len(label2idx)
	idx2labels.append(START_TAG)
	label2idx[STOP_TAG] = len(label2idx)
	idx2labels.append(STOP_TAG)
	label_size = len(label2idx)
	logger.info("#labels: {}".format(label_size))
	logger.info("label 2idx: {}".format(label2idx))
	return idx2labels, label2idx

def build_spanlabel_idx(insts: List[Instance]) -> Tuple[List[str], Dict[str, int]]:
	label2idx = {}
	idx2label = []
	# label2idx[PAD] = len(label2idx)
	# idx2labels.append(PAD)
	label2idx['O'] = len(label2idx)
	idx2label.append('O')
	for inst in insts:
		for spanlabel in inst.span_labels:
			entity_type = spanlabel[0]
			if entity_type not in label2idx:
				idx2label.append(entity_type)
				label2idx[entity_type] = len(label2idx)
			else:
				continue

	label_size = len(label2idx)
	print("#span labels: {}".format(label_size))
	print("spanlabel 2idx: {}".format(label2idx))
	return idx2label, label2idx

def check_all_labels_in_dict(insts: List[Instance], label2idx: Dict[str, int]):
	for inst in insts:
		for label in inst.labels:
			if label not in label2idx:
				raise ValueError(f"The label {label} does not exist in label2idx dict. The label might not appear in the training set.")


def build_word_idx(trains:List[Instance], devs:List[Instance], tests:List[Instance]) -> Tuple[Dict, List, Dict, List]:
	"""
	Build the vocab 2 idx for all instances
	:param train_insts:
	:param dev_insts:
	:param test_insts:
	:return:
	"""
	word2idx = dict()
	idx2word = []
	word2idx[PAD] = 0
	idx2word.append(PAD)
	word2idx[UNK] = 1
	idx2word.append(UNK)

	char2idx = {}
	idx2char = []
	char2idx[PAD] = 0
	idx2char.append(PAD)
	char2idx[UNK] = 1
	idx2char.append(UNK)

	# extract char on train, dev, test
	for inst in trains + devs + tests:
		for word in inst.words:
			if word not in word2idx:
				word2idx[word] = len(word2idx)
				idx2word.append(word)
	# extract char only on train (doesn't matter for dev and test)
	for inst in trains:
		for word in inst.words:
			for c in word:
				if c not in char2idx:
					char2idx[c] = len(idx2char)
					idx2char.append(c)
	return word2idx, idx2word, char2idx, idx2char


def check_all_obj_is_None(objs):
	for obj in objs:
		if obj is not None:
			return False
	return [None] * len(objs)

def build_deplabel_idx(insts: List[Instance]) -> Tuple[Dict[str, int], int]:
	deplabel2idx = {}
	deplabels = []
	if self_label not in deplabel2idx:
		deplabels.append(self_label)
		deplabel2idx[self_label] = len(deplabel2idx)
	for inst in insts:
		for label in inst.deplabels:
			if label not in deplabels:
				deplabels.append(label)
				deplabel2idx[label] = len(deplabel2idx)
	root_dep_label_id = deplabel2idx[root_dep_label]
	print("dep labels: {}".format(len(deplabels)))
	print("dep label 2idx: {}".format(deplabel2idx))
	return deplabel2idx, root_dep_label_id

def head_to_adj(max_len, words, heads):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    directed = 0
    self_loop = False #config.adj_self_loop
    ret = np.zeros((max_len, max_len), dtype=np.float32)

    for i, head in enumerate(heads):
        if head == 0:
            continue
        ret[head, i] = 1

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in range(len(words)):
            ret[i, i] = 1

    return ret


def head_to_adj_label(max_len, words, heads, dep_labels, root_dep_label_id):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    directed = 0
    self_loop = False

    dep_label_ret = np.zeros((max_len, max_len), dtype=np.int64)

    for i, head in enumerate(heads):
        if head == 0:
            continue
        dep_label_ret[head, i] = dep_labels[i]

    if not directed:
        dep_label_ret = dep_label_ret + dep_label_ret.T

    if self_loop:
        for i in range(len(words)):
            dep_label_ret[i, i] = root_dep_label_id

    return dep_label_ret

TypedSpan = Tuple[int, Tuple[int, int]]
TypedStringSpan = Tuple[str, Tuple[int, int]]


def build_pos_idx(insts: List[Instance]) -> Tuple[List[str], Dict[str, int]]:
	"""
	Build the mapping from label to index and index to labels.
	:param insts: list of instances.
	:return:
	"""
	pos2idx = {}
	idx2pos = []
	pos2idx[root_dep_label] = len(pos2idx)
	idx2pos.append(root_dep_label)
	for inst in insts:
		for tag in inst.pos:
			if tag not in pos2idx:
				idx2pos.append(tag)
				pos2idx[tag] = len(pos2idx)

	tag_size = len(pos2idx)
	logger.info("#pos_labels: {}".format(tag_size))
	logger.info("pos_label 2idx: {}".format(pos2idx))
	return idx2pos, pos2idx

class Token:
    """
    A simple token representation, keeping track of the token's text, offset in the passage it was
    taken from, POS tag, dependency relation, and similar information.  These fields match spacy's
    exactly, so we can just use a spacy token for this.

    # Parameters

    text : `str`, optional
        The original text represented by this token.
    idx : `int`, optional
        The character offset of this token into the tokenized passage.
    idx_end : `int`, optional
        The character offset one past the last character in the tokenized passage.
    lemma_ : `str`, optional
        The lemma of this token.
    pos_ : `str`, optional
        The coarse-grained part of speech of this token.
    tag_ : `str`, optional
        The fine-grained part of speech of this token.
    dep_ : `str`, optional
        The dependency relation for this token.
    ent_type_ : `str`, optional
        The entity type (i.e., the NER tag) for this token.
    text_id : `int`, optional
        If your tokenizer returns integers instead of strings (e.g., because you're doing byte
        encoding, or some hash-based embedding), set this with the integer.  If this is set, we
        will bypass the vocabulary when indexing this token, regardless of whether `text` is also
        set.  You can `also` set `text` with the original text, if you want, so that you can
        still use a character-level representation in addition to a hash-based word embedding.
    type_id : `int`, optional
        Token type id used by some pretrained language models like original BERT

        The other fields on `Token` follow the fields on spacy's `Token` object; this is one we
        added, similar to spacy's `lex_id`.
    """

    __slots__ = [
        "text",
        "idx",
        "idx_end",
        "lemma_",
        "pos_",
        "tag_",
        "dep_",
        "ent_type_",
        "text_id",
        "type_id",
    ]
    # Defining the `__slots__` of this class is an optimization that dramatically reduces
    # the size in memory of a `Token` instance. The downside of using `__slots__`
    # with a dataclass is that you can't assign default values at the class level,
    # which is why we need a custom `__init__` function that provides the default values.

    text: Optional[str]
    idx: Optional[int]
    idx_end: Optional[int]
    lemma_: Optional[str]
    pos_: Optional[str]
    tag_: Optional[str]
    dep_: Optional[str]
    ent_type_: Optional[str]
    text_id: Optional[int]
    type_id: Optional[int]

    def __init__(
        self,
        text: str = None,
        idx: int = None,
        idx_end: int = None,
        lemma_: str = None,
        pos_: str = None,
        tag_: str = None,
        dep_: str = None,
        ent_type_: str = None,
        text_id: int = None,
        type_id: int = None,
    ) -> None:
        assert text is None or isinstance(
            text, str
        )  # Some very hard to debug errors happen when this is not true.
        self.text = text
        self.idx = idx
        self.idx_end = idx_end
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.tag_ = tag_
        self.dep_ = dep_
        self.ent_type_ = ent_type_
        self.text_id = text_id
        self.type_id = type_id

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()

    def ensure_text(self) -> str:
        """
        Return the `text` field, raising an exception if it's `None`.
        """
        if self.text is None:
            raise ValueError("Unexpected null text for token")
        else:
            return self.text

class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return " ".join(self.tag_sequence)


T = TypeVar("T", str, Token)


def enumerate_spans(
    sentence: List[T],
    offset: int = 0,
    max_span_width: int = None,
    min_span_width: int = 1,
    filter_function: Callable[[List[T]], bool] = None,
) -> List[Tuple[int, int]]:
    """
    Given a sentence, return all token spans within the sentence. Spans are `inclusive`.
    Additionally, you can provide a maximum and minimum span width, which will be used
    to exclude spans outside of this range.

    Finally, you can provide a function mapping `List[T] -> bool`, which will
    be applied to every span to decide whether that span should be included. This
    allows filtering by length, regex matches, pos tags or any Spacy `Token`
    attributes, for example.

    # Parameters

    sentence : `List[T]`, required.
        The sentence to generate spans for. The type is generic, as this function
        can be used with strings, or Spacy `Tokens` or other sequences.
    offset : `int`, optional (default = `0`)
        A numeric offset to add to all span start and end indices. This is helpful
        if the sentence is part of a larger structure, such as a document, which
        the indices need to respect.
    max_span_width : `int`, optional (default = `None`)
        The maximum length of spans which should be included. Defaults to len(sentence).
    min_span_width : `int`, optional (default = `1`)
        The minimum length of spans which should be included. Defaults to 1.
    filter_function : `Callable[[List[T]], bool]`, optional (default = `None`)
        A function mapping sequences of the passed type T to a boolean value.
        If `True`, the span is included in the returned spans from the
        sentence, otherwise it is excluded..
    """
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            # add 1 to end index because span indices are inclusive.
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end))
    return spans

def _spans_from_upper_triangular(seq_len: int):
    """Spans from the upper triangular area.
    """
    for start in range(seq_len):
        for end in range(start+1, seq_len+1):
            yield (start, end)

def _is_overlapping(chunk1: tuple, chunk2: tuple):
    # `NESTED` or `ARBITRARY`
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 < e2 and s2 < e1)


def _is_ordered_nested(chunk1: tuple, chunk2: tuple):
    # `chunk1` is nested in `chunk2`
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s2 <= s1 and e1 <= e2)

def _is_nested(chunk1: tuple, chunk2: tuple):
    # `NESTED`
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)


def _is_clashed(chunk1: tuple, chunk2: tuple, allow_level: int=NESTED):
    if allow_level == FLAT:
        return _is_overlapping(chunk1, chunk2)
    elif allow_level == NESTED:
        return _is_overlapping(chunk1, chunk2) and not _is_nested(chunk1, chunk2)
    else:
        return False

def filter_clashed_by_priority(chunks: List[tuple], allow_level: int=NESTED):
    filtered_chunks = []
    for ck in chunks:
        if all(not _is_clashed(ck, ex_ck, allow_level=allow_level) for ex_ck in filtered_chunks):
            filtered_chunks.append(ck)
    return filtered_chunks

# 检测给定的标记片段列表中是否存在嵌套或重叠的情况，并确定它们的层次。
def detect_overlapping_level(chunks: List[tuple]):
    level = FLAT
    for i, ck1 in enumerate(chunks):
        for ck2 in chunks[i+1:]:
            if _is_nested(ck1, ck2):
                level = NESTED
            elif _is_overlapping(ck1, ck2):
                # Non-nested overlapping -> `ARBITRARY`
                return ARBITRARY
    return level


def detect_nested(chunks1: List[tuple], chunks2: List[tuple] = None, strict: bool = True):
	"""Return chunks from `chunks1` that are nested in any chunk from `chunks2`.
    """
	if chunks2 is None:
		chunks2 = chunks1

	nested_chunks = []
	for ck1 in chunks1:
		if any(_is_ordered_nested(ck1, ck2) and (ck1 != ck2) and (not strict or ck1[1:] != ck2[1:]) for ck2 in chunks2):
			nested_chunks.append(ck1)
	return nested_chunks