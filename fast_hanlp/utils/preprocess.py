import numpy as np


def input_padding(input_ids, pad_id):
    max_len = max([len(li) for li in input_ids])
    for input_ids_ in input_ids:
        if len(input_ids_) < max_len:
            input_ids_.extend([pad_id] * (max_len - len(input_ids_)))
    return input_ids


def get_cws_inputs(sents, token_to_idx, unk_tok='<unk>', pad_tok='<pad>'):
    if type(sents) == str:
        sents = [sents]
    input_ids = []
    for sent in sents:
        input_ids.append([])
        for word in sent:
            if word in token_to_idx:
                input_ids[-1].append(token_to_idx[word])
            else:
                input_ids[-1].append(token_to_idx[unk_tok])
    if len(input_ids) > 1:
        input_ids = input_padding(input_ids, token_to_idx[pad_tok])
    return np.array(input_ids)


def bmes_to_words(chars, tags):
    result = []
    if len(chars) == 0:
        return result
    word = chars[0]

    for c, t in zip(chars[1:], tags[1:]):
        if t == 'B' or t == 'S':
            result.append(word)
            word = ''
        word += c
    if len(word) != 0:
        result.append(word)

    return result


def get_cws_outputs(tag_vocab, Y, inputs):
    Y = np.argmax(Y, axis=2)
    results = []
    for y, inp in zip(Y, inputs):
        tags = [tag_vocab[int(y_)] for y_ in y[:len(inp)]]
        results.append(bmes_to_words(list(inp), tags))
    return results


def get_pos_outputs(tag_vocab, Y, inputs):
    Y = np.argmax(Y, axis=2)
    results = []
    for y, inp in zip(Y, inputs):
        tags = [tag_vocab[int(y_)] for y_ in y[:len(inp)]]
        results.append(tags)
    return results


def iobes_to_span(words, tags):
    delimiter = ' '
    if all([len(w) == 1 for w in words]):
        delimiter = ''  # might be Chinese
    entities = []
    for tag, start, end in get_entities(tags):
        entities.append((delimiter.join(words[start:end]), tag, start, end))
    return entities


def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 2), ('LOC', 3, 4)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        cells = chunk.split('-')
        if suffix:
            tag = chunk[-1]
            type_ = cells[0]
        else:
            tag = chunk[0]
            type_ = cells[-1]
        if len(cells) == 1:
            type_ = ''

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start
