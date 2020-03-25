import numpy as np
from light_hanlp.utils.char_table import CharTable


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
        chars = CharTable.normalize_chars(sent)
        for word in chars:
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


def get_dep_inputs(sents, token_to_idx, pos_to_idx, word_embed_range, root_tok='<bos>',
                   root_pos='<bos>', unk_tok='<unk>', pad_tok='<pad>', unk_pos='<bos>', pad_pos='<bos>'):
    if type(sents) == str:
        sents = [sents]
    ext_input_ids = []
    input_ids = []
    pos_ids = []
    mask = []
    x_lengths = []
    for sent in sents:
        # 句法分析开头先加个root
        ext_input_ids.append([token_to_idx[root_tok]])
        input_ids.append([token_to_idx[root_tok]])
        pos_ids.append([pos_to_idx[root_pos]])
        mask.append([1])
        for word, pos in sent:
            if word in token_to_idx:
                if token_to_idx[word] < word_embed_range:
                    input_ids[-1].append(token_to_idx[word])
                else:
                    input_ids[-1].append(token_to_idx[unk_tok])
                ext_input_ids[-1].append(token_to_idx[word])
            else:
                input_ids[-1].append(token_to_idx[unk_tok])
                ext_input_ids[-1].append(token_to_idx[unk_tok])
            if pos in pos_to_idx:
                pos_ids[-1].append(pos_to_idx[pos])
            else:
                pos_ids[-1].append(pos_to_idx[unk_pos])

            mask[-1].append(1)

        x_lengths.append(len(ext_input_ids[-1]))

    if len(input_ids) > 1:
        input_ids = input_padding(input_ids, token_to_idx[pad_tok])
        ext_input_ids = input_padding(ext_input_ids, token_to_idx[pad_tok])
        pos_ids = input_padding(pos_ids, token_to_idx[pad_pos])
        mask = input_padding(mask, 0)

    return np.array(input_ids), np.array(ext_input_ids), np.array(pos_ids), np.array(mask), np.array(x_lengths)


def get_dep_outputs(arc_scores, rel_scores, lengths, rel_vocab):
    sents = []
    arc_preds = np.argmax(arc_scores, -1)
    rel_preds = np.argmax(rel_scores, -1)

    for arc_sent, rel_sent, length in zip(arc_preds, rel_preds, lengths):
        arcs = list(arc_sent)[1:length + 1]
        rels = list(rel_sent)[1:length + 1]
        sents.append([(a, rel_vocab[r[a]]) for a, r in zip(arcs, rels)])

    return sents

def get_sdp_outputs(arc_scores, rel_scores, lengths, rel_vocab, orphan_relation='<root>'):
    sents = []
    # SDP和DEP最大的差别在于,DEP的关系唯一，SDP可能有多个，可能0个
    arc_preds = arc_scores > 0
    rel_preds = np.argmax(rel_scores, -1)

    for arc_sent, rel_sent, length in zip(arc_preds, rel_preds, lengths):
        sent = []
        for arc, rel in zip(list(arc_sent[1:, 1:]), list(rel_sent[1:, 1:])):
            ar = []
            for idx, (a, r) in enumerate(zip(arc, rel)):
                if a:
                    ar.append((idx + 1, rel_vocab[r]))
            if not ar:
                # orphan
                ar.append((0, orphan_relation))
            sent.append(ar)
        sents.append(sent)
    return sents