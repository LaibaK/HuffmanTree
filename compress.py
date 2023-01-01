"""
Assignment 2 starter code
CSC148, Winter 2022
Instructors: Bogdan Simion, Sonya Allin, and Pooja Vashisth

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression

def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    dic = {}
    length = len(text)
    for i in range(length):
        if text[i] in dic:
            dic[text[i]] += 1
        else:
            dic[text[i]] = 1
    return dic


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    lis = []

    # Create list with tuples
    for i in freq_dict:
        freq = freq_dict[i]
        tree = HuffmanTree(i)
        item = (freq, tree)
        lis.append(item)

    # If only only item in lis, return tree for it
    if len(lis) == 1:
        lef = lis[0][1]
        rig = lis[0][1]
        return HuffmanTree(left=lef, right=rig)

    while len(lis) > 1:
        # Sort stack in order according to freq_dict
        # https://stackoverflow.com/questions/3766633/how-to-sort-with-lambda-in-python
        # used above link as a refresher to make sure I was sorting correctly
        lis = sorted(lis, key=lambda x: x[0])
        # Assign variables to first item in lis
        lt = lis.pop(0)
        rt = lis.pop(0)
        num = lt[0] + rt[0]
        huf = HuffmanTree(None, lt[1], rt[1])
        lis.append((num, huf))
    return lis[0][1]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    return helper_get_codes(tree, '', {})


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    helper_number_nodes(tree, 0)


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    # Get codes
    codes = get_codes(tree)

    # Get total number of bits (multiply freq with code length)
    bits = 0.0
    for k in freq_dict:
        total = freq_dict[k] * len(codes[k])
        bits += total

    # Total freq
    freq = 0
    for val in freq_dict.values():
        freq += val

    # Average length (divide total bits by the total freqency)
    final = bits / freq
    return final


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    lis = []
    st = ''

    for i in text:
        st += codes[i]

        if len(st) > 8:
            ran = st[:8]
            add = bits_to_byte(ran)
            lis += [add]
            # Reset st to new
            st = st[8:]

    # If st not empty
    if st != '':
        add = bits_to_byte(st)
        lis += [add]
    return bytes(lis)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    lis = []
    result = []
    po(tree, lis)

    for tre in lis:
        l_l = tre.left.left
        l_r = tre.left.right
        # left side
        if l_l is None and l_r is None:
            result.extend([0, tre.left.symbol])
        else:
            result.extend([1, tre.left.number])

        # right side
        r_l = tre.right.left
        r_r = tre.right.right
        if r_r is None and r_l is None:
            result.extend([0, tre.right.symbol])
        else:
            result.extend([1, tre.right.number])
    return bytes(result)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)

# Helper Functions


def helper_get_codes(tree: HuffmanTree, code: str, dic: dict) -> dict:
    """
    Return dictionary
    """
    if isinstance(tree.symbol, int):
        dic[tree.symbol] = code
        return dic

    if not isinstance(tree.symbol, int):
        first = code + '0'
        second = code + '1'
        helper_get_codes(tree.left, first, dic)
        helper_get_codes(tree.right, second, dic)

    return dic


def helper_number_nodes(tree: HuffmanTree, num: int) -> int:
    """
    Return None or integer representing internal nodes
    tree: Huffman Tree
    number: int
    """
    # Postorder Left, Right, Node
    if tree.symbol is not None:
        if tree.left is None:
            if tree.right is None:
                return num
    else:
        # go through left side of the tree checking for empty nodes
        if tree.left:
            num = helper_number_nodes(tree.left, num)

        # checks right
        if tree.right:
            num = helper_number_nodes(tree.right, num)

        # assigns the tree the largest number
        tree.number = num
        # add one for the root
        final = num + 1
        return final
    return None


def po(tree: HuffmanTree, lis: list) -> None:
    """ Add to a list the internal nodes in tree
    according to postorder traversal.
    tree : HuffmanNode
    lis: list
    """

    if tree.left is not None:
        po(tree.left, lis)

    if tree.right is not None:
        po(tree.right, lis)

    if not(tree.left is None or tree.right is None):
        lis.append(tree)


def reverse(d: dict) -> dict:
    '''
    Return inverse dict
    d: dict
    '''
    r_dict = {}
    for key, value in d.items():
        r_dict[value] = key
    return r_dict

# ====================
# Functions for decompression


def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    lt = node_lst[root_index].l_type
    rt = node_lst[root_index].r_type
    l_d = node_lst[root_index].l_data
    r_d = node_lst[root_index].r_data

    if lt == 0:
        # if leaf, store 'node' version
        lf = HuffmanTree(l_d)
    else:
        # if node, new index -> node from root
        lf = generate_tree_general(node_lst, l_d)

    # right side
    if rt == 0:
        rig = HuffmanTree(r_d)
    else:
        rig = generate_tree_general(node_lst, r_d)

    # make tree
    return HuffmanTree(None, lf, rig)


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    lt = node_lst[root_index].l_type
    rt = node_lst[root_index].r_type
    l_d = node_lst[root_index].l_data
    r_d = node_lst[root_index].r_data
    root = root_index - 1

    if lt == 0 and rt == 0:
        n = HuffmanTree(left=HuffmanTree(l_d), right=HuffmanTree(r_d))
    elif lt == 0 and rt == 1:
        n = HuffmanTree(left=HuffmanTree(l_d),
                        right=generate_tree_postorder(node_lst, root))
    elif lt == 1 and rt == 0:
        n = HuffmanTree(left=generate_tree_postorder(node_lst, root),
                        right=HuffmanTree(r_d))
    else:
        rig = generate_tree_postorder(node_lst, root)
        number_nodes(rig)
        new = root - (rig.number + 1)
        n = HuffmanTree(left=generate_tree_postorder(node_lst, new), right=rig)
    number_nodes(n)
    return n


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """

    codes = reverse(get_codes(tree))
    s = ''
    list_text = list(text)
    for i in list_text:
        s += byte_to_bits(i)

    uncomp = []
    tx = ''
    for bit in s:
        tx += bit
        if tx in codes:
            add = codes[tx]
            uncomp.append(add)
            tx = ''
            if len(uncomp) == size:
                break
    return bytes(uncomp)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions


def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    t = []
    # Create list of tuples in (byte, freq)
    for (byte, freq) in freq_dict.items():
        t.append((byte, freq))

    # Sort by second element
    t.sort(key=lambda x: x[1])

    tr = [tree]
    # While tree is not empty
    while len(tr) != 0:
        # assign node to first element in tran
        n = tr.pop(0)
        # If node does not have more values afterwards
        if n.is_leaf():
            second_value = t.pop()[0]
            n.symbol = second_value
        ran = [n.left, n.right]
        for i in ran:
            if i is not None:
                tr.append(i)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
