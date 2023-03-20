import re
import six
from six.moves import range  # pylint: disable=redefined-builtin

PAD = "<pad>"
EOS = "<EOS>"
UNK = "<UNK>"
SEG = "|"
RESERVED_TOKENS = [PAD, EOS, UNK]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1
UNK_ID = RESERVED_TOKENS.index(UNK)  # Normally 2

if six.PY2:
    RESERVED_TOKENS_BYTES = RESERVED_TOKENS
else:
    RESERVED_TOKENS_BYTES = [bytes(PAD, "ascii"), bytes(EOS, "ascii")]

# Regular expression for unescaping token strings.
# '\u' is converted to '_'
# '\\' is converted to '\'
# '\213;' is converted to unichr(213)
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")
_ESCAPE_CHARS = set(u"\\_u;0123456789")


def strip_ids(ids, ids_to_strip):
    """Strip ids_to_strip from the end ids."""
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


class TextEncoder(object):
    """Base class for converting from ints to/from human readable strings."""

    def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
        self._num_reserved_ids = num_reserved_ids

    @property
    def num_reserved_ids(self):
        return self._num_reserved_ids

    def encode(self, s):
        """Transform a human-readable string into a sequence of int ids.

        The ids should be in the range [num_reserved_ids, vocab_size). Ids [0,
        num_reserved_ids) are reserved.

        EOS is not appended.

        Args:
        s: human-readable string to be converted.

        Returns:
        ids: list of integers
        """
        return [int(w) + self._num_reserved_ids for w in s.split()]

    def decode(self, ids, strip_extraneous=False):
        """Transform a sequence of int ids into a human-readable string.

        EOS is not expected in ids.

        Args:
        ids: list of integers to be converted.
        strip_extraneous: bool, whether to strip off extraneous tokens
            (EOS and PAD).

        Returns:
        s: human-readable string.
        """
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids):
        """Transform a sequence of int ids into a their string versions.

        This method supports transforming individual input/output ids to their
        string versions so that sequence to/from text conversions can be visualized
        in a human readable format.

        Args:
        ids: list of integers to be converted.

        Returns:
        strs: list of human-readable string.
        """
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self._num_reserved_ids)
        return [str(d) for d in decoded_ids]

    @property
    def vocab_size(self):
        raise NotImplementedError()


class ByteTextEncoder(TextEncoder):
    """Encodes each byte to an id. For 8-bit strings only."""

    def encode(self, s):
        numres = self._num_reserved_ids
        if six.PY2:
            if isinstance(s, unicode):
                s = s.encode("utf-8")
            return [ord(c) + numres for c in s]
        # Python3: explicitly convert to UTF-8
        return [c + numres for c in s.encode("utf-8")]

    def decode(self, ids, strip_extraneous=False):
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        numres = self._num_reserved_ids
        decoded_ids = []
        int2byte = six.int2byte
        for id_ in ids:
            if 0 <= id_ < numres:
                decoded_ids.append(RESERVED_TOKENS_BYTES[int(id_)])
            else:
                decoded_ids.append(int2byte(id_ - numres))
        if six.PY2:
            return "".join(decoded_ids)
        # Python3: join byte arrays and then decode string
        return b"".join(decoded_ids).decode("utf-8", "replace")

    def decode_list(self, ids):
        numres = self._num_reserved_ids
        decoded_ids = []
        int2byte = six.int2byte
        for id_ in ids:
            if 0 <= id_ < numres:
                decoded_ids.append(RESERVED_TOKENS_BYTES[int(id_)])
            else:
                decoded_ids.append(int2byte(id_ - numres))
        # Python3: join byte arrays and then decode string
        return decoded_ids

    @property
    def vocab_size(self):
        return 2**8 + self._num_reserved_ids


class ByteTextEncoderWithEos(ByteTextEncoder):
  """Encodes each byte to an id and appends the EOS token."""

  def encode(self, s):
        return super(ByteTextEncoderWithEos, self).encode(s) + [EOS_ID]


class TokenTextEncoder(TextEncoder):
    """Encoder based on a user-supplied vocabulary (file or list)."""

    def __init__(self,
               vocab_filename,
               reverse=False,
               vocab_list=None,
               replace_oov=None,
               num_reserved_ids=NUM_RESERVED_TOKENS):
        """Initialize from a file or list, one token per line.

        Handling of reserved tokens works as follows:
        - When initializing from a list, we add reserved tokens to the vocab.
        - When initializing from a file, we do not add reserved tokens to the vocab.
        - When saving vocab files, we save reserved tokens to the file.

        Args:
            vocab_filename: If not None, the full filename to read vocab from. If this
                is not None, then vocab_list should be None.
            reverse: Boolean indicating if tokens should be reversed during encoding
                and decoding.
            vocab_list: If not None, a list of elements of the vocabulary. If this is
                not None, then vocab_filename should be None.
            replace_oov: If not None, every out-of-vocabulary token seen when
                encoding will be replaced by this string (which must be in vocab).
            num_reserved_ids: Number of IDs to save for reserved tokens like <EOS>.
        """
        super(TokenTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        if vocab_filename:
            self._init_vocab_from_file(vocab_filename)
        else:
            assert vocab_list is not None
            self._init_vocab_from_list(vocab_list)
        self.pad_index = self._token_to_id[PAD]
        self.eos_index = self._token_to_id[EOS]
        self.unk_index = self._token_to_id[UNK]
        self.seg_index = self._token_to_id[SEG] if SEG in self._token_to_id else self.eos_index

    def encode(self, s):
        """Converts a space-separated string of tokens to a list of ids."""
        sentence = s
        tokens = sentence.strip().split()
        if self._replace_oov is not None:
            tokens = [t if t in self._token_to_id else self._replace_oov
                        for t in tokens]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret

    def decode(self, ids, strip_eos=False, strip_padding=False):
        if strip_padding and self.pad() in list(ids):
            pad_pos = list(ids).index(self.pad())
            ids = ids[:pad_pos]
        if strip_eos and self.eos() in list(ids):
            eos_pos = list(ids).index(self.eos())
            ids = ids[:eos_pos]
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in seq]

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def __len__(self):
        return self.vocab_size

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" % idx)

    def _init_vocab_from_file(self, filename):
        """Load vocab from a file.

        Args:
        filename: The file to load vocabulary from.
        """
        with open(filename) as f:
            tokens = [token.strip() for token in f.readlines()]

        def token_gen():
            for token in tokens:
                yield token

        self._init_vocab(token_gen(), add_reserved_tokens=False)

    def _init_vocab_from_list(self, vocab_list):
        """Initialize tokens from a list of tokens.

        It is ok if reserved tokens appear in the vocab list. They will be
        removed. The set of tokens in vocab_list should be unique.

        Args:
        vocab_list: A list of tokens.
        """
        def token_gen():
            for token in vocab_list:
                if token not in RESERVED_TOKENS:
                    yield token

        self._init_vocab(token_gen())

    def _init_vocab(self, token_generator, add_reserved_tokens=True):
        """Initialize vocabulary with tokens from token_generator."""

        self._id_to_token = {}
        non_reserved_start_index = 0

        if add_reserved_tokens:
            self._id_to_token.update(enumerate(RESERVED_TOKENS))
            non_reserved_start_index = len(RESERVED_TOKENS)

        self._id_to_token.update(
            enumerate(token_generator, start=non_reserved_start_index))

        # _token_to_id is the reverse of _id_to_token
        self._token_to_id = dict((v, k)
                                for k, v in six.iteritems(self._id_to_token))

    def pad(self):
        return self.pad_index

    def eos(self):
        return self.eos_index

    def unk(self):
        return self.unk_index

    def seg(self):
        return self.seg_index

    def store_to_file(self, filename):
        """Write vocab file to disk.

        Vocab files have one token per line. The file ends in a newline. Reserved
        tokens are written to the vocab file as well.

        Args:
        filename: Full path of the file to store the vocab to.
        """
        with open(filename, "w") as f:
            for i in range(len(self._id_to_token)):
                f.write(self._id_to_token[i] + "\n")

    def sil_phonemes(self):
        return [p for p in self._id_to_token.values() if not p[0].isalpha()]
