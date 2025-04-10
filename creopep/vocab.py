import re
import pandas as pd

class PepVocab:
    def __init__(self):
        self.token_to_idx = { 
            '<MASK>': -1, '<PAD>': 0, 'A': 1, 'C': 2, 'E': 3, 'D': 4, 'F': 5, 'I': 6, 'H': 7,
            'K': 8, 'M': 9, 'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14,
            'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19, 'G': 20, 'O': 21, 'U': 22, 'Z': 23, 'X': 24}
        self.idx_to_token = { 
            -1: '<MASK>', 0: '<PAD>', 1: 'A', 2: 'C', 3: 'E', 4: 'D', 5: 'F', 6: 'I', 7: 'H',
            8: 'K', 9: 'M', 10: 'L', 11: 'N', 12: 'Q', 13: 'P', 14: 'S',
            15: 'R', 16: 'T', 17: 'W', 18: 'V', 19: 'Y', 20: 'G', 21: 'O', 22: 'U', 23: 'Z', 24: 'X'}
        
        self.get_attention_mask = False
        self.attention_mask = []
        
    def set_get_attn(self, is_get: bool):
        self.get_attention_mask = is_get

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        '''
        note: input should a splited sequence

        Args:
            tokens: a token or token list of splited
        '''
        if not isinstance(tokens, (list, tuple)):
            # return self.token_to_idx.get(tokens)
            return self.token_to_idx[tokens]
        return [self.__getitem__(token) for token in tokens]
    
    def vocab_from_txt(self, path):
        '''
        note: this function use for constructing vocab mapping
        but it is only suitable for special txt format
        it support one column txt file, which column name is 0
        '''
        token_to_idx = {}
        idx_to_token = {}
        chr_idx = pd.read_csv(path, header=None, sep='\t')
        if chr_idx.shape[1] == 1:
            for idx, token in enumerate(chr_idx[0]):
                token_to_idx[token] = idx
                idx_to_token[idx] = token
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        
    def to_tokens(self, indices):
        '''
        note: input should a integer list
        '''
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]
    
    def add_special_token(self, token: str|list|tuple) -> None:
        if not isinstance(token, (list, tuple)):
            if token in self.token_to_idx:
                raise ValueError(f"token {token} already in the vocab")
            self.idx_to_token[len(self.idx_to_token)] = token
            self.token_to_idx[token] = len(self.token_to_idx)
        else:
            [self.add_special_token(t) for t in token]
        
    def split_seq(self, seq: str|list|tuple) -> list:
        if not isinstance(seq, (list, tuple)):
            return re.findall(r"<[a-zA-Z0-9]+>|[a-zA-Z-]", seq)
        return [self.split_seq(s) for s in seq] # a list of list
    
    def truncate_pad(self, line, num_steps, padding_token='<PAD>') -> list:

        if not isinstance(line[0], list):
            if len(line) > num_steps:
                if self.get_attention_mask:
                    self.attention_mask.append([1]*num_steps)
                return line[:num_steps]
            if self.get_attention_mask:
                self.attention_mask.append([1] * len(line) + [0] * (num_steps - len(line)))
            return line + [padding_token] * (num_steps - len(line))
        else:
            return [self.truncate_pad(l, num_steps, padding_token) for l in line]   # a list of list
    
    def get_attention_mask_mat(self):
        attention_mask = self.attention_mask
        self.attention_mask = []
        return attention_mask

    def seq_to_idx(self, seq: str|list|tuple, num_steps: int, padding_token='<PAD>') -> list:
        '''
        note: ensure to execut this function after add_special_token
        '''

        splited_seq = self.split_seq(seq)
        # **********************
        # after split, we need to mask sequence
        # note: 
        # 1. mask tokens by probability
        # 2. return a list or list of list
        # **********************
        padded_seq = self.truncate_pad(splited_seq, num_steps, padding_token)

        return self.__getitem__(padded_seq)



class MutilVocab:
    def __init__(self, data, AA_tok_len=2):
        """
        Args:
            data (_type_):
            AA_tok_len (int, optional): Defaults to 1.
            start_token (bool, optional): True is required for encoder-based model.
        """
        ## Load train dataset
        self.x_data = data
        self.tok_AA_len = AA_tok_len
        self.default_AA = list("RHKDESTNQCGPAVILMFYW")
        # AAs which are not included in default_AA
        self.tokens = self._token_gen(self.tok_AA_len)

        self.token_to_idx = {k: i + 4 for i, k in enumerate(self.tokens)}
        self.token_to_idx["[PAD]"] = 0  ## idx as 0 is PAD
        self.token_to_idx["[CLS]"] = 1  ## idx as 1 is CLS
        self.token_to_idx["[SEP]"] = 2  ## idx as 2 is SEP
        self.token_to_idx["[MASK]"] = 3  ## idx as 3 is MASK
        
    def split_seq(self):
        self.X = [self._seq_to_tok(seq) for seq in self.x_data]
        return self.X
    
    def tok_idx(self, seqs):
        '''
        note: ensure to execut this function before truancate_pad
        '''

        seqs_idx = []
        for seq in seqs:
            seq_idx = []
            for s in seq:
                seq_idx.append(self.token_to_idx[s])
            seqs_idx.append(seq_idx)

        return seqs_idx



    def _token_gen(self, tok_AA_len: int, st: str = "", curr_depth: int = 0):
        """Generate tokens based on default amino acid residues
            and also includes "X" as arbitrary residues.
            Length of AAs in each token should be provided by "tok_AA_len"

        Args:
            tok_AA_len (int): Length of token
            st (str, optional): Defaults to ''.
            curr_depth (int, optional): Defaults to 0.

        Returns:
            List: List of tokens
        """
        curr_depth += 1
        if curr_depth <= tok_AA_len:
            l = [
                st + t
                for s in self.default_AA
                for t in self._token_gen(tok_AA_len, s, curr_depth)
            ]
            return l
        else:
            return [st]

    def _seq_to_tok(self, seq: str):
        """Convert each token to index

        Args:
            seq (str): AA sequence

        Returns:
            list: A list of indexes
        """

        seq_idx = []
        
        seq_idx += ["[CLS]"]

        for i in range(len(seq) - self.tok_AA_len + 1):
            curr_token = seq[i : i + self.tok_AA_len]
            seq_idx.append(curr_token)
        seq_idx += ['[SEP]']
        return seq_idx
