from typing import List, Tuple
from transformers import PreTrainedTokenizer

class Dataset():
    def __init__(
        self,
        srcs: List[str],
        trgs: List[str],
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        is_inference: bool = False
    ) -> None:
        self.srcs = srcs
        self.trgs = trgs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = len(srcs)
        self.is_inference = is_inference
        if self.is_inference:
            self.orig_index = list(range(len(self.srcs)))
    
    def __getitem__(self, idx: int) -> dict:
        src = self.srcs[idx]
        trg = self.trgs[idx]
        src_encode = self.tokenizer(
            src,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        trg_encode = self.tokenizer(
            trg,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        labels = trg_encode['input_ids']
        labels[labels[:] == self.tokenizer.pad_token_id] = -100
        return_dict = {
            'input_ids': src_encode['input_ids'].squeeze(),
            'attention_mask': src_encode['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
        if self.is_inference:
            return_dict['orig_index'] = self.orig_index[idx]
        return return_dict

    def __len__(self):
        return self.len
    
    def sort_by_length(self):
        '''This is for inference.
        '''
        data = [
            (src, trg, orig_index) \
            for src, trg, orig_index in zip(self.srcs, self.trgs, self.orig_index)
        ]
        data = sorted(data, key=lambda x:len(x[0]))
        self.srcs = [x[0] for x in data]
        self.trgs = [x[1] for x in data]
        self.orig_index = [x[2] for x in data]