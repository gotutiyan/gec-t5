
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .retokenize import retokenize
from tqdm import tqdm

def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sources: list[str],
    batch_size: int=32,
    retok: bool = False
) -> float:
    num_sents = len(sources)
    model.eval()
    pred_ids = [0] * num_sents

    # In the inferece, there is no target text,
    #  so it uses sources and targets due to the interface problem.
    dataset = Dataset(
        srcs=sources,
        trgs=sources,
        tokenizer=tokenizer,
        max_len=128,
        is_inference=True
    )
    dataset.sort_by_length()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader)) as pbar:
            for _, batch in pbar:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                ids = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=128,
                    num_beams=5,
                    do_sample=False,
                    length_penalty=1.0
                )
                orig_index = batch['orig_index'].tolist()
                for i, p_id in enumerate(ids):
                    pred_ids[orig_index[i]] = p_id
    predictions = tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    if retok:
        predictions = [retokenize(p) for p in predictions]
    return predictions