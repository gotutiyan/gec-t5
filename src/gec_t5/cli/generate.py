import argparse
from gec_t5 import generate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.restore_dir)
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    predictions = generate(
        model=model,
        tokenizer=tokenizer,
        sources=open(args.input).read().rstrip().split('\n'),
        batch_size=args.batch_size,
        retok=args.retok
    )
    print('\n'.join(predictions))

def cli_main():
    args = get_parser()
    main(args)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--restore_dir', required=True)
    parser.add_argument('--retok', action='store_true')
    args = parser.parse_args()
    return args