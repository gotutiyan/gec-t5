import argparse
from gec_t5 import retokenize
from pathlib import Path

def main(args):
    srcs = Path(args.input).read_text().rstrip().split('\n')
    tok_srcs = [retokenize(s) for s in srcs]
    Path(args.output).write_text('\n'.join(tok_srcs))

def cli_main():
    args = get_parser()
    main(args)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)