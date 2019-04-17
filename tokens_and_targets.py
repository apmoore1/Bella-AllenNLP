import argparse
from pathlib import Path

from bella.tokenisers import spacy_tokeniser
from bella.data_types import TargetCollection
import spacy


def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def get_end_offset(word: str, start_offset:int) -> int:
    return len(word) + start_offset

if __name__=='__main__':
    force_space_help = 'Whether or not to force a space between the target '\
                       'token start and end with the other tokens surrounding'\
                       ' the target token(s)'

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=parse_path,
                        help='Directory to the TDSA dataset e.g. ~/.Bella/Datasets')
    parser.add_argument("dataset_name", type=str, 
                        choices=['Laptop', 'Restaurant', 'Election'])
    parser.add_argument("--force_space", action='store_true', help=force_space_help)
    args = parser.parse_args()

    data_dir = args.data_dir
    dataset_name = args.dataset_name
    tokeniser = spacy.blank('en')
    split_names = ['Train', 'Val', 'Test']

    for split_name in split_names:
        dataset_fp = Path(data_dir, f'{dataset_name} {split_name}')
        dataset = TargetCollection.load_from_json(dataset_fp)

        retrieve_target_from_tokens = []
        cannot_retrieve_target_from_tokens = []

        for target in dataset.data_dict():
            text = target['text']
            target_word = target['target'].strip()
            target_start_offset = target['spans'][0][0]
            target_end_offset = target['spans'][0][1]
            if args.force_space:
                before_text = text[:target_start_offset].strip()
                after_text = text[target_end_offset:].strip()
                new_target_word = f' {target_word} '
                target_start_offset = len(before_text) + 1
                target_end_offset = target_start_offset + len(target_word)
                text = f'{before_text}{new_target_word}{after_text}'

            tokens = tokeniser(text)
            got_start = False
            for index, token in enumerate(tokens):
                start_offset = token.idx
                end_offset = get_end_offset(token.text, token.idx)
                if start_offset == target_start_offset:
                    got_start = True
                if got_start:
                    if end_offset == target_end_offset:
                        retrieve_target_from_tokens.append((tokens, target))
                        break
                if not got_start and start_offset > target_start_offset:
                    cannot_retrieve_target_from_tokens.append((tokens, target))
                    break
                if got_start and end_offset > target_end_offset:
                    cannot_retrieve_target_from_tokens.append((tokens, target))
                    break
                if index == (len(tokens) - 1):
                    cannot_retrieve_target_from_tokens.append((tokens, target))
                    break

        dataset_size = len(dataset.data_dict())
        missed_labels_through_tokenisation = len(cannot_retrieve_target_from_tokens)
        missed_error = (f'Number missed {missed_labels_through_tokenisation} '
                        f'plus number captured {len(retrieve_target_from_tokens)}'
                        f' should equal the whole dataset {dataset_size}')
        assert missed_labels_through_tokenisation + len(retrieve_target_from_tokens) == dataset_size, missed_error

        missed_labels_through_tokenisation_percentage = missed_labels_through_tokenisation / dataset_size
        missed_labels_through_tokenisation_percentage = missed_labels_through_tokenisation_percentage * 100
        print(f'{dataset_name} {split_name} can get {missed_labels_through_tokenisation}'
              f'({missed_labels_through_tokenisation_percentage}%) of targets')