# TODO
1. Add tests for the augmeted data iterator

# Target Extraction
We are treating this problem as a sequence labelling problem. As the given datasets are not pre-tokenised first they must be tokenised. The tokeniser we use is Spacy. However as the text is not pre-tokenised we want to first see how many of the tokens line up with the span offsets that are sets as the target word and that must be predicted in this task. To do this we created the `./tokens_and_targets.py` script which prints out the number of targets (samples) that the target word(s) does not neatly fit within the tokens created from the tokeniser we call these `tokenisation errors`. An example of this error can be seen below:

`Turned on BBCQT and thought I was watching a translation of a Greek political show.Anti austerity from the SNP obviously funded on oil#bbcqt`

Where the target is `bbcqt` but the token that spacy found was `oil#bbcqt` therefore predicting that as the target word would be incorrect as it incorporates more than just the target. So running the following command:
``` bash
python tokens_and_targets.py ~/.Bella/Datasets/ Laptop
python tokens_and_targets.py ~/.Bella/Datasets/ Restaurant
python tokens_and_targets.py ~/.Bella/Datasets/ Election
```
We find the following for each of the datasets: 0.94%, 0.18%, and 1.53% of the test datasets to have `tokenisation errors` which is minimal but note worthy. One thing we did find is that in the Laptop training dataset one of the targets included a space within sentence id 1436. Here we only report errors on the test sets as with the training and validation sets we can force a space between the target word and text within the text and change the span offsets and thus remove the `tokenisation errors` as shown by the following commands:
``` bash
python tokens_and_targets.py ~/.Bella/Datasets/ Laptop --force_space
python tokens_and_targets.py ~/.Bella/Datasets/ Restaurant --force_space
python tokens_and_targets.py ~/.Bella/Datasets/ Election --force_space
```
The reason we do not do this for the test datasets is because we want a fair compriason with previous work and thus do not change the dataset in any way to avoid the tokenisation errors.