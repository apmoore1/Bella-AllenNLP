# TODO
1. Add variational dropout if needed else it can be done through the AugmentedLSTM by allennlp. Check if the dropout is applied to the output of the LSTM it is not normally.


From what I have established the Augmented LSTM does do variational dropout on the time dependent/recurrent part of the LSTM, however the stacked LSTM that uses these augemnted LSTM's does not apply variational or any type of dropout to the INput or output to the stacked LSTM's.

The normal LSTM in PyTorch on the other hand only uses naive dropout but this is done on the output of every layer that is not the last layer in the stacked LSTM structure.

To solve the LSTM problem we can copy and paste the code from the stacked LSTM and add the already implemented variation dropout to the input of each stacked LSTM