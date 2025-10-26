# 2 biLSTM ASR models trained with custom CTC loss function on both quantized inputs and MFCCs for .wav files

This is the directory for Information Extraction Project 3.
There are two folders: data, and checkpoints.

Checkpoints contains model_best_primary.pt and model_best_secondary.pt, the models for the primary and secondary systems, ie discrete and MFCC input systems.

Data contains all of the files given for the assignment, unchanged.

In addition to all of the given code files, with TODOs completed, there is ctc_loss.py, which defines class CTCLoss, for the extra credit. There is also the IE Project 3 Report pdf, which contains the loss function graphs. These graphs are also available as pngs, primary_loss.png and secondary_loss.png.

Finally, there are test results files for each system, Discrete_test_results.txt and MFCC_test_results.txt

NOTE ABOUT CTC LOSS IMPLEMENTATION:
The numbers it returns are not exactly the same as the torch CTC outputs. However, they are extremely close and in the same order of magnitude. This loss function can also be used to train and the loss decreases, although it doesn't train as effectively as with the torch CTCLoss.
