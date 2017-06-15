# BinaryClassifier
Basic binary classifier using naive bayes that can tell whether a translation was created by a human or by a machine. Sample input is translation from Chinese to English

To run the code, use the following command

python3 MTClassifier.py <train file>  <test file, ending in .unlabeled>

For example:

python3 MTClassifier.py split3.labeled split3.unlabeled

The labeled test file will be outputed with the same file name as the test file but with '.predicted' file extension

nltk needs to be installed on the machine or virtual environment to run the code.
