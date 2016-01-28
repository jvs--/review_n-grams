## Classifieng positive and negative movie reviews with n-gram models

In this project we train language models to distinguish between positive and negative movie reviews based on the concept of Markov chains, using character-level models of different length combined with different smoothing techniques to account for data sparseness.

# The Concept

In natural languages certain characters and sequences of characters have a specific frequency distribution. For example, in English the character sequence "ment" as in "achievement", "payment" or "enjoyment" shows up more often than for example "axol" as in "axolotl". Conventionally, a character sequence of length n-1 is called n-gram. Certain n-grams show up with a high frequency since they catch certain language structures. In the case of "ment" what we observe is a surface expression of an underlying syntactic structure of the language, namely the formation of nouns from verbs.
We can use such knowledge about the probability with which certain n-grams occur to reason about the text they show up in.

# Dependencies 
This implementation requires installed NLTK modules and corpora. For instructions on how to set up your system we refer you to http://nltk.org/install.html
