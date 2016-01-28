from nltk.corpus import movie_reviews
from math import log
import re
# attribution: utility functions for evaluation by S. Evert:
# alphabet for n-gram models (all other characters are deleted)
alphabet = "abcdefghijklmnopqrstuvwxyz "
re_nonalphabet = re.compile("[^%s]" % alphabet)
re_whitespace = re.compile(r"\s+")

# -- getPOS() and getNEG() are fudged to ensure models aren't trained on the new test data --
# returns positive movie reviews (only first 1000 of each set to keep amount of data manageable)
def getPOS():
  return [normalize(movie_reviews.raw(fileids=file)) for file in movie_reviews.fileids("pos")[200:1000]] + [normalize(movie_reviews.raw(fileids=file)) for file in movie_reviews.fileids("pos")[800:1000]]

# returns all Republican Training sentences
def getNEG():
  return [normalize(movie_reviews.raw(fileids=file)) for file in movie_reviews.fileids("neg")[200:1000]] +  [normalize(movie_reviews.raw(fileids=file)) for file in movie_reviews.fileids("neg")[800:1000]]
  

# normalize string (normalise whitespace, fold to lowercase, remove all non-alphabet characters such as digits and punctuation) 
def normalize(s):
  s = re_whitespace.sub(" ", s.lower()) # fold to lowercase and normalise whitespace
  s = re_nonalphabet.sub("", s)         # delete all non-alphabet characters
  return s[-500:]                       # cut each review to last 500 characters

def evaluate(smoothingMethod, POSfreqs, NEGfreqs, POStest, NEGtest, name="", terse=False, validate=False):  
  """
Evaluate the accuracy of trained n-gram models (given by frequency lists and a smoothing method)
in distinguishing between positive and negative movie reviews.

Arguments:
  - smoothingMethod: a function f(w, freqs) that takes two arguments, a string <w> of length <n> and
      a frequencyList <freqs> as explained below, and returns the smoothed conditional probability
      of the n-gram <w>, i.e. Pr(w_n | w_1 .. w_{n-1}) -- or Pr(w[n-1]|w[0:n-1]) in Python indexing
  - POSfreqs: a list of dictionaries containing n-gram frequency counts for positive movie reviews,
      which can be used by smoothingMethod to compute conditional n-gram probabilities; it is
      customary to list unigram frequency counts first, followed by bigrams, etc.; however, any other
      data structure can be passed provided that it is accepted as a second argument by smoothingMethod
  - NEGfreqs: same as above, for negative movie reviews
  - POStest: list of positive reviews (characters strings) used to evaluate the trained n-gram models
  - NEGtest: same as above, for negative reviews
  - name: name of the smoothing/interpolation method (for evaluation report)
  - terse: if True, show compact evaluation report in single line
  - validate: test whether smoothingMethod produces a valid probability distribution without zeroes
"""

  # for the contest evaluation, we force validation and terse reporting
  validate = True
  terse = True
  
  # we also use a new "secret" test set, ignoring the parameters POStest and NEGtest
  # (this should have been held-out data, but by a silly mistake all movie reviews were included in the data set; so we use the first 200 reviews in each set, making sure they aren't part of the training data)
  POStest = [normalize(movie_reviews.raw(fileids=file)) for file in movie_reviews.fileids("pos")[0:200]]
  NEGtest = [normalize(movie_reviews.raw(fileids=file)) for file in movie_reviews.fileids("neg")[0:200]]

  if len(POSfreqs) != len(NEGfreqs):
    raise Exception("Both lists of n-gram frequency counts (POSfreqs / NEGfreqs) must have the same length!")
  if len(POStest) != len(NEGtest):
    raise Exception("Test set must contain same number of positive and negative reviews for a fair evaluation!")
  n = len(POSfreqs) - 1 # size of n-gram model, with POSfreqs = [corpus size, {unigrams}, {bigrams}, ...]
  
  n_guesses = 0
  guess = [0, 0, 0]   # n-gram classifier: none / pos / neg
  gold = [0, 0, 0]    # gold standard: none / pos / neg
  correct = [0, 0, 0] # whether classifier is correct: correct / wrong / no decision
  zeroProb = 0
  inconsistent = 0
  posCount = len(POStest)
  negCount = len(NEGtest)
    
  for category, test_set in [(1, POStest), (2, NEGtest)]:
    for review in test_set:
      review = normalize(review) # make sure that strings are properly normalized
      ngram = [' '] * n # first n-gram only consists of stop-characters
      POSlogp = 0       # calculate probabilities on logarithmic scale to avoid underflow
      NEGlogp = 0
      POSzero = False  # flag is set if any of the conditional n-gram probabilities is zero (cannot be represented on log scale)
      NEGzero = False
      for character in review:
        ngram.pop(0)
        ngram.append(character)

        ngram_string = ''.join(ngram) # convert ngram in fifo to string
        history = ngram_string[:-1]
        next_char = ngram_string[-1:]

        # call smoothingMethod to calculate conditional n-gram probability
        POSngramp = smoothingMethod(ngram_string, POSfreqs)
        NEGngramp = smoothingMethod(ngram_string, NEGfreqs)

        if POSngramp <= 0.0:
          POSzero = True
          if validate:
            print "Error: Pr(%s|%s) = 0 (positive model)" % (next_char, history)
        else:
          POSlogp += log(POSngramp)

        if NEGngramp <= 0.0:
          NEGzero = True
          if validate:
            print "Error: Pr(%s|%s) = 0 (negative model)" % (next_char, history)
        else:
          NEGlogp += log(NEGngramp)

        if validate:
          POScumprob = 0.0
          NEGcumprob = 0.0
          for c in alphabet:
            POScumprob += smoothingMethod(history + c, POSfreqs)
            NEGcumprob += smoothingMethod(history + c, NEGfreqs)
          if (abs(POScumprob - 1.0) > 1e-6):
            print "Error: Sum Pr(*|%s) = %f does not sum to 1 (positive model)" % (history, POScumprob)
            inconsistent += 1
          if (abs(NEGcumprob - 1.0) > 1e-6):
            print "Error: Sum Pr(*|%s) = %f does not sum to 1 (positive model)" % (history, NEGcumprob)
            inconsistent += 1
      
      # zero probability flagged: set log(Pr(w)) = -Inf for classifier, issue warning later
      if POSzero:
        zeroProb += 1
        POSlogp = -9e99  # practically -Inf
      if NEGzero:
        zeroProb += 1
        NEGlogp = -9e99

      # determine classifier decision and check whether it is correct
      if POSlogp > NEGlogp:
        classifier = 1 # decision: pos
      elif POSlogp < NEGlogp:
        classifier = 2 # decision: neg
      else:
        classifier = 0 # no decision (e.g. if both models have Pr(w) = 0)

      guess[classifier] += 1
      gold[category] += 1

      n_guesses += 1
      if classifier == 0:
        correct[2] += 1
      elif classifier == category:
        correct[0] += 1
      else:
        correct[1] += 1

  accuracy = float(correct[0]) / n_guesses * 100
  baseline = float(max(gold)) / n_guesses * 100
  if (gold[1] >= gold[2]):
    majority = "positive"
  else:
    majority = "negative"

  if terse:
    print 'accuracy:%6.2f%% (%3d/%3d/%3d) %4d positive,%4d negative  %s' % (accuracy, correct[0], correct[1], correct[2], guess[1], guess[2], name)
  else:
    print ' EVALUATION: %s' % (name)
    print '='*35
    print 
    print 'Correct        : %3d reviews' % correct[0]
    print 'Wrong          : %3d reviews' % correct[1]
    print 'Not classified : %3d reviews' % correct[2]
    print '-'*35
    print 'Accuracy       :%6.2f%%' % (accuracy)
    print 'Baseline       :%6.2f%%  (always classify as %s)' % (baseline, majority)
    print
    if zeroProb > 0:
      print 'Warning: probability = 0 estimated in %d cases -- need better smoothing!' % zeroProb
      print

  if validate and (zeroProb > 0 or inconsistent > 0):
    error_message = "N-Gram model (%s) failed validation: %d zero probabilities, %d inconsistencies" % (name, zeroProb, inconsistent)
    raise Exception(error_message)

    
