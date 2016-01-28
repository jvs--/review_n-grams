#!/usr/bin/env python
from util import evaluate, getPOS, getNEG, alphabet

# extracting positive and negative reviews
POSreviews = getPOS()
NEGreviews = getNEG()

# seperating corpus in training and test set
POStrain = POSreviews[0:800]    # 80% for training
NEGtrain = NEGreviews[0:800]
POStest = POSreviews[800:1000]  # 20% for testing
NEGtest = NEGreviews[800:1000]


# set up list of dictionaries for n-gram counts 
# from unigram up to ngram specified by n
maxn = 5
POSfreqs = [0] + [{}.copy() for x in range(maxn)]
NEGfreqs = [0] + [{}.copy() for x in range(maxn)]

# compute n-gram frequency counts for positive and negative reviews
print "Computing n-gram frequency counts ... please be patient ..."
for freqs, reviews in [(POSfreqs, POStrain), (NEGfreqs, NEGtrain)]:
  for review in reviews:
    fifo = [' '] * maxn  # initialise n-gram storage (fifo pipe) with blanks as stop characters
    
    for character in ' '+review: # make sure that n-gram containing only stop chars is in each dictionary
      fifo.pop(0)
      fifo.append(character)
      
      freqs[0] += 1
      for idx in range(maxn):
        n = idx + 1
        ngram = ''.join(fifo[:n]) # n-gram = first n characters in fifo
        if ngram not in freqs[n]:
          freqs[n][ngram] = 1
        else:
          freqs[n][ngram] += 1

# smoothing functions return conditional probability Pr(x1 x2 .. xi-1 xi|x1 x2 .. xi-1)
# for ngram = x1 x2 .. xi-1 xi

absSigma = len(alphabet) # alphabet size (including blank as word separator and stop character)

# MLE = maximum likelihood estimate (unsmoothed)
def MLE(ngram, freqs):
  n = len(ngram)
  # note that freqs[n] contains n-gram frequencies, freqs[n-1] contains (n-1)-gram frequencies, etc.
  if ngram in freqs[n]:
    numerator = freqs[n][ngram]
  else:
    return 0.0
  
  if n == 1:
    denominator = freqs[0] # unigram probability (unconditional): f(x) / corpus_size
  else:
    history = ngram[0:n-1] # conditional ngram probability: f(x_1 .. x_n) / f(x_1 .. x_{n-1})
    if history in freqs[n-1]:
      denominator = freqs[n-1][history]
    else:
      return 0.0
  
  return float(numerator)/denominator

# Uniform distribution (without training data)
#def Uniform(ngram, freq):
#  return 1.0 / absSigma

# Add-one smoothing
def AddOne(ngram, freqs):
  n = len(ngram)
  if ngram in freqs[n]:
    numerator = freqs[n][ngram] + 1
  else:
    numerator = 1

  if n == 1:
    denominator = freqs[0] + absSigma
  else:
    history = ngram[0:n-1]
    if history in freqs[n-1]:
      denominator = freqs[n-1][history] + absSigma
    else:
      denominator = absSigma

  return float(numerator)/denominator

# Add-lambda smoothing
def AddLambda(ngram, freqs):
  global Lambda # meta-parameter given by global variable Lambda
  n = len(ngram)
  if ngram in freqs[n]:
    numerator = freqs[n][ngram] + Lambda
  else:
    numerator = Lambda

  if n == 1:
    denominator = freqs[0] + absSigma * Lambda
  else:
    history = ngram[0:n-1]
    if history in freqs[n-1]:
      denominator = freqs[n-1][history] + absSigma * Lambda
    else:
      denominator = absSigma * Lambda

  return float(numerator)/denominator


## --- EVALUATION ---
##

# Evaluating unsmoothed maximum likelohood estimats will show you instances 
# in which the model fails due to datasparsness. We don't want to see those 
# anymore since we have already fixed this adding smoothing. 
# evaluate(MLE, POSfreqs[0:4], NEGfreqs[0:4], POStest, NEGtest, "MLE with 3-grams", validate=True)

evaluate(AddOne, POSfreqs[0:2], NEGfreqs[0:2], POStest, NEGtest, "1-grams with add-1 smoothing", validate=True)

evaluate(AddOne, POSfreqs[0:3], NEGfreqs[0:3], POStest, NEGtest, "2-grams with add-1 smoothing", validate=True)

evaluate(AddOne, POSfreqs[0:4], NEGfreqs[0:4], POStest, NEGtest, "3-grams with add-1 smoothing", validate=True)

evaluate(AddOne, POSfreqs[0:5], NEGfreqs[0:5], POStest, NEGtest, "4-grams with add-1 smoothing", validate=True)

evaluate(AddOne, POSfreqs[0:6], NEGfreqs[0:6], POStest, NEGtest, "5-grams with add-1 smoothing", validate=True)

Lambda = 6
evaluate(AddLambda, POSfreqs[0:6], NEGfreqs[0:6], POStest, NEGtest, "5-grams with add-lambda smoothing", validate=True)

Lambda = 2
evaluate(AddLambda, POSfreqs[0:6], NEGfreqs[0:6], POStest, NEGtest, "5-grams with add-lambda smoothing", validate=True)

