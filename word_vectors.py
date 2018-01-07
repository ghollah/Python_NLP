from __future__ import unicode_literals
from subject_object_extraction import findSVOs
from subject_object_extraction import findSubs

import spacy
from spacy.en import English
from numpy import dot
from numpy.linalg import norm

nlp = spacy.load('en')

parser = English()

# you can access known words from the parser's vocabulary
king = parser.vocab['parent']

# cosine similarity
cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))

# gather all known words, take only the lowercased versions
allWords = list({w for w in parser.vocab if w.has_vector and w.orth_.islower() and w.lower_ != "parent"})

# sort by similarity to king
allWords.sort(key=lambda w: cosine(w.vector, king.vector))
allWords.reverse()
print("Top 10 most similar words to parent:")
for word in allWords[:10]:
    print(word.orth_)

# Let's see if it can figure out this analogy
# Man is to King as Woman is to ??
man = parser.vocab['father']
woman = parser.vocab['daughter']
result = king.vector - man.vector + woman.vector

# gather all known words, take only the lowercased versions
allWords = list({w for w in parser.vocab if w.has_vector and w.orth_.islower() and w.lower_ != "king" and w.lower_ != "man" and w.lower_ != "woman"})
# sort by similarity to the result
allWords.sort(key=lambda w: cosine(w.vector, result))
allWords.reverse()
print("\n----------------------------\nTop 3 closest results for king - man + woman:")
for word in allWords[:3]:
    print(word.orth_)

# can still work even without punctuation
parse = parser("he and his brother drunk wine")
print(findSVOs(parse))

token = nlp(u"he and his brother shot me and my sister")
print(findSubs(token[0]))


