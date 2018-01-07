import spacy

nlp = spacy.load('en')

# Process sentences 'Hello, world. Natural Language Processing in 10 lines of code.' using spaCy
doc = nlp(u'Hello, world. Waking up in the morning is the best thing. blah blah')

# Get first token of the processed document
token = doc[0]
#print(token)

# Print sentences (one sentence per line)
#for sent in doc.sents:
 #   print(sent)

# For each token, print corresponding part of speech tag
#for token in doc:
  #  print('{} - {}'.format(token, token.pos_))


def tokens_to_root(token_param):
    """
    Walk up the syntactic tree, collecting tokens to the root of the given `token`.
    :param token_param: Spacy token
    :return: list of Spacy tokens
    """
    tokens_to_r = []
    while token_param.head is not token_param:
        tokens_to_r.append(token_param)
        token_param = token_param.head
        tokens_to_r.append(token_param)

    return tokens_to_r

# For every token in document, print it's tokens to the root
# for token in doc:
#    print('{} --> {}'.format(token, tokens_to_root(token)))

doc_2 = nlp(u'I went to Kisumu where I met my old friend Jack from campus.')

for ent in doc_2.ents:
   print('{} - {}'.format(ent, ent.label_))

# Print noun chunks for doc_2
print([chunk for chunk in doc_2.noun_chunks])

