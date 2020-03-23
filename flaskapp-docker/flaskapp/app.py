from flask import Flask, request, jsonify
import re
from collections import OrderedDict
import numpy as np
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')

dler = nltk.downloader.Downloader()
dler._update_index()
dler._status_cache['panlex_lite'] = 'installed'
nltk.download('punkt', halt_on_error=False)
nltk.download('averaged_perceptron_tagger', halt_on_error=False)
nltk.download('maxent_ne_chunker', halt_on_error=False)
nltk.download('words', halt_on_error=False)

app = Flask(__name__)

d = 0.85 # damping coefficient, usually is .85
min_diff = 1e-5 # convergence threshold
steps = 10 # iteration steps
node_weight = None # save keywords and its weight

@app.route('/summary', methods = ['POST'])
def text_summary():
    def set_stopwords(stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def get_vocab(sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = symmetrize(g)
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        return g_norm

    def analyze(text, candidate_pos=['NOUN', 'PROPN','VERB'], window_size=4, lower=False, stopwords=list(), number = 10):
        """Main function to analyze text"""

        # Set stop words
        set_stopwords(stopwords)
        # Pare text by spaCy
        doc = nlp(text)
        # Filter sentences
        sentences = sentence_segment(doc, candidate_pos, lower) # list of list of words
        # Build vocabulary
        vocab = get_vocab(sentences)
        # Get token_pairs from windows
        token_pairs = get_token_pairs(window_size, sentences)
        # Get normalized matrix
        g = get_matrix(vocab, token_pairs)
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        # Iteration
        previous_pr = 0
        for epoch in range(steps):
            pr = (1-d) + d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        node_weight = node_weight
        node_weight = OrderedDict(sorted(node_weight.items(), key=lambda t: t[1], reverse=True))
        keyword = []
        for i, (key, value) in enumerate(node_weight.items()):
            keyword.append(key)
            #print(key + ' - ' + str(value))
            if i > number:
                break
        return keyword

    def command_detected(sentence):
        # Detects whether a given String sentence is a command or action-item
        tagged_sentence = pos_tag(word_tokenize(sentence));
        first_word = tagged_sentence[0];
        pos_first = first_word[1];
        first_word = first_word[0].lower()
        for word in prohibited_command_words:
            if word in sentence:
                return False
        for word in command_words:
            if word in sentence:
                return True
        # Checks whether the first sentence is a Modal Verb or other type of Verb that is not a gerund
        if (pos_first == "VB" or pos_first == "VBZ" or pos_first == "VBP") and first_word[-3:] != "ing":
            return True
        return False

    def retrieve_action_items():
        # Returns a list of the sentences containing action items.
        action_items = []
        for sentence in tokenized_transcript:
            possible_command = command_detected(str(sentence))
            if possible_command is True:
                action_items += [(str(sentence))]
        return action_items

    text=request.json
    text = text['data'].replace('Speaker ','')
    source = re.sub(r'\d\s+\d{1,2}\:\d{2}', '', text)
    source = re.sub(r'\s+',' ',source)

    Keywords = analyze(source, candidate_pos = ['NOUN', 'PROPN', 'VERB'], window_size=4, lower=False)
    
    tokenized_transcript = sent_tokenize(source)
    LANGUAGE = "English"
    parser = PlaintextParser.from_string(source,Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(parser.document, len(tokenized_transcript)*0.05)
    transcript_summary = []
    for sentence in summary:
        transcript_summary.append(str(sentence))

    command_words = ["can you", "would you", "can we", "you should", "we should", "we need to", "you need to", "ensure", "make sure", "make it", "we want to", "we must", "you must", "you have to", "we have to" "homework"]
    prohibited_command_words = ["Let me", "?"]
    Action_item = retrieve_action_items()
    
    result = {"keywords :" : Keywords, 'Summary :' : transcript_summary,  'Action Items :': Action_item}
    return jsonify(result)

if __name__=='__main__':
    app.run(debug = True)

