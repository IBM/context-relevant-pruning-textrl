from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from crest.helper.utils import read_file
stop_words = set(stopwords.words('english'))
import string

def is_whitespace(c, use_space=True):
    if (c == " " and use_space) or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def compact_text(paragraph_text):
    doc_tokens = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(' ')
                doc_tokens.append(c)
                prev_is_whitespace = False
            else:
                doc_tokens.append(c)
                prev_is_whitespace = False
    return ''.join(doc_tokens)

# removes punctutions and stop words from the text
def normalize_text(text):
    state = text.lower()
    out = state.translate(str.maketrans('', '', string.punctuation))
    out = word_tokenize(out)
    s_ws = [w for w in out if not w in stop_words]
    return s_ws