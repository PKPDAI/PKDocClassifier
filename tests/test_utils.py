import torch
from transformers import BertModel, BertTokenizer
from pk_classifier.utils import preprocess, encode_bert, tokens_emb, sentence_emb
import spacy

FAKE_TEXTS = ['Candida infections have increased due to the growth and expansion of susceptible patient '
              'populations.', 'The clearance (CLz) of midazolam was 3.25l/h', '']

SPACY_MODEL = spacy.load("en_ner_bc5cdr_md")
BERT_TOKENIZER = BertTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
BERT_MODEL = BertModel.from_pretrained("monologg/biobert_v1.1_pubmed", output_hidden_states=True)


def test_preprocess():
    docs = list(SPACY_MODEL.pipe(FAKE_TEXTS))
    docs_preprocessed = [preprocess(spacy_obj=doc, ident='', unifier=' ;; ', ngram=1, masking=False) for doc in docs]
    assert len(docs_preprocessed) == len(docs)
    assert docs_preprocessed[0] == 'candida ;; infect ;; increas ;; growth ;; expans ;; suscept ;; patient ;; popul'
    assert docs_preprocessed[1] == 'clearanc ;; clz ;; midazolam ;; ## ;; lh'
    assert docs_preprocessed[2] == ''

    # Adding a title identifier and masking drugnames
    docs_preprocessed_title_drugs = [preprocess(spacy_obj=doc, ident='T', unifier=' ;; ', ngram=1, masking=True)
                                     for doc in docs]
    assert docs_preprocessed_title_drugs[1] == 'clearancT ;; clzT ;; drugnamT ;; ##T ;; lhT'

    # Check bigrams and trigrams
    docs_preprocessed_bigr = [preprocess(spacy_obj=doc, ident='T', unifier=' ;; ', ngram=2, masking=True)
                              for doc in docs]
    docs_preprocessed_trigr = [preprocess(spacy_obj=doc, ident='T', unifier=' ;; ', ngram=3, masking=True)
                               for doc in docs]

    assert docs_preprocessed_bigr[0] == 'candidaT ;; infectT ;; increasT ;; growthT ;; expansT ;; susceptT ;; ' \
                                        'patientT ;; populT ;; candidaT infectT ;; infectT increasT ;; increasT ' \
                                        'growthT ;; growthT expansT ;; expansT susceptT ;; susceptT patientT ;; ' \
                                        'patientT populT'

    assert docs_preprocessed_trigr[1] == 'clearancT ;; clzT ;; drugnamT ;; ##T ;; lhT ;; clearancT clzT ;; clzT ' \
                                         'drugnamT ;; drugnamT ##T ;; ##T lhT ;; clearancT clzT drugnamT ;; clzT ' \
                                         'drugnamT ##T ;; drugnamT ##T lhT'


def test_encode_bert():
    docs = [str_text if not str_text == '' else 'xxxxx' for str_text in FAKE_TEXTS]
    docs_emb = [
        encode_bert(inp_text=input_text, nlp_model=SPACY_MODEL, bert_model=BERT_MODEL, bert_tokenizer=BERT_TOKENIZER,
                    maxmin=False) for input_text in docs]
    assert len(docs_emb) == len(docs)
    assert docs_emb[0].shape == docs_emb[1].shape == docs_emb[2].shape == (768,)
    docs_emb_minmax = [encode_bert(inp_text=input_text, nlp_model=SPACY_MODEL, bert_model=BERT_MODEL,
                                   bert_tokenizer=BERT_TOKENIZER, maxmin=True) for input_text in docs]

    assert docs_emb_minmax[0].shape == docs_emb_minmax[1].shape == docs_emb_minmax[2].shape == (768 * 3,)


def test_tokens_and_sentence_pooler():
    sentence = FAKE_TEXTS[1]
    tokens_raw, token_embeddings = tokens_emb(sentence=sentence, inp_tokenizer=BERT_TOKENIZER, inp_model=BERT_MODEL)
    n_tokens = 21
    assert len(tokens_raw) == token_embeddings.shape[0] == n_tokens
    assert token_embeddings.shape == torch.Size([21, 13, 768])
    assert tokens_raw[0] == "[CLS]"
    assert tokens_raw[-1] == "[SEP]"
    sentence_embeddings = sentence_emb(token_embeddings)
    assert sentence_embeddings.shape == torch.Size([19, 768])

