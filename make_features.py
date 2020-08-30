""" This script trains & applies a pre-processing SKlern pipeline and saves the output for future analyses"""

import pandas as pd
from Analyses.Embeddings.distributional_encoders import ProcBoWEmb
import os


def processthem(inp_path, out_path, pipeline, embedding=None):
    data_df = pd.read_parquet(inp_path)
    if embedding:
        preprocessed_df = pipeline.fit_transform(data_df)  # , embedder__emb_type=embedding)
        preprocessed_df['pmid'] = data_df['pmid']
    else:
        preprocessed_df = pipeline.fit_transform(data_df)
        preprocessed_df['pmid'] = data_df['pmid']
    preprocessed_df.to_parquet(out_path)


if __name__ == '__main__':
    # 0. Data path
    data_dir = os.path.join("Analyses", "data")
    # 1. Input paths
    path_dev = os.path.join(data_dir, "subsets", "dev_subset.parquet")
    path_test = os.path.join(data_dir, "subsets", "test_subset.parquet")
    # 2. Output paths
    tail = "biobert_bow_uni_meanmaxmin"
    path_out_dev = os.path.join(data_dir, "encoded", "fields", "dev_preproc_"+tail+".parquet")
    path_out_test = os.path.join(data_dir, "encoded", "fields", "test_preproc_"+tail+".parquet")

    # 3.Process normal

    # processthem(path_dev, path_out_dev, ProcBoW,None)
    # processthem(path_test, path_out_test, ProcBoW,None)

    # 3. Process word2vec
    processthem(path_dev, path_out_dev, ProcBoWEmb, embedding='biobert')
    processthem(path_test, path_out_test, ProcBoWEmb, embedding='biobert')

## Read
# dev_df = pd.read_parquet(path_dev)
# test_df = pd.read_parquet(path_test)
# extra_dev_df = pd.read_parquet(path_extra_dev)
#
## Process
# preprocessed_dev_df = ProcBoW.fit_transform(dev_df)
# preprocessed_test_df = ProcBoW.fit_transform(test_df)
# preprocessed_extra_dev_df = ProcBoW.fit_transform(extra_dev_df)
#
## Add PMID:
#
# preprocessed_dev_df['pmid'] = dev_df['pmid']
# preprocessed_test_df['pmid'] = test_df['pmid']
# preprocessed_extra_dev_df['pmid'] = extra_dev_df['pmid']
#
## Write
# preprocessed_dev_df.to_parquet(path_out_dev)
# preprocessed_test_df.to_parquet(path_out_test)
# preprocessed_extra_dev_df.to_parquet(path_out_extra_dev)
#
a = 1
# Load Features
#            bow_features = pd.read_parquet(path_preproc)
#            # Main parameters
#            rd_seed = 10042006  # define random seed
#            per = 0.20  # Proportion of documents going to test
#            # Define Encoders and Decoder
#            encoder = CountVectorizer(tokenizer=Tokenizer, min_df=2, ngram_range=(1, 1), lowercase=False, preprocessor=None)
#            decoder = XGBClassifier(random_state=rd_seed, nthread=12)  # , max_depth=3
#            # Define pipeline
#            EncDecPip = Pipeline([
#                ('vect', encoder),
#                ('clf', decoder)
#            ])
#
#            # Select
#            exclude = dev_df['source'] != 'random_ids'
#            X = bow_features['BoW_Ready'][exclude]
#            Y = dev_df['label'][exclude]
#            pmids = dev_df['pmid'][exclude]
#            # training test
#            x_train, x_val, y_train, y_val, ids_train, ids_val = train_test_split(X, Y, pmids,
#                                                                                  test_size=per,
#                                                                                  shuffle=True,
#                                                                                  random_state=rd_seed,
#                                                                                  stratify=Y)
#            # Train
#            EncDecPip.fit(x_train, y_train)
#            y_pred = EncDecPip.predict(x_val)  # Test
#            print("Results on development set:", metrics.classification_report(y_val, y_pred))
#
#            ################################################ On Test #######################################################
#
#            path_test = os.path.join("data", "classification", "ready", "test_lab2.parquet")
#            path_test_preprocessed = os.path.join("data", "classification", "preprocessed", "unig_test_preproc.parquet")
#            test_df = pd.read_parquet(path_test)
#            feat_test = pd.read_parquet(path_test_preprocessed)
#
#            X_test = feat_test['BoW_Ready']
#            y_test = test_df['label']
#            y_pred = EncDecPip.predict(X_test)
#
#            print("Results on test set:", metrics.classification_report(y_test, y_pred))
#
#            #
#
#            exclude = dev_df['source'] == 'pk_random'
#            X = bow_features['BoW_Ready'][exclude]
#            Y = dev_df['label'][exclude]
#
#            X_test = feat_test['BoW_Ready']
#            y_test = test_df['label']
#
#            # Train
#            EncDecPip.fit(X, Y)
#            y_pred = EncDecPip.predict(X)  # Test
#            print("Results on development set:", metrics.classification_report(Y, y_pred))
#
#            y_pred = EncDecPip.predict(X_test)
#
#            print("Results on test set:", metrics.classification_report(y_test, y_pred))
#
#            # out = Preprocessor.transform(X[0:3000])
#
#            a = 1
#
#
#            def preprocess(in_string):
#                if in_string == '':
#                    return ['']
#                else:
#                    in_string = check_ascii(in_string)
#                    sp_obj = nlp(in_string)  # convert to spacy object
#                    tokens = [tok.text.lower() for tok in sp_obj if not tok.is_punct]  # Remove punctuation and lowercase
#                    #   clean_tokens = list(flatten([cleanup(tok) for tok in tokens if tok not in spacy_stopwords]))  # Cleans and removes stopwords
#                    #   stemmed_tokens = [ps.stem(tok) for tok in clean_tokens]
#                    # TODO: Cleanup patterns
#                    return 'hi'
#
#
#            tokenized = dev_df['abstract'].apply(preprocess)
#
#            # DONT' ERASE
#
#
#            BoWer = CountVectorizer(tokenizer=Tokenizer, min_df=1, ngram_range=(1, 1), lowercase=False)
#            Embedder = Word2Vec()
#
#            Preprocessor = Pipeline([
#                ('features', FeatureUnion(transformer_list=[
#                    ('proc_abs_bow', Pipeline([
#                        ('colex', TextSelector('abstract')),
#                        ('bow', BoWer)
#                    ])
#                     ),
#                    ('proc title_bow', Pipeline([
#                        ('colex', TextSelector('title')),
#                        ('bow', BoWer)
#
#                    ])
#                     ),
#                    ('proc_abs_w2vec', Pipeline([
#                        ('colex', TextSelector('abstract')),
#                        ('vectorizer', Embedder)
#
#                    ]))
#
#                ], n_jobs=-1))
#            ])
#
#            out = Preprocessor.fit_transform(X[0:2700])
#            # out = Preprocessor.transform(X[0:3000])
#            u = out.toarray()
#
#            # Check
#            nlp = spacy.load("en_ner_bc5cdr_md")
#            tt = X['abstract'].fillna('')
#            n = 5
#
#            list(nlp(tt[n]).vector) == list(u[n][-201:-1])
#
#            a = 1
#
#            #    classifier = Pipeline([
#            #        ('features', FeatureUnion([
#            #            ('text', Pipeline([
#            #
#            #                ('colext', TextSelector('Text')),
#            #                ('tfidf', TfidfVectorizer(tokenizer=Tokenizer, stop_words=stop_words,
#            #                         min_df=.0025, max_df=0.25, ngram_range=(1,3))),
#            #                ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), #for XGB
#            #            ])),
#            #            ('words', Pipeline([
#            #                ('wordext', NumberSelector('TotalWords')),
#            #                ('wscaler', StandardScaler()),
#            #            ])),
#            #        ])),
#            #        ('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),
#            #    #    ('clf', RandomForestClassifier()),
#            #        ])
#            #
#            #
#            #    text_clf = Pipeline([
#            #        ('features', FeatureUnion([
#            #            'proc_abs', Pipeline([
#            #
#            #                ('colext', TextSelector('abstract')),
#            #                ('bow', CountVectorizer(tokenizer=Tokenizer, lowercase=False,  min_df=5)
#            #                 ]
#            #                 )
#            #                ]
#            #            )
#            #
#            #
#            #
#            #        ])),
#            #
#            #         ('vect', CountVectorizer(analyzer='word', tokenizer=dummy_fun,
#            #                                  preprocessor=dummy_fun, lowercase=False, min_df=5)),
#            #     ])
#            #
#
