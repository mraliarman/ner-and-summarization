import os
import nltk
import spacy
import pandas as pd
from nltk import ne_chunk
from summa import summarizer
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

def q_1():
    text = "Natural language processing is fun! This text is a sample text."

    tokens = nltk.word_tokenize(text)
    print("Tokenization:")
    print(tokens)

    pos_tags = nltk.pos_tag(tokens)
    print("\nPOS Tagging:")
    print(pos_tags)

    chunk_grammar = r"""
        NP: {<DT>?<JJ>*<NN>}
    """

    chunk_parser = nltk.RegexpParser(chunk_grammar)
    noun_phrases = chunk_parser.parse(pos_tags)

    print("\nNoun Phrase Chunking:")
    print(noun_phrases)

    noun_phrases.draw()

    text_vp = "She decided to take a stroll in the park."

    tokens_vp = nltk.word_tokenize(text_vp)
    pos_tags_vp = nltk.pos_tag(tokens_vp)

    chunk_grammar_vp = r"""
    VP: {<PRP>?<VBD>?<TO>?<VB>?<DT>?<JJ>*<NN>?}
    """

    chunk_parser_vp = nltk.RegexpParser(chunk_grammar_vp)
    verb_phrases = chunk_parser_vp.parse(pos_tags_vp)

    print("\nVerb Phrase Chunking:")
    print(verb_phrases)

    verb_phrases.draw()

def q_2():
    with open("Rock.txt", "r", encoding="utf-8") as file:
        rock_text = file.read()

    chunk_grammar_iob = r"""
    NP: {<DT>?<JJ>*<NN>}
    """

    chunk_parser_iob = nltk.RegexpParser(chunk_grammar_iob)

    tokens_iob = word_tokenize(rock_text)
    pos_tags_iob = nltk.pos_tag(tokens_iob)

    noun_phrases_iob = ne_chunk(pos_tags_iob, binary=True)

    print("\nIOB Encoding for Noun Phrases:")
    print(noun_phrases_iob)

    stanford_ner_path = "stanford-ner.jar"
    stanford_ner_model = "english.all.3class.distsim.crf.ser.gz"

    stanford_tagger = StanfordNERTagger(stanford_ner_model, stanford_ner_path)

    java_path = r"C:\Program Files\Java\jdk-21\bin"
    os.environ['JAVAHOME'] = java_path

    sentences_stanford = sent_tokenize(rock_text)
    tokens_stanford = [word_tokenize(sentence) for sentence in sentences_stanford]

    named_entities_stanford = stanford_tagger.tag_sents(tokens_stanford)

    named_entities_stanford = [ent for sent in named_entities_stanford for ent in sent if ent[1] in ['PERSON', 'LOCATION']]

    print("\nNamed Entities (Stanford NER):")
    for entity in named_entities_stanford:
        print(entity)

    nlp = spacy.load("en_core_web_sm")
    doc_spacy = nlp(rock_text)

    named_entities_spacy = [(ent.text, ent.label_) for ent in doc_spacy.ents if ent.label_ in ['PERSON', 'GPE']]

    print("\nNamed Entities (Spacy NER):")
    for entity in named_entities_spacy:
        print(entity)

    print("\nComparison between Stanford NER and Spacy NER:")
    print("Named Entities (Stanford NER):", named_entities_stanford)
    print("Named Entities (Spacy NER):", named_entities_spacy)

def q_3():
    with open("Rock.txt", "r", encoding="utf-8") as file:
        rock_text = file.read()

    stop_words = set(stopwords.words("english"))

    tokens = word_tokenize(rock_text)

    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    preprocessed_text = ' '.join(filtered_tokens)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])

    feature_names = tfidf_vectorizer.get_feature_names_out()

    top_keywords = [feature_names[idx] for idx in tfidf_matrix.indices]

    print("\nTop 10 Key Words Extracted with TF-IDF:")
    print(top_keywords)

    sentences = sent_tokenize(rock_text)

    key_sentences = [sentence for sentence in sentences if any(keyword in sentence for keyword in top_keywords)]
    key_sentences = key_sentences[:5]

    print("\nSummary Based on Key Sentences:")
    print(' '.join(key_sentences))

    text_rank_summary = summarizer.summarize(rock_text, ratio=0.15) 

    print("\nSummary Based on TextRank:")
    print(text_rank_summary)



def get_recommendations(df, movie_title):
    movie_index = df[df['title'] == movie_title].index[0]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    sim_scores = list(enumerate(cosine_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6] 

    movie_indices = [x[0] for x in sim_scores]
    recommendations = df['title'].iloc[movie_indices]
    
    return list(zip(recommendations.values, [round(score, 4) for _, score in sim_scores]))

def q_4():
    df = pd.read_csv("tmdb_5000_movies.csv")
    selected_columns = ['title', 'genres', 'keywords']
    df = df[selected_columns]
    df['combined_features'] = df['genres'] + ' ' + df['keywords']

    recommendations_mortal_kombat = get_recommendations(df, 'Mortal Kombat')
    recommendations_flywheel = get_recommendations(df, 'Flywheel')
    recommendations_frozen = get_recommendations(df, 'Frozen')

    print("Recommendations for Mortal Kombat:")
    for title, score in recommendations_mortal_kombat:
        print(f"{title} - Similarity Score: {score}")

    print("\nRecommendations for Flywheel:")
    for title, score in recommendations_flywheel:
        print(f"{title} - Similarity Score: {score}")

    print("\nRecommendations for Frozen:")
    for title, score in recommendations_frozen:
        print(f"{title} - Similarity Score: {score}")

#q_1()
#q_2()
#q_3()
q_4()