# Text Preprocessing and aggregating summary statistics
directory = os.getcwd() + '\\'

def feature_engineering(
        df='Dataset for processing',
        text_column='excerpts',
        target_column='target',
        get_sent_lengths=True,
        get_word_lengths=True,
        get_pos_counts=True,
        top_n_word_count=True,
        list_of_pos="list of pos's to get counts of: e.g. ['nouns', 'verbs', 'adverbs', 'adjectives']",
        file_name='train'):
    # Access the text
    excerpts = df[text_column]

    # Lets clean the strings
    def clean_text(string):
        string = pd.Series(string)
        # remove white space and lowercase words
        string = string.apply(str.strip).apply(str.lower)
        # remove '\n'
        string = string.map(lambda x: re.sub('\\n', ' ', str(x)))
        # remove punctuations
        string = string.map(lambda x: re.sub(r"[^\w\s]", '', str(x)))

        return string

    excerpts = clean_text(excerpts)
    print('Step 1: Text has been cleaned')

    # Create an English language SnowballStemmer object
    stemmer = SnowballStemmer("english")

    # Defining a function to perform both stemming and tokenization
    def tokenize_and_stem(text):

        # Tokenize by sentence, then by word
        tokens = [y for x in sent_tokenize(text) for y in word_tokenize(x)]

        # Filter out raw tokens to remove noise
        filtered_tokens = [
            token for token in tokens if re.search('[a-zA-Z]', token)
        ]
        # Stem the filtered_tokens
        stems = [stemmer.stem(word) for word in filtered_tokens]

        # Remove stopwords
        cleaned = [x for x in stems if x not in stopwords.words('english')]

        # Join the cleaned tokens together
        joined = ' '.join(cleaned)

        return joined

    print(
        'Step 2: Executing the tokenizer and stemmer...might take a while..sit tight...'
    )

    tokenized_stemmed_path = directory + file_name + '_tokenized_stemmed.csv'
    if not os.path.exists(tokenized_stemmed_path):
        excerpts = excerpts.progress_apply(tokenize_and_stem)
        print('saved as csv file..')
        excerpts.to_csv(tokenized_stemmed_path)
    else:
        excerpts = pd.read_csv(tokenized_stemmed_path,
                               skiprows=1,
                               header=None,
                               index_col=0,
                               squeeze=True)
    excerpts = excerpts.fillna('')
    print('Done')

    # Lets get the number of top words that overlap in each document
    if top_n_word_count==True:

        # Instantiate the TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english',
                                min_df=3,
                                max_features=None,
                                ngram_range=(1, 1),
                                use_idf=True,
                                smooth_idf=True,
                                sublinear_tf=True,
                                tokenizer=None,
                                preprocessor=None)

        # get the relevant vectors
        def get_tfidf(excerpts):
            excerpts_tfidf = tfidf.fit_transform([x for x in excerpts])
            return excerpts_tfidf

        print('Getting tfidf vectors of cleaned and tokenized text...')
        excerpts_tfidf = get_tfidf(excerpts)

        feature_array = np.array(tfidf.get_feature_names())
        tfidf_sorting = np.argsort(excerpts_tfidf.toarray()).flatten()[::-1]

        # Get the top n words from the tfidf vectorizer
        def top_n_words(tfidf_sorting):
            n = 1510
            top_n = feature_array[tfidf_sorting][:n]
            top_pop = list()
            for i in tqdm(excerpts, colour='green'):
                counter = 0
                for x in top_n:
                    if x in i:
                        counter += 1
                top_pop.append(counter)
            return top_pop

        print(
            'Retrieving the top 1510 words and counting instances of top words in every document...'
        )
        df['top_pop'] = top_n_words(tfidf_sorting)
    else:
        pass

    # Word Length
    def get_word_length(string):
        string = string.split()
        temp = np.array([len(x) for x in string])
        temp = temp.mean()
        return temp

    def remove_repeating_words(string):
        string = string.split()
        k = []
        for i in (string):
            if (string.count(i) > 1 and (i not in k) or string.count(i) == 1):
                k.append(i)
        return ' '.join(k)

    print('Keeping only non repeating words in the corpus for gathering statistics...')
    non_repeating_word_corpus = excerpts.progress_apply(
        remove_repeating_words
    )  # The excerpts passed here has been cleaned and tokenized and stemmed

    # Get sent_lenghts and word_count after getting the non_repeating_word_corpus

    print('Getting mean word length of every document...')
    df['mean_word_length'] = excerpts.progress_apply(get_word_length)

    # sentence lengths: Character count
    if get_sent_lengths == True:
        sent_lenghts = pd.Series([
            len(non_repeating_word_corpus[i])
            for i in range(len(non_repeating_word_corpus))
        ])
        print('Getting character counts aka sentence lengths of each document...')
        df['sent_lengths'] = sent_lenghts
    else:
        pass

    # word_count
    if get_word_lengths == True:
        print('Getting word counts for each document')
        df['word_count'] = non_repeating_word_corpus.apply(
            str.split).apply(len)
    else:
        pass


    # Part of speech Magic
    if get_pos_counts==True:
        pos = {
            'verbs': ['VB', 'VBG', 'VBN', 'VBP', 'VBD', 'VBZ'],
            'nouns': ['NN', 'NNS', 'NNP', 'NNPS'],
            'adverbs': ['RB', 'RBR', 'RBS'],
            'adjectives': ['JJ', 'JJR', 'JJS'],
            'pronouns': ['PRP', 'PRP$']
        }

        def get_counts(x='list of sentences',
                       y='list of pos to get counts',
                       repeating_words='Yes',
                       correlation_=False,
                       add_to_dataframe=True,
                       temp_tagged_texts=[]):
            if not bool(temp_tagged_texts)==True:
                if repeating_words == 'Yes':
                    for_tagging = clean_text(x)
                    for_tagging = for_tagging.apply(str.split)
                    for_tagging = for_tagging.to_list()
                elif repeating_words == 'No':
                    remove_repeating_words_path = directory + file_name + '_repeating_words_removed.csv'
                    if not os.path.exists(remove_repeating_words_path):
                        for_tagging = x.progress_apply(remove_repeating_words)
                        print('saved as csv file..')
                        for_tagging.to_csv(remove_repeating_words_path)
                        for_tagging = for_tagging.apply(str.split)
                        for_tagging = for_tagging.to_list()
                    else:
                        for_tagging = pd.read_csv(remove_repeating_words_path,
                                           skiprows=1,
                                           header=None,
                                           index_col=0,
                                           squeeze=True)
                        for_tagging = for_tagging.apply(str.split)
                        for_tagging = for_tagging.to_list()
                print('Tagging parts of speech...')
                temp_tagged_texts = pos_tag_sents(for_tagging)
            else:
                pass
            pos_list = []
            for i in tqdm(range(len(temp_tagged_texts))):
                a, b = zip(*temp_tagged_texts[i])
                pos_list.append(list(b))
            num_pos = list()
            for i in tqdm(pos_list):
                cnt = Counter(i)
                z = 0
                for j in y:
                    z += cnt[j]
                num_pos.append(z)
            if add_to_dataframe == True:
                df_name = 'num_'+ p
                df[str(df_name)] = num_pos
            elif add_to_dataframe == False:
                return np.corrcoef(num_pos, df['target_column'])
            return temp_tagged_texts


        print('Getting parts of speech counts of: ', list_of_pos)
        tagged_texts=[]
        for ind, p in enumerate(list_of_pos):
            print('Getting counts of all ', p, '...')
            if ind == 0:
                tagged_texts.append(get_counts(x=df[text_column],
                           y=pos[p],
                           repeating_words='No',
                           correlation_=False,
                           add_to_dataframe=True, temp_tagged_texts = tagged_texts))
            if ind > 0:
                tagged_texts.append(get_counts(x=df[text_column],
                           y=pos[p],
                           repeating_words='No',
                           correlation_=False,
                           add_to_dataframe=True, temp_tagged_texts = tagged_texts[0]))
        print('All necessary parts of speech counts have been processed')
    else:
        pass

    print('Done')

    return df
