import os
import csv
import json
import numpy as np
import pandas as pd
import gc
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn import preprocessing as pp
from bs4 import BeautifulSoup
from scipy import stats
from html.parser import HTMLParser


warnings.filterwarnings("ignore")


PATH_TO_DATA = '../initial data'
AUTHOR = 'Vladimir_Sapachev' 


def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result


def get_actual_content(html):
    soup = BeautifulSoup(html)
    content = soup.find('div', class_='postArticle-content')
    if not content:
        return html
    return content


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

    
def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def get_post_tags(html):
    soup = BeautifulSoup(html)
    try:
        tags = [li.text for li in soup.find('ul', class_="tags--postTags").find_all('li')]
    except Exception:
        return 'No_tags'   
    tags = ['_'.join(tag.split()) for tag in tags]
    if not tags:
        return 'No_tags'
    return ' '.join(tags)


def get_fig_num(html):
    soup = BeautifulSoup(html)
    content = soup.find('div', class_='postArticle-content')
    if not content:
        return -1
    return len(content.find_all('figure'))


def get_data_from_json(source):
    """ Create three types of train/test files:
        1) Domain, author and published time
        2) HTML content
        3) Articles titles
    """
    with open(os.path.join(PATH_TO_DATA, f'{source}.json'), encoding='utf-8') as f:
        for line in f.readlines():
            json_data = read_json_line(line)
            with open(os.path.join(PATH_TO_DATA, f'{source}_data.csv'), 'a', 
                      encoding='utf-8', newline='\n') as csvfile:
                fw = csv.writer(csvfile, delimiter=';')
                try:
                    domain = json_data['domain']
                except KeyError:
                    domain = ''
                try:
                    author = json_data['meta_tags']['author']
                except KeyError:
                    author = ''
                try:
                    pub_time = json_data['meta_tags']['article:published_time']
                except KeyError:
                    pub_time = ''
                fw.writerow([domain, author, pub_time])

            with open(os.path.join(PATH_TO_DATA, f'{source}_content.txt'), 'a', encoding='utf-8') as fc:
                fc.write(json_data['content'].replace('\n', ' ').replace('\r', ' ') + '\n')

            with open(os.path.join(PATH_TO_DATA, f'{source}_title.txt'), 'a', encoding='utf-8') as ft:
                ft.write(json_data['title'].replace('\n', ' ').replace('\r', ' ') + '\n')
                
                
def get_data_from_content(source):
    """ Get article tags and html inside class 'postArticle-content'
    """
    df = pd.DataFrame()
    df['content'] = pd.read_csv(
        os.path.join(PATH_TO_DATA, f'{source}_content.txt'), sep='\n', header=None)[0]
    df['post_tags'] = df.content.apply(get_post_tags)
    df['actual_content'] = df.content.apply(get_actual_content)
    df.post_tags.to_csv(
        os.path.join(PATH_TO_DATA, f'{source}_post_tags.txt'), index=None, header=None)
    df.actual_content.to_csv(
        os.path.join(PATH_TO_DATA, f'{source}_actual_content.txt'), index=None, header=None)    
    

def get_data_from_actual_content(source):
    """ Get article text without html tags, text length and number of figures in article'
    """
    df = pd.DataFrame()
    df['actual_content'] = pd.read_csv(
        os.path.join(PATH_TO_DATA, f'{source}_actual_content.txt'), header=None)[0]
    df['post_text'] = df.actual_content.apply(strip_tags)
    df['post_text_len'] = df.post_text.str.len()
    df['fig_num'] = df.actual_content.apply(get_fig_num)
    df.post_text.to_csv(
        os.path.join(PATH_TO_DATA, f'{source}_post_text.txt'), index=None, header=None)
    df.post_text_len.to_csv(
        os.path.join(PATH_TO_DATA, f'{source}_post_text_len.txt'), index=None, header=None)
    df.fig_num.to_csv(
        os.path.join(PATH_TO_DATA, f'{source}_fig_num.txt'), index=None, header=None)


# parse json and create txt files with data
get_data_from_json('train')
get_data_from_json('test')
get_data_from_content('train')
get_data_from_content('test')
get_data_from_actual_content('train')
get_data_from_actual_content('test')


def create_base_df(source):
    df = pd.DataFrame()
    cols = ['domain', 'author', 'pub_time']
    df[cols] = pd.read_csv(
        os.path.join(PATH_TO_DATA, f'{source}_data.csv'), sep=';', header=None, parse_dates=[2])
    df['title'] = pd.read_csv(
        os.path.join(PATH_TO_DATA, f'{source}_title.txt'), sep='\n', header=None)[0]
    df['post_text'] = pd.read_csv(
        os.path.join(PATH_TO_DATA, f'{source}_post_text.txt'), header=None)[0]
    df['post_tags'] = pd.read_csv(
        os.path.join(PATH_TO_DATA, f'{source}_post_tags.txt'), sep='\n', header=None)[0]
    df['fig_num'] = pd.read_csv(
        os.path.join(PATH_TO_DATA, f'{source}_fig_num.txt'), sep='\n', header=None)[0]
    df['post_text_len'] = pd.read_csv(
        os.path.join(PATH_TO_DATA, f'{source}_post_text_len.txt'), sep='\n', header=None)[0]
    df['content'] = pd.read_csv(
        os.path.join(PATH_TO_DATA, f'{source}_content.txt'), sep='\n', header=None)[0]
    return df


train_df = create_base_df('train')
train_df['log_recommends'] = pd.read_csv(
    os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'))['log_recommends']
train_df['is_train'] = 1
train_df['id'] = pd.read_csv(
    os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'))['id']

test_df = create_base_df('test')
test_df['log_recommends'] = 999
test_df['is_train'] = 0
test_df['id'] = pd.read_csv(
    os.path.join(PATH_TO_DATA, 'sample_submission.csv'))['id']

df = pd.concat([train_df, test_df]).sort_values(by='pub_time')
df.loc[df.post_text.isna(), 'post_text'] = df.loc[df.post_text.isna(), 'content']

# get rid of old articles - last 15 months go to train dataset
idx_old = df[df.pub_time <= '2016-03-31'].shape[0]
idx_train_test_split = train_df.shape[0]

del train_df
del test_df
df = df.drop(columns=['content'])
gc.collect()

# useful features after CV
df['post_text_len_log'] = np.log1p(df.post_text_len)
author_df = pd.get_dummies(df['author'], prefix='author')
fig_num_df = pd.get_dummies(df['fig_num'], prefix='fig_num')
domain_df = pd.get_dummies(df['domain'], prefix='domain')

# TF-IDF vectorizers for article text, title and tags
vect_post_text = TfidfVectorizer(
    ngram_range=(1, 4),
    max_features=400000,
    sublinear_tf=True,
    stop_words='english',
)

vect_title = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=200000,
    sublinear_tf=True,
)

vect_post_tags = TfidfVectorizer(
    ngram_range=(1, 1),
    max_features=50000, 
    sublinear_tf=True,
)

# create matrices with TF-IDF vectorizers
def get_tfidf_matrices(vect, feature):
    train = vect.fit_transform(df[idx_old:idx_train_test_split][feature])
    test = vect.transform(df[idx_train_test_split:][feature])
    return train, test

X_train_post_text, X_test_post_text = get_tfidf_matrices(vect_post_text, 'post_text')
X_train_title, X_test_title = get_tfidf_matrices(vect_title, 'title')
X_train_post_tags, X_test_post_tags = get_tfidf_matrices(vect_post_tags, 'post_tags')

# train/test split
X_train = df[idx_old:idx_train_test_split]
y_train = df[idx_old:idx_train_test_split].log_recommends
X_test = df[idx_train_test_split:]

# scaling features
X_train_scaled = pp.StandardScaler().fit_transform(X_train[['post_text_len_log']])
X_test_scaled = pp.StandardScaler().fit_transform(X_test[['post_text_len_log']])

# combine all stuff into two train/test matrices 
X_train_sparse = hstack([
    X_train_scaled,
    X_train_title,
    X_train_post_tags,
    X_train_post_text,
    domain_df[idx_old:idx_train_test_split],
    fig_num_df[idx_old:idx_train_test_split],
    author_df[idx_old:idx_train_test_split],
]).tocsr()

X_test_sparse = hstack([
    X_test_scaled,
    X_test_title,
    X_test_post_tags,
    X_test_post_text,
    domain_df[idx_train_test_split:],
    fig_num_df[idx_train_test_split:],
    author_df[idx_train_test_split:],
]).tocsr()

# train model and get predictions
ridge = Ridge(random_state=17, alpha=2)
ridge.fit(X_train_sparse, np.log1p(y_train))
ridge_pred_test = np.expm1(ridge.predict(X_test_sparse))
df['log_recommends'][idx_train_test_split:] = ridge_pred_test

# Approximate date when claps system was introduced
idx_clap = df[df.pub_time <= '2017-08-15'].shape[0]

# Adjust predictions after 2017-08-15 due to difference beetween likes and claps systems.
# Maximum adjustment around 50-100 claps (4-5 in log), and minimal on both ends (around 0 and 12 log).
# Overall mean approximately 4.333
df['log_recommends'][idx_clap:] = df['log_recommends'][idx_clap:] + 0.55 + 15 * stats.lognorm.pdf(
    df['log_recommends'][idx_clap:], s=0.6, scale=10, loc=-3)

submit_df = df[idx_train_test_split:][['id', 'log_recommends']]

# create submission file
submit_df = submit_df.set_index('id').sort_index()
submit_df.to_csv(os.path.join(PATH_TO_DATA, f'submission_medium_{AUTHOR}_solution.csv'))
