#!conda install -c conda-forge xgboost -y
#!conda install -c anaconda gensim -y
#!conda install -c conda-forge wordcloud -y
#!conda install -c conda-forge phantomjs -y
#!conda intall selenium -y
#!conda install -c conda-forge firefox geckodriver -y


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from pprint import pprint
from xgboost import XGBClassifier
from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import brown
from nltk import FreqDist
from wordcloud import WordCloud 
from collections import OrderedDict
from matplotlib import pyplot as plt


## Clusters visualization
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource
from bokeh.io import export_png

# Criando um Data Frame com 5 colunas: diretorio, categoria, nome_arq, titulo, texto

path = 'data/arquivos_bbc/bbc/'

categories = []
titles = []
all_data = []
#df = pd.DataFrame()
# dirname
for dirname, categoryname, filenames in os.walk(path):
    # filename
    for filename in filenames:
        if filename == 'README.TXT':
            filenames.remove(filename)
        else:
            # Absolute path
            current_file = os.path.abspath(os.path.join(dirname, filename))
            open_file = open(current_file, 'r', encoding="latin-1")
            # text_data
            text_data = open_file.read().split('\n')
            text_data = list(filter(None, text_data))
            titles.append(text_data[0])
            all_data.append((dirname, dirname.rsplit('/',1)[1], filename, text_data[0], text_data[1:]))
            #data_df = f"Directory: {dirname}, Category: {dirname.rsplit('/',1)[1]}, FileName: {filename}, Title: {text_data[0]}, Text: {text_data[1:]}"
            #print(data_df)
df = pd.DataFrame(all_data, columns=['directory', 'category', 'fileName', 'title', 'text'])
df['text'] = df.text.astype(str)
print(df.head()) 


df.describe()


df.category.value_counts()


bar_plot=df.category.value_counts().plot(kind='barh', figsize=(8, 6), color='teal')
plt.xlabel("Nº de Artigos", labelpad=14)
plt.ylabel("Categoria", labelpad=14)
plt.title("Nº de artigos na categoria", y=1.02, color='navy')

for index, value in enumerate(df.category.value_counts()):
    plt.text(value, index, str(value))
plt.savefig('num_artigos_categoria.png')

## Label Encoder

# 0 - business, 1 -entertainment, 2 - politics, 3 - sport, 4 - tech
label_enc = LabelEncoder()
df['label'] = label_enc.fit_transform(df['category'])
df.head()


# Um conjunto de palavras
df_txt = np.array(df['text'])


df_txt


## Pré-processe e vetorização do texto


stopwords = nltk.corpus.stopwords.words('english')

def docs_preprocessor(docs):
    # recuperando apenas letras
    tokenizer = RegexpTokenizer('[A-Za-z]\w+')
    
    for idx in range(len(docs)):
         # converte para minusculo
        docs[idx] = docs[idx].lower() 
        # Dividido em palavras
        docs[idx] = tokenizer.tokenize(docs[idx])  
    
    # Lematizar todas as palavras com len>2 em documentos
    lemmatizer = WordNetLemmatizer()
    docs = [[nltk.stem.WordNetLemmatizer().lemmatize(token) for token in doc if len(token) > 2 and token not in stopwords] for doc in docs]
         
    return docs


df_txt = docs_preprocessor(df_txt)


'''
* bigrama
Um bigrama ou digrama é uma sequência de dois elementos adjacentes de uma sequência de tokens, que normalmente são letras, sílabas ou palavras.

*trigram
Sigla constituída de três caracteres reunidos

'''


# Adicione bigramas e trigramas a documentos (somente aqueles que aparecem 10 vezes ou mais)
bigram = Phrases(df_txt, min_count=10)
trigram = Phrases(bigram[df_txt])

for idx in range(len(df_txt)):
    for token in bigram[df_txt[idx]]:
        if '_' in token:
            df_txt[idx].append(token)
    for token in trigram[df_txt[idx]]:
        if '_' in token:
            df_txt[idx].append(token)

## Removendo palavras raras:

# Cria uma representação de dicionário dos documentos
dictionary = Dictionary(df_txt)
print('Nº de palavras únicas em documentos iniciais:', len(dictionary))

# Filtra palavras que ocorrem em menos de 10 documentos ou em mais de 20% dos documentos
dictionary.filter_extremes(no_below=10, no_above=0.2)
print('Nº de palavras únicas depois de remover palavras raras e comuns:', len(dictionary))

df['text2'] = df_txt

df['text3'] = [' '.join(map(str, j)) for j in df['text2']]

df.iloc[1475:1480,:]

### Vetores de palavras

vectorizer = TfidfVectorizer(input='content', analyzer = 'word', lowercase=True, stop_words='english',\
                                   ngram_range=(1, 3), min_df=40, max_df=0.20,\
                                  norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
text_vector = vectorizer.fit_transform(df.text3)
dtm = text_vector.toarray()
features = vectorizer.get_feature_names()

h = pd.DataFrame(data = text_vector.todense(), columns = vectorizer.get_feature_names())
h.iloc[990:1000,280:300]

corpus = [dictionary.doc2bow(txt) for txt in df_txt]

print(f'Número de tokens exclusivos: {len(dictionary)}')
print(f'Número de documentos: {len(corpus)}')

## Classification Model

X = text_vector
y = df.label.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

### RandomForestClassifier
svc1 = RandomForestClassifier(random_state = 42)
svc1.fit(X_train, y_train)
svc1_pred = svc1.predict(X_test)
print(f"Test Accuracy: {svc1.score(X_test, y_test)*100:.3f}%")

### XGBClassifier
svc2 = XGBClassifier(random_state = 42, use_label_encoder=False)
svc2.fit(X_train, y_train)
svc2_pred = svc2.predict(X_test)
print(f"Test Accuracy: {svc2.score(X_test, y_test)*100:.3f}%")

### SGDClassifier
svc3 = SGDClassifier(random_state = 42)
svc3.fit(X_train, y_train)
svc3_pred = svc3.predict(X_test)
print(f"Test Accuracy: {svc3.score(X_test, y_test)*100:.3f}%")

### KNeighborsClassifier
svc4 = KNeighborsClassifier()
svc4.fit(X_train, y_train)
svc4_pred = svc4.predict(X_test)
print(f"Test Accuracy: {svc4.score(X_test, y_test)*100:.3f}%")

# Função para calcular o erro absoluto médio
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Recebe um modelo, treina o modelo e avalia o modelo no conjunto de teste
def fit_and_evaluate(model):
    
    # Traina o modelo
    model.fit(X_train, y_train)
    
    # previsões e avaliação
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    
    # Retorna a métrica de desempenho
    return model_mae

svc1_mae = fit_and_evaluate(svc1)
svc2_mae = fit_and_evaluate(svc2)
svc3_mae = fit_and_evaluate(svc3)
svc4_mae = fit_and_evaluate(svc4)

plt.style.use('fivethirtyeight')
# fig = plt.figure(figsize(8, 6))

# Dataframe para armazenar os resultados
model_comparison = pd.DataFrame({'model': ['RandomForest Classifier', 'XGBClassifier', 
                                           'SGDClassifier', 'KNeighborsClassifier'
                                          ],
                                 'mae': [svc1_mae, svc2_mae, 
                                         svc3_mae, svc4_mae]})

# Horizontal bar chart of test mae
model_comparison.sort_values('mae', ascending = False).plot(x = 'model', y = 'mae', kind = 'barh',
                                                           color = 'yellow', edgecolor = 'black')

plt.ylabel('')
plt.yticks(size = 14)
plt.xlabel('Mean Absolute Error')
plt.xticks(size = 14)
plt.title('Comparação de Modelos no Teste MAE', size = 20)
plt.savefig('Comparacao_Modelos_Teste_MAE.png')

## Nuvem de etiquetas
## Gobal
wc = WordCloud(width = 800, height = 800, 
                background_color ='white', 
               stopwords=stopwords,
                min_font_size = 10, random_state=42).generate(df.text3.to_string())

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wc)
plt.tight_layout(pad = 0) 
plt.axis("off")
plt.savefig('Nuvem_de_etiquetas_global.png')
# plt.show()

## por categoria
for x in df.category.unique():
    wc = WordCloud(width = 800, height = 800, background_color ='white', stopwords=stopwords,
                   min_font_size = 10, random_state=42)
    wc.generate(df.text3[(df.category == x)].to_string())
    
    plt.imshow(wc)
    plt.tight_layout(pad = 0) 
    plt.title(x)
    plt.axis("off")
    plt.savefig(f'Nuvem_de_etiquetas_categoria_{x}.png')
    # plt.show()

# Clustering
# Definir parâmetros de treinamento
num_topics = 5
chunksize = 1000 # Número de documentos a serem considerados de uma só vez (afeta o consumo de memória)
passes = 30 # Número de passagens por documentos
iterations = 500
eval_every = 1  

# cria um índice para o dicionário de palavras
temp = dictionary[0] 
id2word = dictionary.id2token

model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                       alpha='auto', eta='auto', random_state=78, \
                       iterations=iterations, num_topics=num_topics, \
                       passes=passes, eval_every=eval_every)

# Frequência das principais palavras em cada tópico
def explore_topic(lda_model, topic_number, topn, output=True):
    terms = []
    for term, frequency in lda_model.show_topic(topic_number, topn=topn):
        terms += [term]
        if output:
            print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))   
    return terms

topic_summaries = []

print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')
for i in range(num_topics):
    print('Topic '+str(i)+'\n')
    tmp = explore_topic(model,topic_number=i, topn=10, output=True)
    topic_summaries += [tmp[:5]]
    print('\n')

# Atribui um rotulo interpretável
top_labels = {0: 'business', 1:'sport', 2:'tech', 3:'entertainment', 4:'politics'}

top_dist =[]
for d in corpus:
    tmp = {i:0 for i in range(num_topics)}
    tmp.update(dict(model[d]))
    vals = list(OrderedDict(tmp).values())
    top_dist += [np.array(vals)]

def get_doc_topic_dist(model, corpus, kwords=False):
    '''
        Transformação LDA, para cada doc só retorna tópicos com peso diferente de zero Esta função faz uma 
        transformação matricial de docs no espaço de tópicos. 

        model: o modelo LDA 
        corpus: os documentos 
        kwords: se True adiciona e retorna as chaves
    '''
    top_dist =[]
    keys = []

    for d in corpus:
        tmp = {i:0 for i in range(num_topics)}
        tmp.update(dict(model[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [np.array(vals)]
        if kwords:
            keys += [np.array(vals).argmax()]

    return np.array(top_dist), keys

top_dist, lda_keys= get_doc_topic_dist(model, corpus, True)

top_ws = []
for n in range(len(dtm)):
    inds = np.int0(np.argsort(dtm[n])[::-1][:4])
    tmp = [features[i] for i in inds]
    
    top_ws += [' '.join(tmp)]
    
df['Text_Rep'] = pd.DataFrame(top_ws)
df['clusters'] = pd.DataFrame(lda_keys)
df['clusters'].fillna(10, inplace=True)

cluster_colors = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'red', 4: 'skyblue'}

df['colors'] = df['clusters'].apply(lambda j: cluster_colors[j])

# Atribui rótulos interpretáveis
df['category_lda'] = df['clusters'].replace([0, 1, 2, 3, 4],['business','sport','tech','entertainment','politics'])

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(top_dist)

df['X_tsne'] =X_tsne[:, 0]
df['Y_tsne'] =X_tsne[:, 1]

output_notebook()

source = ColumnDataSource(dict(
    x=df['X_tsne'],
    y=df['Y_tsne'],
    color=df['colors'],
    label=df['clusters'].apply(lambda l: top_labels[l]),
    topic_key= df['clusters'],
    title= df[u'title'],
    content = df['text3'],
    legend_field=df['category_lda']
))

df = df.drop(columns=['colors','Text_Rep','X_tsne','Y_tsne'])

title = 'Visualização de tópicos'

plot_lda = figure(plot_width=1000, plot_height=600,
                     title=title, 
                     x_axis_type=None, y_axis_type=None, min_border=1)
plot_lda.scatter(x='x', y='y', legend_field='legend_field',  source=source,
                 color='color', alpha=0.6, size=5.0)

# hover tools
hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips = {"content": "Title: @title, KeyWords: @content - Topic: @topic_key "}
plot_lda.legend.location = "top_left"
export_png(plot_lda, filename="visualizacao_de_topicos.png")
# show(plot_lda)

print(f"Categorias corretas: {len(df[df.category==df.category_lda])}")

print(f"Porcentagem de categorias corretas: {round(len(df[df.category==df.category_lda])/len(df)*100,2)}%")



