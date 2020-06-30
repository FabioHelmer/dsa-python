import nltk
import ConnectMongo
stopWords = nltk.corpus.stopwords.words('portuguese')
base = [('anonymous são legais', 'alegria'),
        ('apoio os anonymous', 'alegria'),
        ('anonymous desmascarando os crimes', 'alegria'),
        ('anonymous hackeiam governos', 'alegria'),
        ('anonymous hackers', 'raiva'),
        ('anonymous hackeia e publicam dados de bolsonaro', 'raiva'),
        ('grupo anonymous ataca pessoas', 'raiva'),
        ('anonymous ameaçam governos ', 'alegria'),
        ('FBI esta em busca do grupo anonymous', 'raiva'),
        ('anonymous estão apoiando os protetos vidas negras importam', 'alegria'),
        ('i love anonymous', 'alegria')]


def fazstemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p))
                       for p in palavras.split() if p not in stopWords]
        frasesstemming.append((comstemming, emocao))
    return frasesstemming


frasescomstemming = fazstemmer(base)
frases = frasescomstemming


def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras


palavras = buscapalavras(frases)


def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras


frequencia = buscafrequencia(palavras)
print(frequencia)


def busca_palavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq


palavrasunicas = busca_palavrasunicas(frequencia)


def extrai_palavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicas:
        caracteristicas['% s' % palavras] = (palavras in doc)
    return caracteristicas


basecompleta = nltk.classify.apply_features(extrai_palavras, frasescomstemming)
classificador = nltk.NaiveBayesClassifier.train(basecompleta)

for frase in ConnectMongo.base:
    frase = "eu gosto muito de você"
    testestem = []
    stemmer = nltk.stem.RSLPStemmer()
    for (palavras) in frase.split():
        comstem = [p for p in palavras.split()]
        testestem.append(str(stemmer.stem(comstem[0])))

        nova_frase = extrai_palavras(testestem)

        distribuicao = classificador.prob_classify(nova_frase)
        print('--------- FRASE --------')
        print(frase)
        print('-----------------------')
        for classe in distribuicao.samples():
            print("%s: %f" % (classe, distribuicao.prob(classe)))
        print()

print(classificador.show_most_informative_features(5))
