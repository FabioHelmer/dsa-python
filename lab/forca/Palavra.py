import Dados
import random


class Palavra():
    def __init__(self):
        self.toBe = Palavra.escolhePalavra(self)

    def escolhePalavra(self):
        data = Dados.data
        palavra = data[random.randrange(0, len(data), 2)]
        return dict(enumerate("fabio"))

    def oculta(self):
        strInicio = ""
        for i in range(len(self.toBe)):
            strInicio += "__ "
        print(strInicio)
        return dict(enumerate(strInicio))

    def getIndex(self, word):
        index = None
        for i in self.toBe.values():
            if word == i:
                return index
            index += 1
