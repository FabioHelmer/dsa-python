from Boneco import Boneco
from Palavra import Palavra
import Dados


class Jogo():
    def __init__(self):
        self.boneco = Boneco()
        self.palavraOculta = Palavra()

    def revelaLetra(self, word):
        index = Palavra.getIndex(self, word)
        self.palavraOculta.toBe[self.palavraOculta.getIndex(
            self, word)] = word + " "
        txt = " "
        for i in self.palavraOculta.toBe.values():
            txt += i+" "
        print(txt)


jogo = Jogo()
jogo.revelaLetra("a")
