class Boneco():
    def __init__(self):
        self.atributos = ["cabeça", "braço E", "braço D",
                          "tronco", "perna E", "Perna D"]
        self.vidas = len(self.atributos)

    def diminirVida(self):
        self.vidas -= 1
