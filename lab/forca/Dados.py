file = open("Dados.csv", encoding="UTF8")
data = {}
i = 0
for linha in file:
    data[i] = linha[:-1]
    i += 1
