# imports
from pymongo import MongoClient

# criando a conexão ao MongoDB
client = MongoClient('localhost', 27017)
# criando o banco de dados
db = client.anonymousdb
# criando a col
col = db.anonymous
