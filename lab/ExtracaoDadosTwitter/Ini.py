import Coletandodados
import ConnectMongo

# criando uma lista de pavras chaves para a query de busca do twitter
search = ['anonymous', 'Anonymous']

# coletando os tweets
# iniciano o filtro e gravando os tweets no MongoDB
Coletandodados.mystream.filter(track=search)
