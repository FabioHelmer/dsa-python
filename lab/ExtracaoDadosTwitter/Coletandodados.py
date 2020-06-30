import ConexaoTwitter as ctw
import ConnectMongo


class MyListener(ctw.StreamListener):
    def on_data(self, dados):
        try:
            tweet = ctw.json.loads(dados)
            id_str = tweet["id_str"]
            created_at = tweet["created_at"]
            text = tweet["text"]

        except KeyError:
            id_str = ""
            created_at = ""
            text = ""

        finally:
            obj = {"id_str": id_str, "created_at": created_at, "text": text}
            tweetind = ConnectMongo.col.insert_one(obj).inserted_id
            print(obj)
            return True


# criando o objt mylistener
mylistener = MyListener()

# criando o objt mystream
mystream = ctw.Stream(ctw.auth, listener=mylistener)
