#from flask import Flask
#from simplekv.memory import DictStore
#from flask.ext.kvsession import KVSessionExtension

from istengine import app
#from simplekv.memory import DictStore
#from flask.ext.kvsession import KVSessionExtension

if __name__ == "__main__":
    #app.debug = True
    #store = DictStore()
    #KVSessionExtension(store, app)
    app.run()
