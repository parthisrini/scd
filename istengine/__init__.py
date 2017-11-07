from flask import Flask
import matplotlib
from simplekv.memory import DictStore
from flask.ext.kvsession import KVSessionExtension
import getpass

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

#store = DictStore()
app = Flask(__name__)
app.debug = True
store = DictStore()
KVSessionExtension(store, app)

#KVSessionExtension(store, app)
#app.config['SQLALCHEMY_DATABASE_URI'] =  os.environ['DATABASE_URL']

usr=getpass.getuser()

print '==========='
print usr
print '==========='

if usr=='pbalapra':
    app.config['SQLALCHEMY_DATABASE_URI'] =  'mysql+pymysql://newuser:password@localhost/db'
else:    
    app.config['SQLALCHEMY_DATABASE_URI'] =  'mysql://root:intern_app123@localhost/db'

app.secret_key = 'development key'


from models import db
db.init_app(app)
with app.test_request_context():
	db.create_all()

import istengine.routes


#app.session_interface = SqliteSessionInterface(path)
#app.run()