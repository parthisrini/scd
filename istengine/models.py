from flask.ext.sqlalchemy import SQLAlchemy
from werkzeug import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
  __tablename__ = 'users'
  uid = db.Column(db.Integer, primary_key = True)
  firstname = db.Column(db.String(100))
  lastname = db.Column(db.String(100))
  companyname = db.Column(db.String(100))
  email = db.Column(db.String(120), unique=True)
  phone = db.Column(db.Integer)
  pwdhash = db.Column(db.String(54))
  
  def __init__(self, firstname, lastname, companyname, email, phone, password):
    self.firstname = firstname.title()
    self.lastname = lastname.title()
    self.companyname = companyname.title()
    self.email = email.lower()
    self.phone = phone
    self.set_password(password)
    
  def set_password(self, password):
    self.pwdhash = generate_password_hash(password)
  
  def check_password(self, password):
    return check_password_hash(self.pwdhash, password)

  def get_lastname(self):
    return self.lastname
  def get_firstname(self):
    return self.firstname
  def get_companyname(self):
    return self.companyname
  def get_phone(self):
    return self.phone
