from istengine import app
from flask import render_template, send_file, request, flash, session, url_for, redirect, make_response
from forms import SignupForm, SigninForm
from models import db, User

import os
import gzip
import uuid
import shutil

from werkzeug import secure_filename
import analytics as acs


import copy
import cPickle

import csv

ALLOWED_EXTENSIONS = set(['txt', 'csv', 'dat'])
SANDBOX='sandbox' 
PREBUILT='prebuilt'



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
           
def createSessionDirectory(dirname):
    print dirname
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        os.makedirs(os.path.join(dirname,'results'))
    print 'created directory....'    

def deleteSessionDirectory(dirname):
    print dirname
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    print 'deleted directory....'  

def loadGzip(filename):
    """Loads a compressed object from disk
    """
    
    f = gzip.GzipFile(filename, 'rb')
    obj = cPickle.load(f)
    f.close()

    return obj


def prebuiltDict():
    result = {}
    try:
        propertyFilename='prebuiltModels.csv'
        filename=os.path.join(session['prebuilt_dir'],propertyFilename)
        reader = csv.reader(open(filename))
        for row in reader:
            key = (row[0]).strip()
            if key in result:
                continue
            result[key] = (row[1]).strip()
            
    except:
        print 'prebuilt model file not found'
    result.pop('key', None)    
    print result    
    return result
    
@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html',msg=e), 500


@app.route('/', methods=['GET', 'POST'])
def agreement():
    return render_template('about.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
  form = SigninForm()
  if 'uid' in session:
    return redirect(url_for('profile1'))
  #form.email.errors=None
  #form.password.errors=None    
  
  if request.method == "POST":
    if form.validate() == False:
      print 'coming here..post'  
      return render_template('home.html', form=form )
    else:
      try:  
          session.regenerate()
      except:
          print 'session.regenerate error'
      
      session['firstname']=form.data_firstname()
      session['lastname']=form.data_lastname()
      session['companyname']=form.data_companyname()
      session['phone']=form.data_phone()
      session['email']= form.email.data
      
      session['uid'] = str(uuid.uuid4())
      session['desc_html']=None
      
      session['sandbox'] = SANDBOX
      session['prebuilt'] = PREBUILT
      
      session['session_dir'] = session['sandbox'] +'/'+session['uid']
      session['prebuilt_dir'] = session['prebuilt'] 
      createSessionDirectory(session['session_dir'])
      session['train_filenames']=[]
      session['test_filenames']=[]

      obj=acs.DataAnalytics(session['uid'],session['session_dir'])
      session['data_object']=obj
      session['prebuilt_dict']=prebuiltDict()
      
      session.modified = True
      #return render_template('agreement.html')
      return redirect(url_for('profile1'))
         
  elif request.method == 'GET':
    return render_template('home.html', form=form)


@app.route('/about')
def about():
  return render_template('about.html')



@app.route('/signup', methods=['GET', 'POST'])
def signup():
  form = SignupForm()

  if 'uid' in session:
    return redirect(url_for('profile1'))

  if request.method == 'POST':
    if form.validate() == False:
      return render_template('signup.html', form=form)
    else:
      newuser = User(form.firstname.data, form.lastname.data, form.companyname.data, form.email.data, form.phone.data, form.password.data)
      db.session.add(newuser)
      db.session.commit()

      session['email'] = newuser.email
      session['firstname'] = newuser.firstname
      session['lastname'] = newuser.lastname
      session['companyname'] = newuser.companyname
      session['phone'] = newuser.phone
      
      
      return redirect(url_for('profile1'))

  elif request.method == 'GET':
    return render_template('signup.html', form=form)


@app.route('/signin', methods=['GET', 'POST'])
def signin():

  form = SigninForm()

  if 'uid' in session:
    print 'email found'
    
    return redirect(url_for('profile1'))
    
  if request.method == 'POST':
    if form.validate() == False:
      return render_template('signin.html', form=form)
    else:
      session['email'] = form.email.data
      return redirect(url_for('profile1'))
      

  elif request.method == 'GET':
    return render_template('signin.html', form=form)


@app.route('/signout')
def signout():
  print session
  deleteSessionDirectory(session['session_dir'])
  try:
      session.destroy()
  except:
      print 'session.destroy failed'
  
  print session 
  return redirect(url_for('agreement'))
  


@app.route('/profile1', methods=['GET', 'POST'])
def profile1():
    print session
    print 'coming to profile1'

    if 'uid' not in session:
        return redirect(url_for('home'))

    user = User.query.filter_by(email = session['email']).first()
    
    if user is None:
        return redirect(url_for('home'))
    else:
        return render_template('profile.html')



@app.route('/profile', methods=['GET', 'POST1'])
def profile():
    if 'uid' not in session:
        return redirect(url_for('home'))
    user = User.query.filter_by(email = session['email']).first()

    print session
    
    if user is None:
        return redirect(url_for('home'))
    else:
      dataDict={}  
      dataDict['email']=session['email'] 
      dataDict['firstname']=session['firstname'] 
      dataDict['lastname']=session['lastname'] 
      dataDict['companyname']=session['companyname'] 
      dataDict['phone']=session['phone'] 
      return render_template('info.html',data=dataDict)


@app.route('/tab1', methods=['GET', 'POST'])
def tab1():
  if request.method == 'GET':
    return render_template('tab1.html', )
  if request.method == 'POST':
    value = request.form.getlist('vehicle')
    with open('properties.csv', 'wb') as f1:
        for i in value:
            if(i==len(value)-1):
                f1.write('%s'%i)
                break
            f1.write('%s\n'%i)

    return redirect(url_for('home'))



@app.route('/uploadtrain')
def uploadtrain():

    html=''
    reqDict=session['data_object'].trainFileNames    
    if len(reqDict.keys()) > 0:
        table='<br><br><table border=5>\n' 
        table='<br><br><table class="table-striped table-bordered table-condensed table-hover" width="647">\n'
        table=table+'<tr><th colspan="4"><h3>Uploaded training properties</h3></th></tr>'
        table=table+'<th>response</th> <th>filename</th> <th> # compounds </th> <th> remove </th>' 
        for key in reqDict.keys():
           table=table+'<tr class="success">\n'
           table=table+'<td> %s </td> <td> %s </td>  <td> %s </td> <td> <input type="checkbox" name="remove" value="%s">  </td>\n'%(key,os.path.basename(reqDict[key]),session['data_object'].trainSize[key],key)
           table=table+'</tr>\n'  
        table=table+'</table>\n'    
        html=table    
    
    
    pre_html=''
    reqDict=session['prebuilt_dict']    
    if len(reqDict.keys()) > 0:
        for key in reqDict.keys():
            table='<tr><th></th> <th></th> <th><input type="checkbox" name="prebuilt" value="%s"><b> %s </b> </th> </tr>' % (key,reqDict[key])
            pre_html=pre_html+table+'\n'
        
    print pre_html
    
    return render_template('upload_train.html',html=html,pre_html=pre_html)



@app.route('/navigate', methods=['GET', 'POST'])
def navigate():
    
    return redirect(url_for(request.form['submit']))
    
    if request.form['submit'] == 'uploadtrain':
       return redirect(url_for('uploadtrain')) 
    elif request.form['submit'] == 'validation':
       return redirect(url_for('validation'))
    else:
       pass 
    

@app.route('/uploadtraindata', methods=['GET', 'POST'])
def uploadtraindata():
    
    if request.form['submit'] == 'next':
        return redirect(url_for('fimp'))
    if request.form['submit'] == 'prev':
        return redirect(url_for('profile1'))    
    
    if request.method == 'POST':
        removes = request.form.getlist('remove')
        values = request.form.getlist('prebuilt')        
        file1 = request.files['file1']
        
        
        #session['prebuilt_dir']
        if len(values) > 0:
            for prop in values:
                key=prop.lower()
                filename=str(key)+'/'+str(key)+'.pklz'
                temp=os.path.join(session['prebuilt_dir'],filename.lower())
                print temp                
                obj=loadGzip(temp) 
                print session['data_object']
                session['data_object'].trainFileNames[key]=obj['trainFileNames']
                session['data_object'].trainSmilesDF[key]=obj['trainSmilesDF']
                session['data_object'].trainSize[key]=obj['trainSize']
                session['data_object'].trainSuppInput[key]=obj['trainSuppInput']
                session['data_object'].trainDescDF[key]=obj['trainDescDF']
                session['data_object'].flowPipeline[key]=[os.path.join(session['prebuilt_dir'],k) for k in obj['flowPipeline']]
                session['data_object'].problemType[key]=obj['problemType']
                session['data_object'].responseUniqueVals[key]=obj['responseUniqueVals']
                session['data_object'].trainPreProcessModels[key]=os.path.join(session['prebuilt_dir'],obj['trainPreProcessModels'])
                session['data_object'].trainFeatureImportanceModels[key]=os.path.join(session['prebuilt_dir'],obj['trainFeatureImportanceModels'])
                session['data_object'].trainResponseDF[key]=obj['trainResponseDF']
                session['data_object'].fimpImages[key]=os.path.join(session['prebuilt_dir'],obj['fimpImages'])
                session['data_object'].crossValImages[key]=[os.path.join(session['prebuilt_dir'],k) for k in obj['crossValImages']]
                modDict=obj['modelsDict']                
                for k in modDict.keys():
                    modDict[k]=os.path.join(session['prebuilt_dir'],modDict[k])
                session['data_object'].modelsDict[key]=modDict#[os.path.join(session['prebuilt_dir'],k) for k in ]
                session['data_object'].crossValidationResults[key]=obj['crossValidationResults']
                session['data_object'].validationDict[key]=obj['validationDict']
                
                
                print session['data_object'].trainFileNames
                print session['data_object'].fimpImages
                print session['data_object'].crossValImages
                print session['data_object'].trainPreProcessModels
                print session['data_object'].flowPipeline
                print session['data_object'].trainPreProcessModels
                print session['data_object'].trainFeatureImportanceModels
                print session['data_object'].fimpImages
                print session['data_object'].crossValImages
                print session['data_object'].modelsDict[key]
                
                print 'done populating data' 
            message=', '.join(map(lambda x: x.lower(), values))+' uploaded sucessfully'
            flash(message,category='error')
                
        if file1 and allowed_file(file1.filename):
            filename = secure_filename(file1.filename)
            temp=os.path.join(session['session_dir'], filename)
            file1.save(temp)
            #session['train_filenames'].append(filename)
            #print session['train_filenames']
            session['data_object'].readInputFiles([temp],train=True)
            message='File '+filename+' uploaded sucessfully'
            flash(message,category='success') 
            
        if len(removes) > 0:
           for prop in removes:
               prop=str(prop)
               print 'remving '+str(prop)
               session['data_object'].trainFileNames.pop(prop,None)
               session['data_object'].trainSmilesDF.pop(prop,None)
               session['data_object'].trainSize.pop(prop,None)
               session['data_object'].trainSuppInput.pop(prop,None)
               session['data_object'].trainDescDF.pop(prop,None)
               session['data_object'].flowPipeline.pop(prop,None)
               session['data_object'].problemType.pop(prop,None)
               session['data_object'].responseUniqueVals.pop(prop,None)
               session['data_object'].trainPreProcessModels.pop(prop,None)
               session['data_object'].trainFeatureImportanceModels.pop(prop,None)
               session['data_object'].trainResponseDF.pop(prop,None)
               session['data_object'].fimpImages.pop(prop,None)
               session['data_object'].crossValImages.pop(prop,None)
               session['data_object'].modelsDict.pop(prop,None)
               session['data_object'].crossValidationResults.pop(prop,None) 
               session['data_object'].validationDict.pop(prop,None)
           message=', '.join(removes)+' removed from analysis'
           flash(message,category='success')
            
    #print session['train_filenames']    
    print session['data_object'].trainFileNames.keys()
    if len(session['data_object'].trainFileNames.keys()) > 0:
        flash('You can proceed to 2. Descriptor Importance and/or 3. Validation')
    session.modified = True
    print 'done upload_train_data' 
           
    return redirect(url_for('uploadtrain'))



@app.route('/uploadtest')
def uploadtest():
    html=''
    print session['data_object'].testFileNames
    print session['data_object'].testSmilesDF.keys()
    reqDict = session['data_object'].testFileNames    
    print reqDict
    if len(reqDict.keys()) > 0:
        table='<br><br><table class="table-striped table-bordered table-condensed table-hover" width="647">\n' 
        table=table+'<tr><th colspan="3"><h3>Uploaded testing files</h3></th></tr>'
        table=table+'<th>filename</th> <th> #compounds </th> <th> remove </th>' 
        for key in reqDict.keys():
            table=table+'<tr class="success">\n'
            table=table+'<td> %s </td> <td> %s </td> <td> <input type="checkbox" name="remove" value="%s">  </td> \n'%(os.path.basename(key),session['data_object'].testSize[key],key)
            table=table+'</tr>\n'  
        table=table+'</table>\n'    
        html=table    

    return render_template('upload_test.html',html=html)



@app.route('/uploadtestdata', methods=['GET', 'POST'])
def uploadtestdata():
    
    if request.form['submit'] == 'next':
        return redirect(url_for('predict'))

    if request.form['submit'] == 'prev':
        return redirect(url_for('validation'))
        
    if request.method == 'POST':
        file1 = request.files['file2']
        removes = request.form.getlist('remove')
        print "test remove"
        print removes
        if file1 and allowed_file(file1.filename):
            filename = secure_filename(file1.filename)
            temp=os.path.join(session['session_dir'], filename)
            file1.save(temp)
            session['test_filenames'].append(filename)
            session['data_object'].readInputFiles([temp],train=False)
        
        if len(removes) > 0:
           for prop in removes:
               prop=str(prop)
               session['data_object'].testSize.pop(prop,None)
               session['data_object'].testFileNames.pop(prop,None)
               print session['data_object'].testSmilesDF
               session['data_object'].testSmilesDF.pop(prop,None)
               session['data_object'].testDescDF.pop(prop,None)
               #session['data_object'].testProcessedDescDF.pop(prop,None)
               session['data_object'].testResponseDF.pop(prop,None)
               session['data_object'].testResDF.pop(prop,None)
               session['data_object'].ciResDF.pop(prop,None)

               
    session.modified = True    
    print 'done upload_test_data'
    if len(session['data_object'].trainFileNames.keys()) > 0 and len(session['data_object'].testFileNames.keys()) > 0:
        flash('You can proceed to 5. Predictions')

    return redirect(url_for('uploadtest'))



@app.route('/fimp',methods=['GET', 'POST'])
def fimp():
    print "route fimp"
    print "done route fimp"
    if len(session['data_object'].trainFileNames.keys()) == 0:
        #html='<h3> Please select at least one learning set </h3>' 
        flash('Please select at least one learning set')
        #return render_template('upload_train.html',html=None)
        return redirect(url_for('uploadtrain'))
    else:
        flash('You can proceed to 3. Validation after image(s) display')
        return render_template("descimp.html")


@app.route('/descimp')
def descimp():
    print "route descimp"
    session['data_object'].processDataPredict()
    if request.method == 'GET':
        print session
        img=session['data_object'].generateFeatureImportanceImage()
        print session['data_object'].fimpImages.values()
        print img
        session.modified = True
        print "done route descimp"
        return send_file(img, mimetype='image/png',cache_timeout=1)
        

@app.route('/validation',methods=['GET', 'POST'])
def validation():
    if len(session['data_object'].trainFileNames.keys()) == 0:
        #html='<h3> Please select at least one learning set </h3>'
        flash('Please select at least one learning set')
        return redirect(url_for('uploadtrain'))
        #return render_template('upload_train.html',html=None)
    else:
        flash('You can proceed to 4. Upload Compounds to Test after image(s) display')
        return render_template("validation.html")


@app.route('/valresult')
def valresult():
    print 'route cross valid'
    #session['data_object'].crossValImages
    session['data_object'].processDataPredict()
    session['data_object'].validateModels()
    session['data_object'].crossValidationAccuracy()
    session['data_object'].crossValidationPlot()
    print 'done route validation'
    session.modified = True
    
    print "route cross valid image"
    print request.method 
    if request.method == 'GET':
        print session['data_object'].crossValImages
        img=session['data_object'].generateR2Image()
        print img
        #session['data_object']=copy.deepcopy(temp)
        print session['data_object'].crossValImages
        session.modified = True
        return send_file(img, mimetype='image/png',cache_timeout=1)



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if len(session['data_object'].trainFileNames.keys()) == 0:
        #html='<h3> Please select at least one learning set </h3>' 
        flash('Please select at least one learning set')
        return render_template('upload_train.html',html=None)
    
    if len(session['data_object'].testFileNames.keys()) == 0:
        #html='<h3> Please uplaod at least one test set </h3>'
        flash('Please upload at least one test set')
        return render_template('upload_test.html',html='')
    
    return render_template("results.html")    

@app.route('/predictres', methods=['GET', 'POST'])
def predictres():    
    for output in session['data_object'].modelsDict.keys():
        if session['data_object'].modelsDict[output] == True:
            filename=str(output)+'/'+str(output)+'_model.pklz'
            temp=os.path.join(session['prebuilt_dir'],filename.lower())
            print temp                
            obj=loadGzip(temp)
            session['data_object'].flowPipeline[output]=cPickle.loads(obj['flowPipeline'])
            session['data_object'].modelsDict[output]=cPickle.loads(obj['modelsDict'])
    
    session['data_object'].processDataPredict()
    session['data_object'].trainModels()
    session['data_object'].testModels()
    
    print session['data_object'].modelsDict.keys()
    print session['data_object'].testResDF.keys()
    session['data_object'].computeCI()
    session['data_object'].generateHTML(session['session_dir'])
    html=session['data_object'].allHTML
    session.modified = True
    print html
    
    return render_template("resultsdisplay.html",html=html)




#@app.after_request
#def add_header(response):
#    response.cache_control.max_age = 1
#    return response       

