from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import sys

import numpy as np
from scipy import stats

import pandas as pd

from sklearn import ensemble
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib.colors as colors


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import StringIO
from PIL import Image
import copy
import pickle
import os
import gzip
import shutil
import cPickle

pd.set_option('max_colwidth', 250)

class DataAnalytics:
    def __init__(self,uid,path):

        self.nfold=5
        self.uid=uid
        self.path=path

        self.trainFileNames={}
        self.testFileNames={}

        self.trainSmilesDF={}
        self.testSmilesDF={}

        self.trainSize={}
        self.testSize={}

        self.trainSuppInput={}

        self.trainDescDF={}
        self.testDescDF={}

        self.flowPipeline={}
        self.problemType={}
        self.responseUniqueVals={}

        #self.trainProcessedDescDF={}
        #self.testProcessedDescDF={}

        self.trainPreProcessModels={}
        #self.trainPreProcessDescDF={}

        self.trainFeatureImportanceModels={}
        #self.trainFeatureImportanceDescDF={}

        #self.trainFeatureReductionModels={}
        #self.trainFeatureSelectionDescDF={}


        self.trainResponseDF={}
        self.testResponseDF={}

        self.crossValidationResults={}

        self.fimpImages={}
        self.crossValImages={}

        self.modelsDict={}

        self.validationDict={}

        self.testResDF={}

        self.ciResDF={}
        self.allHTML=None



        ntrees=1000
        self.algoOptionsReg = {
           'br':ensemble.BaggingRegressor(n_estimators=ntrees,n_jobs=-1),
           'etr':ensemble.ExtraTreesRegressor(n_estimators=ntrees,n_jobs=-1),
           'gbr':ensemble.GradientBoostingRegressor(n_estimators=ntrees),
           'rfr':ensemble.RandomForestRegressor(n_estimators=ntrees,n_jobs=-1),
           'abr':ensemble.AdaBoostRegressor(n_estimators=ntrees),
           }

        self.algoOptionsCls = {
           'br':ensemble.BaggingClassifier(n_estimators=ntrees,n_jobs=-1),
           'etr':ensemble.ExtraTreesClassifier(n_estimators=ntrees,n_jobs=-1),
           'gbr':ensemble.GradientBoostingClassifier(n_estimators=ntrees),
           'rfr':ensemble.RandomForestClassifier(n_estimators=ntrees,n_jobs=-1),
           'abr':ensemble.AdaBoostClassifier(n_estimators=ntrees),
           }


    def processColNames(self,colNames):
        colNames=[x.lower().strip(' ') for x in colNames]
        colNames=[x.replace(' ','_') for x in colNames]
        return colNames

    def makeDirectory(self,output):
        dirName=os.path.join(self.path,output)
        if os.path.exists(dirName):
            shutil.rmtree(dirName)
        os.makedirs(dirName)


    def removeInvalidMols(self,df):
        smiles=df['smiles'].values.tolist()
        inValidMolIndex=[]
        nms=[x[0] for x in Descriptors._descList]
        print nms
        descDF=pd.DataFrame()
        descDF['Descriptors']=nms
        #descDF.to_csv('../prebuilt/descriptors.csv',index=False)
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
        #print calc
        for i, molStr in enumerate(smiles):
            m = Chem.MolFromSmiles(smiles[i])
            if m is None:
                inValidMolIndex.append(i)
            else:
                descrs=calc.CalcDescriptors(m)
                if not np.isfinite(np.sum(descrs)):
                    inValidMolIndex.append(i)

        assert len(inValidMolIndex) < len(smiles)
        return inValidMolIndex





    def readInputFiles(self,fileNames,train=False):
        try:
            print fileNames
            for fname in fileNames:
                print 'reading data file:'+str(fname)
                df=pd.read_table(str(fname), sep=',',header=0)
                df.columns=self.processColNames(df.columns)
                print df['smiles'].values
                inValidMolIndex=self.removeInvalidMols(df)
                print inValidMolIndex
                df.drop(df.index[[inValidMolIndex]],inplace=True)
                inValidMolIndex=self.removeInvalidMols(df)
                print inValidMolIndex

                if train==False:
                    self.testSmilesDF[fname]=df
                    self.testFileNames[fname]=True
                    self.testSize[fname]=df.shape[0]
                else:
                    suppInput=[]
                    allInput=['smiles'] # default
                    suppInput=filter(lambda x: x.startswith('inp_'),df.columns)
                    allInput.extend(suppInput)
                    allOutput=set(df.columns).difference(set(allInput))
                    allOutput=list(allOutput)
                    assert list(allOutput) > 0

                    print allOutput
                    print allInput
                    for output in allOutput:
                        temp=copy.deepcopy(allInput)
                        self.trainSuppInput[output]=suppInput
                        temp.append(output)
                        print temp
                        self.trainSmilesDF[output]=df[temp]
                        self.trainFileNames[output]=fname
                        self.trainSize[output]=df.shape[0]
                        self.makeDirectory(output)
        except:
            print "error in opening the data file:", sys.exc_info()[0]
            raise


    def descriptorGeneration(self,smiles):
        print 'Generating Descriptors...'
        try:
            ms=[Chem.MolFromSmiles(x) for x in smiles]
            nms=[x[0] for x in Descriptors._descList]
            print nms
            # a descriptor calculator makes it easy to generate sets of descriptors:
            calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
            # generate the descriptors (can take a while):
            descrs = [calc.CalcDescriptors(x) for x in ms]
            descrs = np.asarray(descrs)
            descrsDF = pd.DataFrame(descrs)
            descriptorNames=[x for x in calc.GetDescriptorNames()]
            descrsDF.columns=descriptorNames
            #descrsDF.dropna(0,inplace=True)
            return descrsDF
        except:
            print "error in opening the data file:", sys.exc_info()[0]
            raise

    def generateDescriptorsAndResponse(self):
        print 'coming to generateDescriptorsAndResponse'
        print self.trainSmilesDF.keys()
        print self.trainDescDF.keys()
        try:
            if len(self.trainSmilesDF.keys())> 0:
                for output in self.trainSmilesDF.keys():
                    if output not in self.trainDescDF.keys():
                       df=self.trainSmilesDF[output]
                       smiles=df['smiles'].values.tolist()
                       df.drop('smiles', 1,inplace=True)
                       trainDescDF=self.descriptorGeneration(smiles)
                       #print df.shape
                       print trainDescDF.shape
                       trainDescDF.dropna(0,inplace=True)
                       print trainDescDF.shape
                       for key in self.trainSuppInput[output]:
                          trainDescDF[key]=df[key].values.tolist()
                          df.drop(key, 1,inplace=True)
                       assert df.shape[1] == 1
                       self.trainDescDF[output]=trainDescDF
                       self.trainResponseDF[output] = df

                       assert len(set(self.trainSuppInput[output]).difference(set(trainDescDF.columns.tolist()))) == 0
                    else:
                       print '%s already found'%output

            if len(self.testSmilesDF.keys())> 0:
                for key in self.testSmilesDF.keys():
                    if key not in self.testDescDF.keys():
                       df=self.testSmilesDF[key]
                       #print df
                       smiles=df['smiles'].values.tolist()
                       #print df
                       #print smiles
                       self.testDescDF[key]=self.descriptorGeneration(smiles)


                       suppInput=filter(lambda x: x.startswith('inp_'),df.columns)
                       for supp in suppInput:
                           self.testDescDF[key][supp]=df[supp].values.tolist()
                           df.drop(supp, 1,inplace=True)
                           print self.testDescDF[key]
                       #trainDescDF.dropna(0,inplace=True)
                       if df.shape[1] > 1:
                           self.testResponseDF[key] = df.drop('smiles', 1)
                           print self.testResponseDF[key]
                           #sys.exit(0)
                       else:
                           self.testResponseDF[key] = None
                    else:
                       print '%s already found'%key

        except:
            print "error in generateDescriptorsAndResponse:", sys.exc_info()[0]
            raise


    def identifyProblemType(self):
        print 'coming to identifyProblemType'
        print self.trainResponseDF.keys()
        try:
            if len(self.trainResponseDF.keys())> 0:
                for output in self.trainResponseDF.keys():
                    if output not in self.problemType.keys():

                        Y=self.trainResponseDF[output]
                        y=Y[output].values#.tolist()
                        uniqueVal=np.unique(y)
                        self.responseUniqueVals[output]=uniqueVal
                        if len(uniqueVal) > 10:
                            self.problemType[output]='reg'
                        else:
                            self.problemType[output]='cls'

        except:
            print "error in identifyProblemType:", sys.exc_info()[0]
            raise


    def preProcessData(self,df):
        #print df
        model = preprocessing.StandardScaler().fit(df)
        return model

    def transformWithModel(self,modelname,df,keep=False):
        print modelname
        model=loadGzipLocal(modelname,self.path)
        resDF = model.transform(df)
        resDF = pd.DataFrame(resDF)

        if keep:
            resDF.columns=df.columns

        del model
        return resDF

    def updateFlowPipeline(self,output,model):
        pipeline=[]
        if output in self.flowPipeline:
           pipeline=self.flowPipeline[output]
        pipeline.append(model)
        self.flowPipeline[output] = pipeline

    def preProcessTrainData(self):
        print 'coming to preProcessTrainData'
        try:
            if len(self.trainDescDF.keys())> 0:
                for output in self.trainDescDF.keys():
                    if output not in self.trainPreProcessModels.keys():
                       df=self.trainDescDF[output]
                       model=copy.deepcopy(self.preProcessData(df))
                       filename=os.path.join(output,'%s_preProcess.pklz'%(output))
                       print filename
                       saveGzip(model,filename,self.path)
                       del model
                       self.trainPreProcessModels[output]=filename
                       self.updateFlowPipeline(output,filename)
                       print output
                       print self.trainPreProcessModels[output]
                    else:
                       print '%s already found'% output
        except:
            print "error in preProcessTrainData:", sys.exc_info()[0]
            raise



    def featureImportance(self,X,Y,key):
        print 'coming to featureImportance'

        y=Y[key].values
        if self.problemType=='cls':
            forest = ensemble.ExtraTreesClassifier(n_estimators=1000,n_jobs=-1)
        else:
            forest = ensemble.ExtraTreesRegressor(n_estimators=1000,n_jobs=-1)
        forest.fit(X, y)
        feature_importance = forest.feature_importances_
        #print feature_importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        print feature_importance

        sorted_idx = np.argsort(feature_importance)[-30:]
        pos = np.arange(sorted_idx.shape[0]) + .5
        sortedDataInputNames=map(lambda x: X.columns[x],sorted_idx)
        labels = sortedDataInputNames


        pngFileName=os.path.join(self.path,key,'fimp_'+key+'.png')
        print pngFileName
        fig = plt.figure()
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, labels)
        plt.xlabel('Relative Importance')
        plt.title('response='+key+'; training set size='+str(self.trainSize[key]))
        fig.savefig(pngFileName, bbox_inches='tight')
        plt.close(fig)

        if 'prebuilt' in pngFileName:
            self.fimpImages[key]=os.path.join(key,'fimp_'+key+'.png')
        else:
            self.fimpImages[key]=pngFileName


        return forest


    def featureImportanceTrainData(self):
        print 'coming to featureImportanceTrainData'
        print self.trainSmilesDF.keys()
        try:
            if len(self.trainDescDF.keys())> 0:
                for output in self.trainDescDF.keys():
                    if output not in self.trainFeatureImportanceModels:
                        if output in self.trainPreProcessModels:
                           df=self.trainDescDF[output]
                           print df.shape
                           model=self.trainPreProcessModels[output]
                           X=self.transformWithModel(model,df,True)
                           Y=self.trainResponseDF[output]
                           print X.shape
                           print Y.shape
                           model=copy.deepcopy(self.featureImportance(X,Y,output))
                           filename=os.path.join(output,'%s_featureImportance.pklz'%(output))
                           saveGzip(model,filename,self.path)
                           del model
                           self.trainFeatureImportanceModels[output]=filename
                           self.updateFlowPipeline(output,filename)
                        else:
                           print '%s did not get preprocessed '%output
                    else:
                        print '%s feature importance is already processed'%output
        except:
            print "error in featureImportanceTrainData:", sys.exc_info()[0]
            raise


    def generateFeatureImportanceImage(self):
        print 'generateFeatureImportanceImage'
        print self.fimpImages
        img= StringIO.StringIO()
        images = map(Image.open, self.fimpImages.values())
        w = max(i.size[0] for i in images)
        mh = sum(i.size[1] for i in images)
        result = Image.new("RGBA", (w, mh))
        x = 0
        for i in images:
            result.paste(i, (0, x))
            x += i.size[1]
        result.save(img, format='png')
        img.seek(0)
        return img


    def fitModels(self,trainData,response,output):
        fittedModels ={}
        algoOptions=self.algoOptionsReg

        if self.problemType[output] == 'cls':
            algoOptions=self.algoOptionsCls
            print self.problemType[output]
            print algoOptions

        for algo in algoOptions.keys():
            model = algoOptions[algo]
            model.fit(trainData, response)
            filename=os.path.join(output,'%s_model_%s.pklz'%(output,algo))
            saveGzip(copy.deepcopy(model),filename,self.path)
            del model
            fittedModels[algo]=filename

        return fittedModels


    def crossValidation(self,output):
        Y=self.trainResponseDF[output]
        y=Y[output].values#.tolist()
        uniqueVal=np.unique(y)
        if len(uniqueVal) > 10:
            algoOptions=self.algoOptionsReg
        else:
            algoOptions=self.algoOptionsCls

        #apply transformation
        X=self.trainDescDF[output]

        print 'original training data shape'
        print X.shape

        print 'applying preprocesing'
        for model in self.flowPipeline[output]:
            X=self.transformWithModel(model,X)
            print X.shape

        skf = KFold(len(y), n_folds=self.nfold, shuffle=True, random_state=1)

        algoOptions=self.algoOptionsReg

        if self.problemType[output] == 'cls':
            algoOptions=self.algoOptionsCls
            print self.problemType[output]
            print algoOptions

        resDF=pd.DataFrame()
        for i, (train, test) in enumerate(skf):
            print 'train size:'+str(len(train))
            print 'test size:'+str(len(test))
            currDF=pd.DataFrame()
            obsr=y[test]
            for key in algoOptions.keys():
                model=algoOptions[key]
                model.fit(X.loc[train], y[train])
                pred=model.predict(X.loc[test])
                currDF[key+'.pred']=pred
                print pred
            currDF['obsr']=obsr
            currDF['fold']=i
            resDF=resDF.append(currDF)
        resDF.reset_index(drop=True,inplace=True)

        return resDF


    def validateModels(self):
        print 'coming to model validation'
        print self.trainDescDF.keys()
        for output in self.trainDescDF.keys():
            print self.crossValidationResults
            if output in self.crossValidationResults:
                print str(output)+' already found'
            else:
                print '+++++++++++++++++'
                print output
                self.crossValidationResults[output]=self.crossValidation(output)
                print self.crossValidationResults[output]
                print '+++++++++++++++++'

    def plotScatter(self,x,y,title,fileName):
        assert len(x)==len(y)

        fig = plt.figure()
        plt.scatter(x,y)
        #plist=[1,5,10]
        #color=['g','b','r']
        #labels=['1%', '5%', '10%']
        #for p,q,l in zip(plist,color,labels):
            #upper=map(lambda y: y+(y*p/100),x)
            #lower=map(lambda y: y-(y*p/100),x)
            #plt.plot([min(x),max(x)], [min(upper),max(upper)], '-',linewidth=1.0,c=q,label=l)
            #plt.plot([min(x),max(x)], [min(lower),max(lower)], '-',linewidth=1.0,c=q)
        #plt.legend(loc=2)
        plt.plot(x, x, '-',linewidth=3.0,c='g')
        plt.xlabel("Observed values")
        plt.ylabel("Predicted values")
        plt.title(title)
        plt.savefig(fileName, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    def plotAccuracy(self,x,y,classes,title,fileName):
        assert len(x)==len(y)
        print x
        print y
        print classes
        cm = confusion_matrix(x, y,classes)
        cm_per=100.0*cm/float(cm.sum())

        #corrMatrix=pd.DataFrame(cm)
        #colNames = corrMatrix.columns.values.tolist()
        outputNames = classes

        zeromat=np.zeros((len(cm),len(cm)))
        np.fill_diagonal(zeromat, 1)
        zeromat=pd.DataFrame(zeromat)

        cmap = colors.ListedColormap(['darkred', 'darkgreen'])
        bounds=[0,0.5,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        img=ax.imshow(zeromat,interpolation='nearest',origin='lower',cmap=cmap, norm=norm)
        width = len(cm)
        height = len(cm[0])
        for x in xrange(width):
            for y in xrange(height):
                val='%d \n (%1.1f%%)'%(cm[x][y],cm_per[x][y])
                ax.annotate(str(val), xy=(y, x), horizontalalignment='center', verticalalignment='center',color='white')
        plt.xticks(range(len(outputNames)),outputNames)
        plt.yticks(range(len(outputNames)),outputNames)
        #ax.grid(True)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(title)
        plt.savefig(fileName, bbox_inches='tight')
        plt.clf()
        fig.set_size_inches([10,10])
        plt.close(fig)

        #sys.exit(0)

    def computeR2(self,data):
        data1=data.copy()
        folds = data1['fold'].unique().tolist()
        print folds
        allResDf={}

        cvDF=data1
        obsr=cvDF['obsr'].values
        cvDF.drop(['obsr','fold'],axis=1,inplace=True)
        print cvDF
        algos=cvDF.columns.tolist()
        meanPred=cvDF.mean(axis=1)
        print meanPred
        resDF={}
        resDF['mean.pred']=r2_score(obsr, meanPred)

        for algo in algos:
            pred=cvDF[algo].values
            r2=r2_score(obsr, pred)
            resDF[algo]=r2
        print resDF
        allResDf=resDF

        print allResDf
        return allResDf

    def computeR21(self,data):
        data1=data.copy()
        folds = data1['fold'].unique().tolist()
        print folds
        allResDf={}

        for fold in folds:
            cvDF=data1[data1.fold==fold]
            obsr=cvDF['obsr'].values
            cvDF.drop(['obsr','fold'],axis=1,inplace=True)
            print cvDF
            algos=cvDF.columns.tolist()
            meanPred=cvDF.mean(axis=1)
            print meanPred
            resDF={}
            resDF['mean.pred']=r2_score(obsr, meanPred)

            for algo in algos:
                pred=cvDF[algo].values
                r2=r2_score(obsr, pred)
                resDF[algo]=r2
            print resDF
            allResDf[fold]=resDF

        print allResDf
        return allResDf


    def computeAccuracy(self,data):
        data1=data.copy()
        folds = data1['fold'].unique().tolist()
        print folds
        allResDf={}

        cvDF=data1
        obsr=cvDF['obsr'].values
        cvDF.drop(['obsr','fold'],axis=1,inplace=True)
        print cvDF

        algos=cvDF.columns.tolist()
        result=[]
        for i in range(cvDF.shape[0]):
            currRow=cvDF.iloc[i]
            val = pd.value_counts(currRow)
            res=val.index.tolist()[0]
            result.append(res)
        meanPred=result
        print meanPred
        resDF={}
        resDF['mean.pred']=accuracy_score(obsr, meanPred)

        for algo in algos:
            pred=cvDF[algo].values
            acc=accuracy_score(obsr, pred)
            resDF[algo]=acc
        print resDF
        allResDf=resDF

        print 'allResDf:'
        print allResDf
        return allResDf



    def computeAccuracy1(self,data):
        data1=data.copy()
        folds = data1['fold'].unique().tolist()
        print folds
        allResDf={}

        for fold in folds:
            cvDF=data1[data1.fold==fold]
            obsr=cvDF['obsr'].values
            cvDF.drop(['obsr','fold'],axis=1,inplace=True)
            print cvDF

            algos=cvDF.columns.tolist()

            result=[]
            for i in range(cvDF.shape[0]):
                currRow=cvDF.iloc[i]
                val = pd.value_counts(currRow)
                res=val.index.tolist()[0]
                result.append(res)
            meanPred=result
            print meanPred
            resDF={}
            resDF['mean.pred']=accuracy_score(obsr, meanPred)
            for algo in algos:
                pred=cvDF[algo].values
                acc=accuracy_score(obsr, pred)
                resDF[algo]=acc
            print resDF
            allResDf[fold]=resDF

        print 'allResDf:'
        print allResDf
        return allResDf


    def crossValidationAccuracy(self):
        print 'coming to crossValidationAccuracy'
        print self.trainDescDF.keys()
        for output in self.crossValidationResults.keys():
            if output not in self.validationDict:
                df=self.crossValidationResults[output]
                if self.problemType[output] == 'cls':
                    self.validationDict[output]=self.computeAccuracy(df)
                else:
                    self.validationDict[output]=self.computeR2(df)
                print self.validationDict[output]



    def crossValidationPlot(self):
        print 'coming to crossValidationAccuracy'
        print self.trainDescDF.keys()

        for output in self.crossValidationResults.keys():
            if output not in self.crossValImages:
                data=self.crossValidationResults[output]
                print data
                nfolds = len(data['fold'].unique().tolist())
                cvDF=data.copy()
                r2=self.validationDict[output]['mean.pred']
                obsr=cvDF['obsr'].values
                obsrUniqueVals=cvDF['obsr'].unique().tolist()
                cvDF.drop(['obsr','fold'],axis=1,inplace=True)

                if self.problemType[output] == 'cls':
                    result=[]
                    allProbs=[]
                    for i in range(cvDF.shape[0]):
                        currRow=cvDF.iloc[i]
                        predClasses=currRow.values.tolist()
                        ff=len(predClasses)
                        probs=[]
                        for k in obsrUniqueVals:
                            ff1=len(filter(lambda x: str(x)==str(k),predClasses))
                            prob=float(ff1)/float(ff)
                            probs.append(prob)
                        allProbs.append(probs)
                        resIndex=probs.index(max(probs))
                        res=obsrUniqueVals[resIndex]
                        result.append(res)
                    pred=result
                    allProbs=pd.DataFrame(allProbs)
                    allProbs.columns = map(lambda x: x, obsrUniqueVals)
                    print allProbs
                    print pd.value_counts(pred)
                    title='response=%s; training set size=%d; \n nfolds=%d; accuracy=%1.3f'%(output,self.trainSize[output],nfolds,r2*100)
                    pngFileName=os.path.join(self.path,output,'acc_'+output+'_'+str(nfolds)+'.png')
                    print pngFileName
                    pngFileName1=os.path.join(output,'acc_'+output+'_'+str(nfolds)+'.png')
                    self.plotAccuracy(obsr,pred,allProbs.columns.tolist(),title,pngFileName)
                else:
                    pred=cvDF.mean(axis=1)
                    title='response=%s; training set size=%d; \n nfolds=%d; R2=%1.3f'%(output,+self.trainSize[output],nfolds,r2)
                    pngFileName=os.path.join(self.path,output,'r2_'+output+'_'+str(nfolds)+'.png')
                    pngFileName1=os.path.join(output,'r2_'+output+'_'+str(nfolds)+'.png')
                    self.plotScatter(obsr, pred,title,pngFileName)

                if 'prebuilt' not in pngFileName:
                    pngFileName1=pngFileName

                if output in self.crossValImages:
                    self.crossValImages[output].append(pngFileName1)
                else:
                    self.crossValImages[output]=[pngFileName1]

                print self.crossValImages[output]




    def crossValidationPlot1(self):
        print 'coming to crossValidationAccuracy'
        print self.trainDescDF.keys()
        for output in self.crossValidationResults.keys():
            if output not in self.crossValImages:
                data=self.crossValidationResults[output]
                print data
                folds = data['fold'].unique().tolist()
                for fold in folds:
                    df1=data[data.fold==fold]
                    cvDF=df1.copy()
                    r2=self.validationDict[output][fold]['mean.pred']
                    obsr=cvDF['obsr'].values
                    obsrUniqueVals=cvDF['obsr'].unique().tolist()
                    cvDF.drop(['obsr','fold'],axis=1,inplace=True)

                    if self.problemType[output] == 'cls':
                        result=[]
                        allProbs=[]
                        for i in range(cvDF.shape[0]):
                            currRow=cvDF.iloc[i]
                            predClasses=currRow.values.tolist()
                            ff=len(predClasses)
                            probs=[]
                            for k in obsrUniqueVals:
                                ff1=len(filter(lambda x: str(x)==str(k),predClasses))
                                prob=float(ff1)/float(ff)
                                probs.append(prob)
                            allProbs.append(probs)
                            resIndex=probs.index(max(probs))
                            res=obsrUniqueVals[resIndex]
                            result.append(res)
                        pred=result
                        allProbs=pd.DataFrame(allProbs)
                        allProbs.columns = map(lambda x: x, obsrUniqueVals)
                        print allProbs
                        print pd.value_counts(pred)
                        #title='response=%s; fold=%d; accuracy=%1.3f'%(output,fold+1,r2*100)
                        title='response=%s; training set size=%d; accuracy=%1.3f'%(output,self.trainSize[output],r2*100)
                        #pngFileName=self.path+'/'+'acc_'+output+'_'+str(int(fold)+1)+'.png'
                        pngFileName=os.path.join(self.path,output,'acc_'+output+'_'+str(int(fold)+1)+'.png')
                        pngFileName1=os.path.join(output,'acc_'+output+'_'+str(int(fold)+1)+'.png')
                        self.plotAccuracy(obsr,pred,allProbs.columns.tolist(),title,pngFileName)
                    else:
                        pred=cvDF.mean(axis=1)
                        #title='response=%s; fold=%d; R2=%1.3f'%(output,int(fold)+1,r2)
                        title='response=%s; training set size=%d; R2=%1.3f'%(output,+self.trainSize[output],r2)
                        #pngFileName=self.path+'/'+'r2_'+output+'_'+str(int(fold)+1)+'.png'
                        pngFileName=os.path.join(self.path,output,'r2_'+output+'_'+str(int(fold)+1)+'.png')
                        pngFileName1=os.path.join(output,'r2_'+output+'_'+str(int(fold)+1)+'.png')
                        self.plotScatter(obsr, pred,title,pngFileName)

                    if 'prebuilt' not in pngFileName:
                        pngFileName1=pngFileName

                    if output in self.crossValImages:
                        self.crossValImages[output].append(pngFileName1)
                    else:
                        self.crossValImages[output]=[pngFileName1]
                    print self.crossValImages[output]

    def generateR2Image(self):
        print 'inside generateR2Image'
        print self.crossValImages
        print self.crossValImages.values()
        imageList=[]
        for key in sorted(self.crossValImages.keys()):
            values=self.crossValImages[key]
            for val in values:
                if '_' in val:
                    imageList.append(val)
        print imageList
        img= StringIO.StringIO()
        images = map(Image.open, imageList)
        w = max(i.size[0] for i in images)
        mh = sum(i.size[1] for i in images)
        result = Image.new("RGBA", (w, mh))
        x = 0
        for i in images:
            result.paste(i, (0, x))
            x += i.size[1]
        result.save(img, format='png')
        img.seek(0)
        print 'done generateR2Image'
        return img


    def trainModels(self):
        print 'coming to model fit'
        print self.trainDescDF.keys()
        for output in self.trainDescDF.keys():
            if output not in self.modelsDict:
                Y=self.trainResponseDF[output]
                y=Y[output].values
                trainDF=self.trainDescDF[output]

                X=trainDF
                print 'applying preprocesing'
                print self.flowPipeline[output]
                for modelfilename in self.flowPipeline[output]:
                    print 'loading '+modelfilename+' for preprocessing!'
                    X=self.transformWithModel(modelfilename,X)
                    print X.shape

                df=X
                print '======='
                print output
                print df.shape
                print '======='
                self.modelsDict[output]=copy.deepcopy(self.fitModels(df,y,output))

    def testModels(self):
        print 'coming to model predict'
        if len(self.testDescDF.keys())> 0:
            for output in self.trainResponseDF.keys(): #for each training file
                print output
                fittedModels=self.modelsDict[output]
                for testKey in self.testDescDF.keys(): #for each test file
                    resDF=pd.DataFrame()
                    testDF=self.testDescDF[testKey]
                    colNames=self.trainDescDF[output].columns.tolist()
                    print colNames
                    X=testDF.filter(items=colNames)
                    print testDF.columns.tolist()
                    print testDF.columns.tolist()
                    for modelfilename in self.flowPipeline[output]:
                        print 'loading '+modelfilename+' for preprocessing!'
                        print X.shape
                        X=self.transformWithModel(modelfilename,X)
                        print X.shape

                    df=X
                    print '======='
                    print output
                    print df.shape
                    print '======='

                    for algo in fittedModels.keys():
                        modelfilename = fittedModels[algo]
                        print 'loading '+modelfilename+' for prediction!'
                        model=loadGzipLocal(modelfilename,self.path)
                        pred = model.predict(df)
                        resDF[algo] = pred
                        del model
                    self.testResDF[testKey+'$'+output]=resDF

        print self.testResDF

    def computeCI(self):
        print 'coming to CI calculation'
        if len(self.testDescDF.keys())> 0:
            for testKey in sorted(self.testDescDF.keys()):
                libSmiles=self.testSmilesDF[testKey].values.tolist()
                #print self.testSmilesDF[testKey].tolist()
                #print libSmiles
                resDF=pd.DataFrame()
                resDF['row.id']=range(1,len(libSmiles)+1)
                resDF['smiles']=libSmiles
                for trainKey in self.modelsDict.keys():
                    #Y=self.trainResponseDF[trainKey]
                    for output in [trainKey]:
                        predsDF=self.testResDF[testKey+'$'+output]
                        print 'predsDF:'
                        print predsDF
                        if self.problemType[output] == 'cls':
                            obsrUniqueVals=self.responseUniqueVals[output]
                            allProbs=[]
                            for i in range(predsDF.shape[0]):
                                currRow=predsDF.iloc[i]
                                predClasses=currRow.values.tolist()
                                ff=len(predClasses)
                                probs=[]
                                for k in obsrUniqueVals:
                                    ff1=len(filter(lambda x: str(x)==str(k),predClasses))
                                    prob=100.0*(float(ff1)/float(ff))
                                    probs.append(prob)
                                allProbs.append(probs)
                            resDF1 = pd.DataFrame(allProbs)
                            resDF1.columns = map(lambda x: output+':'+str(x)+'(%)', obsrUniqueVals)
                            resDF=pd.concat([resDF, resDF1], axis=1)
                            print resDF
                        else:
                            resDF[output+'.ci.lower']= predsDF.quantile(q=0.05,axis=1)
                            resDF[output+'.mean']=predsDF.mean(axis=1)
                            resDF[output+'.ci.upper']=predsDF.quantile(q=0.95,axis=1)

                csvFileName=testKey.replace('.csv','_pred.csv')
                resDF.to_csv(csvFileName,index=False)
                self.ciResDF[testKey]=resDF

    def generateHTML(self,path):
        print 'coming to html calculation'
        allHTML='<h1>Predictions</h1> <br><br>\n'
        if len(self.testDescDF.keys())> 0:
            for testKey in self.testDescDF.keys():
                resDF=self.ciResDF[testKey]
                html=resDF.to_html(index=False,float_format=lambda x: '%2.3f' % x,classes="table-striped table-bordered table-condensed table-hover",justify="center")
                header = '<h2>%s</h2> <br> \n'% testKey.replace(path,'').replace('/','')
                html = header + html + '<br> \n'
                allHTML=allHTML+html
        self.allHTML=allHTML
        print self.allHTML



    def processDataPredict(self):
        self.generateDescriptorsAndResponse()
        self.identifyProblemType()
        self.preProcessTrainData()
        self.featureImportanceTrainData()

    def saveModel(self):
        for output in self.trainFileNames.keys():
            reqDF={}
            reqDF['trainFileNames']=self.trainFileNames[output]
            reqDF['trainSmilesDF']=self.trainSmilesDF[output]
            reqDF['trainSize']=self.trainSize[output]
            reqDF['trainSuppInput']=self.trainSuppInput[output]
            reqDF['trainDescDF']=self.trainDescDF[output]
            reqDF['flowPipeline']=self.flowPipeline[output]
            reqDF['problemType']=self.problemType[output]
            reqDF['responseUniqueVals']=self.responseUniqueVals[output]
            reqDF['trainPreProcessModels']=self.trainPreProcessModels[output]
            reqDF['trainFeatureImportanceModels']=self.trainFeatureImportanceModels[output]
            reqDF['trainResponseDF']=self.trainResponseDF[output]
            reqDF['fimpImages']=self.fimpImages[output]
            reqDF['crossValImages']=self.crossValImages[output]
            reqDF['modelsDict']=self.modelsDict[output]
            reqDF['crossValidationResults']=True
            reqDF['validationDict']=True
            print 'creating gzipped pickle for '+output
            fname=os.path.join(output,output+'.pklz')
            saveGzip(reqDF,fname,self.path)

def saveGzip(obj, filename, path, protocol = -1):
    """Saves a compressed object to disk
    """
    if 'prebuilt' not in filename:
        resFileName=os.path.join(path,filename)
    else:
        resFileName=filename

    f = gzip.GzipFile(resFileName, 'wb')
    cPickle.dump(obj, f, protocol = cPickle.HIGHEST_PROTOCOL)
    f.close()


def loadGzipLocal(filename, path):
    """Loads a compressed object from disk
    """
    if 'prebuilt' not in filename:
        resFileName=os.path.join(path,filename)
    else:
        resFileName=filename

    f = gzip.GzipFile(resFileName, 'rb')
    obj = cPickle.load(f)
    f.close()

    return obj

if __name__ == '__main__':

    trainFileNamesListAll=['../prebuilt/A_Caco2_I.csv',\
                        '../prebuilt/A_HIA_I.csv',\
                        '../prebuilt/T_hERG_I.csv',\
                        '../prebuilt/R_A_Caco2_I.csv',\
                        '../prebuilt/LC50_mouse.csv',\
                        '../prebuilt/h2o_sol.csv'
                        ]
    trainFileNamesListAll=['../prebuilt/A_Caco2_I.csv',\
                        '../prebuilt/T_hERG_I.csv',\
                        '../prebuilt/h2o_sol.csv'
                        ]

                        #'../prebuilt/R_A_Caco2_I.csv',\
                        #'../prebuilt/LC50_mouse.csv'
                        #]
    #trainFileNamesListAll=['../prebuilt/cdt1.csv']#,'../prebuilt/h2o_sol.csv']

    for fname in trainFileNamesListAll:
        trainFileNamesList=[fname]
        testFileNames=['../data/test1.csv']

        da=DataAnalytics('100','../prebuilt')
        da.readInputFiles(trainFileNamesList,True)
        da.readInputFiles(testFileNames,False)
        da.processDataPredict()

        try:
            if True:
                da.validateModels()
                da.crossValidationAccuracy()
                da.crossValidationPlot()
                da.trainModels()
                da.saveModel()
        except:
            print "Unexpected error:", sys.exc_info()[0]

        if True:
            da.testModels()
            da.computeCI()
            da.generateHTML('../results')
