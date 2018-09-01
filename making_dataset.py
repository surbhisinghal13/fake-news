import pandas as pd

DataFrameTG=pd.read_csv("./fakenewsapi/tempdata/Clean_TheGuardian.csv")

DataFrameTG["fakeness"] = 0  
DataFrameFake = pd.read_csv("./fakenewsapi/tempdata/fake.csv")

#print("The columns of the guardians are :",DataFrameFake.columns)

DataFrameTG = DataFrameTG.rename(columns={'bodyText' : 'body'})

DataFrameFake = DataFrameFake.rename(columns={'text':'body','title':'headline'})

# Dropping unnecesary columns
DataFrameFake.drop([ u'uuid',u'ord_in_thread',u'author',u'published', 
         u'language', u'crawled', u'site_url', u'country',
        u'thread_title', u'spam_score', u'replies_count', u'participants_count',
        u'likes', u'comments', u'shares', u'type', u'domain_rank',u'main_img_url'],inplace=True,axis=1)

DataFrameTG.drop([u'Unnamed: 0',  u'apiUrl', u'fields',u'id', 
        u'isHosted', u'sectionId', u'sectionName', u'type',
         u'webTitle', u'webUrl', u'pillarId', u'pillarName',u'webPublicationDate'],inplace=True,axis=1)
         
#print("The columns of the guardians are :",DataFrameTG.columns,"Size",len(DataFrameTG))
DataFrameFake["fakeness"] = 1

# Concta the DataFrames of fakeNews and TheGuardian
DataFrameComplete = DataFrameFake.append(DataFrameTG, ignore_index=True)
DataFrameComplete = DataFrameComplete[DataFrameComplete.body != ""]
DataFrameComplete = DataFrameComplete[DataFrameComplete.headline != ""]

#Dropping the Nan values and info
DataFrameComplete=DataFrameComplete.dropna()
DataFrameComplete=DataFrameComplete[DataFrameComplete.headline!="nan"]
DataFrameComplete=DataFrameComplete[DataFrameComplete.body!="nan"]
#print("The final csv shape is:", DataFrameComplete.shape)

DataFrameComplete=DataFrameComplete[['headline','body','fakeness']]
DataFrameComplete=DataFrameComplete.sample(frac=1).reset_index(drop=True)
DataFrameComplete.to_csv("Complete_DataSet_uncleaned.csv")


