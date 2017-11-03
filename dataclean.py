
# coding: utf-8

# In[111]:


import pandas as pd       
from bs4 import BeautifulSoup
import numpy as np


# In[112]:


project_data = pd.read_json("educational_dataset_40k_latest_oct_25_2017.json")
#print(project_data.shape)
#this shows how many data entries (rows) and parameters (columns) there are
print(project_data.columns.values)
#project_data[['slug','blurb','description','goal','backers_count','state']]
#listing out some of the important columns, (this will show a table of only these parameters)


# In[113]:


#pd.DataFrame(project_data)
#this prints out the JSON data in a massive table


# In[115]:


runs = project_data.shape[0] - 300
project = []
for p in range(0,runs):
    temp = {}
    temp['slug'] = project_data['slug'][p]
    temp['description'] = project_data['description'][39999-p]
    temp['blurb'] = project_data['blurb'][p]
    temp['goal'] = project_data['goal'][p]
    temp['state'] = project_data['state'][p]
    temp['id'] = project_data['id'][p]
    project.append(temp)
    if( (p+1)%1000 == 0 ):
        print("Added project %d of %d\n" % ( p+1, runs ))


# In[116]:


import re
import nltk
import ssl
from nltk.corpus import stopwords # Import the stop word list
print(stopwords.words("english"))


# In[117]:


from bs4 import BeautifulSoup

pject = project
for l in range(len(project)):
    pject[l]['description'] = BeautifulSoup(str(project[l]['description'])).get_text()
# Use regular expressions to do a find-and-replace
    project[l]['description'] = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      pject[l]['description'] )
    project[l]['description'] = project[l]['description'].lower().split()     
    stops = set(stopwords.words("english"))                  
    project[l]['description'] = [w for w in project[l]['description'] if not w in stops]  
    if( (l+1)%500 == 0 ):
        print("scrubbed project %d of %d\n" % ( l+1, runs ))


# In[118]:


project[1000]


# In[124]:


final_project = []

for p in range(runs):
    Q = {}
    Q['state'] = project[p]['state']
    Q['description'] = project[p]['description']
    final_project.append(Q)


# In[125]:


import json
json_str = json.dumps(final_project)
# Writing JSON data
with open('final_project.json', 'w') as f:
     json.dump(final_project, f)

