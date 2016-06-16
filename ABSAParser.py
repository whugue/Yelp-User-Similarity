
"""
(II) Parse ABSA Data (XML Format) and Stack into Single Dataframe
"""

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


##Function to Parse the ABSA2014 Annotated Restaurant Review Data (XML File) and Return a Pandas Dataframe
def parse_data_2014(xml_path):
    container = []                                                                                  #Initialize Container (List) for Parse Data
    sentences = ET.parse(xml_path).getroot()                                                        #Get Sentence-Level Nodes
    
    for sentence in sentences:                                                                      #Loop Through Sentences
        sentence_id = sentence.attrib["id"]                                                         #Save ID
        sentence_text = sentence.getchildren()[0].text                                              #Save Text        
        aspects = sentence.getchildren()                                                            #Get Aspect-Level Notes
        
        for aspect in aspects:                                                                      #Loop Through Aspects
            row = {}
            
            if aspect.tag=="aspectCategories":
                opinions = aspect.getchildren()                                                     #Get Opinion-Level Notes
                for opinion in opinions:
                    category = opinion.attrib["category"]
                    
                    row = {"sentence_id": sentence_id, "sentence": sentence_text, "category": category} #Create DF Row
                    container.append(row)                                                           #Add Row to Container
                    
        if row == {}:                                                                               
            row = {"sentence_id":sentence_id, "sentence":sentence_text, "category":np.nan}              #If "aspectCategores Node does not exist, set cat=NaN
            container.append(row)                                                                   #Add row to container
            
    return pd.DataFrame(container)                                                                  #Convert Container to Pandas DF


##Function to Parse the ABSA2015 Annotated Restaurant Review Data (XML File) and Return a Pandas Dataframe
def parse_data_2015(xml_path):
    container = []                                                                                  #Initialize Container (List) for Parse Data
    reviews = ET.parse(xml_path).getroot()                                                          #Get Review-Level Nodes

    for review in reviews:                                                                          #Iterate Through Reviews
        sentences = review.getchildren()[0].getchildren()                                           #Get Sentence-Level Nodes
        
        for sentence in sentences:                                                                  #Iterate Through Sentences
            sentence_id = sentence.attrib["id"]                                                     #Save Sentence ID
            sentence_text = sentence.getchildren()[0].text                                          #Save Sentence Text
            
            try:                                                                                    #If any opinions associated with text
                opinions = sentence.getchildren()[1].getchildren()
            
                for opinion in opinions:                                                            #Iterate through Opinions    
                    category = opinion.attrib["category"]
        
                    row = {"sentence_id":sentence_id, "sentence": sentence_text, "category":category}   #Create DF Row
                    container.append(row)                                                               #Add Row to Container
                
            except IndexError: #if no opinions associated with text
                row = {"sentence_id":sentence_id, "sentence": sentence_text, "category":np.nan}         #Create DF Row
                container.append(row)                                                                   #Add Row to Container
                
    return pd.DataFrame(container)


##Function to Parse XML Data and 
def stack_data(parse_function, xml_path):
    df = parse_function(xml_path)
    return pd.concat([absa_data, df], axis=0)


##Run Functions
absa_data = pd.DataFrame()                                                                          #Intitialize Empty Container

absa_data = stack_data(parse_data_2014, "data/ABSA/2014/Restaurants_Train_v2.xml")
absa_data = stack_data(parse_data_2014, "data/ABSA/2014/restaurants_Trial.xml")
absa_data = stack_data(parse_data_2014, "data/ABSA/2014/Restaurants_Test_Data_phaseB.xml")
absa_data = stack_data(parse_data_2015, "data/ABSA/2015/ABSA-15_Restaurants_Train_Final.xml")
absa_data = stack_data(parse_data_2015, "data/ABSA/2015/ABSA15_Restaurants_Test.xml")

print absa_data.shape
print absa_data.head(5)

absa_data.to_pickle("data/ABSA/all_absa_data.pkl")                                                  #Pickle Parse ABSA Data





