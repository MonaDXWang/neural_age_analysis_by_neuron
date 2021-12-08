import pandas as pd 
    
# declare a dictionary
record = { 
  
 'Name' : ['Ankit', 'Swapnil', 'Aishwarya', 
          'Priyanka', 'Shivangi', 'Shaurya' ],
    
 'Age' : [22, 20, 21, 19, 18, 22], 
    
 'Stream' : ['Math', 'Commerce', 'Science', 
            'Math', 'Math', 'Science'], 
    
 'Percentage' : [90, 90, 96, 75, 70, 80] } 
    
# create a dataframe 
dataframe = pd.DataFrame(record,
                         columns = ['Name', 'Age', 
                                    'Stream', 'Percentage']) 
# show the Dataframe
#print("Given Dataframe :\n", dataframe)

# selecting rows based on condition 
r1 = dataframe['Percentage'] > 70
rslt_df = dataframe[dataframe['Percentage'] > 70] 

print(dataframe.columns)
#print('\nResult dataframe :\n', rslt_df)