# using the conda python for r studio environment. 
#to use the virtual environment within the project itself, use: https://docs.rstudio.com/tutorials/user/using-python-with-rstudio-and-reticulate/

#import packages
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from geopy.distance import distance # to measure distances https://janakiev.com/blog/gps-points-distance-python/

#store raw data in df object
df = pd.read_csv("HousingPrices-Amsterdam-August-2021.csv")
#change global options to display all columns in console
pd.set_option("display.max_columns", None)

#preview the data
df.head()
df.info()

#check for missing values
df.isna().sum()
#update df object to remove the few rows that are missing
df = df.dropna()

#create an empty column to then store distance values
df["Distance_From_Central"] = np.nan #create
#rename the unnamed column for easier comprehension
df.rename(columns={'Unnamed: 0': 'Original_Row'}, inplace = True)

#coordinates of the central station
amsterdam_central_coord = (52.3791, 4.9003)

#for every house in amsterdam, calculate distance from the central station and store this in the df["distance_from_central"] column
for i in df.index:
    d = distance(amsterdam_central_coord, (df["Lat"][i], df["Lon"][i])).km #in kilometers
    df["Distance_From_Central"][i] = d
    
#summary stats on the data
df.describe()

#######linear regression########

import statsmodels.api as sm

#fit linear regression model (https://www.statology.org/sklearn-linear-regression-summary/)
model = sm.OLS(df['Price'], df[['Area', 'Distance_From_Central', 'Room']]).fit()

#view model summary
print(model.summary())

#######End linear regression########

###Unit testing


#scatterplot of price as function of distance and room number
sb.scatterplot(x = 'Distance_From_Central',y = 'Price', hue = 'Room', data = df)
plt.show()

#scatterplot of price as function of area
plt.clf() # clears previous plot
sb.scatterplot(x = 'Area',y = 'Price', data=df)
plt.show()

#limiting to houses under 1,000,000
plt.clf() # clears previous plot
sb.scatterplot(x = 'Area',y = 'Price', data=df[df['Price'] < 1000000])
plt.show()

#rooms, with prices limited to under 1,000,000
plt.clf() # clears previous plot
sb.scatterplot(x = 'Room',y = 'Price', data=df[df['Price'] < 1000000])
plt.show()

#histogram of distribution of number of rooms
plt.clf() # clears previous plot
sb.histplot(x = 'Room', data = df)
plt.show()

#histogram of distribution of area
plt.clf() # clears previous plot
sb.histplot(x = 'Area', data = df)
plt.show()

plt.clf() # clears previous plot
sb.histplot(x = 'Price', data = df)
plt.show()


###ok now next steps
#1) figurre out how seaborn and matplotlib work with one anothr (rewatch + read the python book)
#2) conduct regression of price based on area, room, and distance from central station
#3) construct a dag to get at the causal relationships
#4) exploratory data analysis package for python - to easily do a range of visualizations
#5) convert the outcome variable using a transformation to see if it helps with the regession - or maybe just
# standardize it
#6) look up why we standardize things in the first place
#7) try another machiine learning algorithm like random forest to see if it can do a better job in prediction




