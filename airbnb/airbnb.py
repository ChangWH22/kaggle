import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data['reviews_per_month'] = data['reviews_per_month'].fillna(0)
data.drop('host_name',axis=1,inplace=True)  # host_id and host_name imply the same thing ,thus drop one of these
data.isna().sum()

data['neighbourhood_group'].unique() # array(['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']
len(data['neighbourhood'].unique()) #221
data['room_type'].unique() # array(['Private room', 'Entire home/apt', 'Shared room']
data.describe()

# data exploration
plt.figure(figsize=(20,7))
plt.subplot(1,2,1)
plt.xlabel('neighbourhood_group')
plt.ylabel('count')
plt.hist(data['neighbourhood_group'])

plt.subplot(1,2,2)
data['neighbourhood_group'].value_counts()
category = ["Manhattan", "Brooklyn", "Queens", "Bronx","Staten Island"]
count = [21661, 20104, 5666, 1091,373]
plt.title("neighbourhood_count")
plt.pie(count, labels = category, autopct = "%1.1f%%")
plt.show()

plt.figure(figsize=(20,7))
plt.subplot(1,2,1)
plt.hist(data['room_type'])
plt.xlabel('room_type')
plt.ylabel('count')

plt.subplot(1,2,2)
plt.title("room_type") # 圖的標題
room=['Entire home/apt', 'Private room', 'Shared room']
count_r = [25409,22326,1160]
plt.pie(count_r, labels = room, autopct = "%1.1f%%") # 繪製圓餅圖
plt.show() # 顯現圖形 

plt.figure(figsize=(20,7))
sns.histplot(x = data["neighbourhood_group"], hue = data["room_type"],multiple="dodge", shrink=.8)
plt.show()

plt.figure(figsize=(20,14))
sns.displot(data[data['price']<800]['price'],kde=True)

plt.figure(figsize=(20,14))
sns.displot(data[data['number_of_reviews']>0]['number_of_reviews'],kde=True)


#relation of price and neighbourhood_group or room_type
data_price = data[data['price']>0]
data_price.describe()

print(data_price['price'].describe())
print('\nquantile(0.80) : {}'.format(data_price['price'].quantile(0.80)))

plt.figure(figsize=(25,7))
plt.subplot(1,2,1)
sns.boxplot(x='neighbourhood_group', y='price', data=data_price, showfliers = False, order=["Manhattan", "Brooklyn", "Queens", "Bronx","Staten Island"])
plt.subplot(1,2,2)
sns.boxplot(x='neighbourhood_group', y='price',hue='room_type', data=data_price, order=["Manhattan", "Brooklyn", "Queens", "Bronx","Staten Island"], showfliers = False)

for i in range(5):
    print('\n{} :\n'.format(data_price['neighbourhood_group'].unique()[i]),data_price[data_price['neighbourhood_group']==data_price['neighbourhood_group'].unique()[i]]['price'].describe())
    
normal_price = data_price[data_price['price']<200 ]
high_price = data_price[data_price['price']>= 200 ]
plt.figure(figsize=(20,7))

plt.subplot(1,2,1)
sns.boxplot(x='neighbourhood_group', y='price', data=normal_price, order=["Manhattan", "Brooklyn", "Queens", "Bronx","Staten Island"], showfliers = False)

plt.subplot(1,2,2)
sns.boxplot(x='neighbourhood_group', y='price', data=high_price, order=["Manhattan", "Brooklyn", "Queens", "Bronx","Staten Island"], showfliers = False)

plt.figure(figsize=(17,10))
plt.subplot(1,2,1)
sns.violinplot(x='neighbourhood_group',y='price',data=normal_price, order=["Manhattan", "Brooklyn", "Queens", "Bronx","Staten Island"])

plt.subplot(1,2,2)
sns.violinplot(x='neighbourhood_group',y='price',data=high_price, order=["Manhattan", "Brooklyn", "Queens", "Bronx","Staten Island"])

plt.figure(figsize=(20,7))
sns.boxplot(x='neighbourhood_group', y='price',hue='room_type', data=normal_price, order=["Manhattan", "Brooklyn", "Queens", "Bronx","Staten Island"], showfliers = False

plt.figure(figsize=(20,7))
sns.boxplot(x='neighbourhood_group', y='price',hue='room_type', data=high_price, order=["Manhattan", "Brooklyn", "Queens", "Bronx","Staten Island"], showfliers = False)
       
# review, neightborhood, room type and price
data[data['reviews_per_month']==0].describe()
data[['price','number_of_reviews','reviews_per_month','availability_365','calculated_host_listings_count']].corr(method='spearman')
            
data_review = data.drop(data[(data['availability_365']==0)] .index)
data_review[['number_of_reviews','reviews_per_month']].describe()
            
data_review[['price','number_of_reviews','reviews_per_month','availability_365','calculated_host_listings_count']].corr(method='spearman')            
            
plt.figure(figsize=(20,14))
plt.subplot(2,1,1)
sns.histplot(x=data_review.neighbourhood_group,hue=data_review.room_type,multiple="dodge", shrink=.8)
plt.subplot(2,1,2)
sns.boxplot(x='neighbourhood_group', y='reviews_per_month',hue='room_type', data=data_review, showfliers = False)           
            
data_review_q1 = data_review[data_review['reviews_per_month'] < data_review['reviews_per_month'].quantile(0.25)]
data_review_q3 = data_review[data_review['reviews_per_month'] > data_review['reviews_per_month'].quantile(0.75)]            
            
            
plt.figure(figsize=(20,14))
plt.subplot(2,1,1)
sns.countplot(x=data_review_q3.neighbourhood_group, hue=data_review_q3.room_type, order=["Brooklyn","Manhattan", "Queens","Staten Island", "Bronx"],hue_order=['Private room', 'Entire home/apt', 'Shared room'])
plt.subplot(2,1,2)
sns.boxplot(x='neighbourhood_group', y='reviews_per_month',hue='room_type', data=data_review_q3, showfliers = False, order=["Brooklyn","Manhattan", "Queens","Staten Island", "Bronx"],hue_order=['Private room', 'Entire home/apt', 'Shared room'])            
            
def get_group(element):
    if element <= 70:
        return 0
    elif 70<element<100:
        return 1
    elif 100 <= element<165:
        return 2
    else:
        return 3
            
data_review = data_review.drop(data_review[data_review['price']==0].index)
data_review['price_type']=list(map(get_group, data_review['price'].values))
data_review = data_review[data_review['reviews_per_month'] > data_review['reviews_per_month'].quantile(0.75)]            
            
plt.figure(figsize=(20,21))
plt.subplot(4,1,1)
sns.histplot(x=data_review.neighbourhood_group,hue=data_review.price_type,multiple="dodge", shrink=.8)

plt.subplot(4,1,2)
sns.boxplot(x='neighbourhood_group', y='reviews_per_month',hue='price_type', data=data_review, showfliers = False)

plt.subplot(4,1,3)
sns.histplot(x=data_review.room_type,hue=data_review.price_type,multiple="dodge", shrink=.8)

plt.subplot(4,1,4)
sns.boxplot(x='room_type', y='reviews_per_month',hue='price_type', data=data_review, showfliers = False)           

#             
plt.figure(figsize=(20,12))
sns.scatterplot(x='longitude',y='latitude',hue='neighbourhood_group', data=data_review_q1)            
            
import cv2
plt.figure(figsize=(16,12))
image = cv2.imread('../input/file-image/New_York_City_.png')
plt.imshow(image,extent=[-74.258, -73.7, 40.49,40.92])
sns.scatterplot(x='longitude',y='latitude',hue='neighbourhood_group', data=data_review_q3)
plt.show()            
            
            
plt.figure(figsize=(16,12))
image = cv2.imread('../input/file-image/New_York_City_.png')
plt.imshow(image,extent=[-74.25, -73.69, 40.49,40.92])
plt.scatter(x='longitude',y='latitude',s=5,c='price',data=data_review_q3,cmap='hsv')
cbar = plt.colorbar()
cbar.set_label('price')            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

