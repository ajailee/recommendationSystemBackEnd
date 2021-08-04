
from collections import OrderedDict
from pathlib import Path
import os
import json
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
BASE_DIR = Path(__file__).resolve().parent.parent


def __init__(self):
    print('file loaded')

#Recommendation By Rating module -1
def index(request, year):
    ##Reading the CSV File
    df = pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "productrating.csv"), index_col=0)
    #Filtering the data as per the user Requested Year
    df = df[df.Year >= year]
    #Grouping the product Id and Rating give to that
    c = pd.DataFrame(df.groupby('ProductId')["Rating"].mean())
    #Getting count of the most rated products 
    c["Num_rat"] = df.groupby('ProductId')["Rating"].count()
    #Getting Top 20 of the most rated Products 
    data = c.sort_values(['Num_rat', 'Rating'], ascending=False).head(20)
    #Converting to dict{productId:rating}
    res = pd.DataFrame.to_dict(data)
    #sending the data as JSON to the frontend
    response = res['Rating']
    return JsonResponse(response, content_type='text/json')


#same as previouse but sends output as dictionary
def getRatingOnly():
    df = pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "productrating.csv"), index_col=0)
    df = df[df.Year >= 2018]
    c = pd.DataFrame(df.groupby('ProductId')["Rating"].mean())
    c["Num_rat"] = df.groupby('ProductId')["Rating"].count()
    data = c.sort_values(['Num_rat', 'Rating'], ascending=False).head(20)
    res = pd.DataFrame.to_dict(data)
    response = res['Rating']
    return response

#Recommendation by User model 2
def recommendationByUser(request, latest_productId):
    if request.method == 'GET':
        try:
            #reading the csv file 
            amazon_ratings =  pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "productrating.csv"), index_col=0)
            #removing the null value row 
            amazon_ratings = amazon_ratings.dropna()
            #creating the dataFrame from the csv file
            popular_products = pd.DataFrame(
                amazon_ratings.groupby('ProductId')['Rating'].count())
            most_popular = popular_products.sort_values(
                'Rating', ascending=False)
            #taking the top 5000 products rating given by the user 
            amazon_ratings1 = amazon_ratings.head(5000)
            # transform the dataset into a form called a utility matrix. 
            #Utility Matrix is nothing but a 2D matrix where one axis belongs to the users and the other axis belongs to 
            #he items (movies in this case). 
            # So the value at (i,j) location of the matrix will be the rating that user i gave for movie j.
            ratings_utility_matrix = amazon_ratings1.pivot_table(
                values='Rating', index='UserId', columns='ProductId', fill_value=0)
            #storing the unility matrix in x
            X = ratings_utility_matrix.T
            X1 = X
            #svd -singular value decomposition (SVD)
            #creating  the object if the TruncatedSVD of th sklearn with 10 features 
            SVD = TruncatedSVD(n_components=10)
            #transforming our utility model(X) to the svd matrix
            decomposed_matrix = SVD.fit_transform(X)
            #Numpy implements a corrcoef() function that returns a matrix of 
            # correlations of x with x, x with y, y with x and y with y. 
            # x- user ,y - produxt
            correlation_matrix = np.corrcoef(decomposed_matrix)
            # product id given by the user from the front end
            i = latest_productId 
            # Storing the product names from the X (utility matrix)
            product_names = list(X.index)
            # Storing the product ID 
            product_ID = product_names.index(i)
            # find the correlation of the given product id 
            correlation_product_ID = correlation_matrix[product_ID]
            correlation_product_ID.shape
            # getting the product id's of the product which has correlation greater than 90%
            Recommend = list(X.index[correlation_product_ID > 0.90])
            # removing the user given product id and keeping the rest
            Recommend.remove(i)
            print(Recommend[0:9])
            # if list is empty sending the top rated products
            if not Recommend:
                response = list(getRatingOnly().keys())
            else :
                response = Recommend
            #sending the results to the front end as JSON
            return JsonResponse(response[:10], content_type='text/json', safe=False)

        except Exception as e:
            print(e)
            response = json.dumps([{'Error': e}])
    return HttpResponse(response, content_type='text/json',safe=False)

# recommendation using the Discription module:- 3
def recommendationByDis(request, keyWord):
    if request.method == 'GET':
        try:
            #reading the csv file
            product_descriptions =  pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "wnewproductdis.csv"), index_col=0)
            #shaping the data as arrays
            product_descriptions.shape
            #remove the null rows
            product_descriptions = product_descriptions.dropna()
            #taking the top 500 product description
            product_descriptions1 = product_descriptions.head(500)
            # used to calculate the “Term Frequency – Inverse Document”
            # Term Frequency - gives the most occured Word
            # Inverse Document - gives the least occured Word
            # The TfidfVectorizer will tokenize documents, learn the vocabulary and 
            # give inverse document frequency weightings,
            vectorizer = TfidfVectorizer(stop_words='english')
            #sending the product description to the TfidVectorizer model
            X1 = vectorizer.fit_transform(
                product_descriptions1["product_description"])
            X1
            X = X1
            # find the similar words for the given word using the k-menas ++
            kmeans = KMeans(n_clusters=10, init='k-means++')
            #seeing the model as graph internal use
            y_kmeans = kmeans.fit_predict(X)
            plt.plot(y_kmeans, ".")
            plt.show()
            # number of cluster to be created 
            true_k = 10
            #creating model with no of cluster i.e 10 initialization of centroid with k-means++
            # maximium iternation 100 and starts with 1
            model = KMeans(n_clusters=true_k, init='k-means++',
                           max_iter=100, n_init=1)
            model.fit(X1)
            #getting the ordered clusters using sort
            order_centroids = model.cluster_centers_.argsort()[:, ::-1]
            # getting the recommended words
            terms = vectorizer.get_feature_names()
            # storing the words
            Y = vectorizer.transform([keyWord])
            # geting the other predicted key words 
            prediction = model.predict(Y)
            diskeyword = [keyWord]
            # storing the pridicted key words in the list (diskeyword)
            def print_cluster(i):
                print("Cluster %d:" % i),
                for ind in order_centroids[i, :10]:
                    print(' %s' % terms[ind])
                    diskeyword.append(terms[ind])
            print_cluster(prediction[0])
            # getting the product id of the products which has the predected keywords
            response = getProductKey(diskeyword,keyWord)
            # removing the duplicate product id buy using set Data type
            newdisc=list(OrderedDict.fromkeys(response))
            # sending the data to the front end as JSON
            return JsonResponse(newdisc[:15], content_type='text/json', safe=False)
        except Exception as e:
            print(e)
            return JsonResponse([], content_type='text/json', safe=False)


def getProductKey(mylist,keyWord):
    productid = []
    print(keyWord)
    dictobj = pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "wnewproductdis.csv"),
                          header=None, index_col=0, squeeze=True).to_dict()
    for item in mylist:
        for key, obj in dictobj.items():
            obj=obj.lower()
            keyWord=keyWord.lower()
            if re.search(keyWord, obj):
                print(keyWord)
                print(obj)
                print(item)
                productid.append(key)
    return productid


def all(request, year, latest_productId, keyWord):
    df = pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "productrating.csv"), index_col=0)
    df = df[df.Year >= year]
    c = pd.DataFrame(df.groupby('ProductId')["Rating"].mean())
    c["Num_rat"] = df.groupby('ProductId')["Rating"].count()
    data = c.sort_values(['Num_rat', 'Rating'], ascending=False).head(20)
    res = pd.DataFrame.to_dict(data)
    indexresponse = res['Rating']
    # user user response

    amazon_ratings = pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "productrating.csv"), index_col=0)
    amazon_ratings = amazon_ratings.dropna()
    popular_products = pd.DataFrame(
        amazon_ratings.groupby('ProductId')['Rating'].count())
    most_popular = popular_products.sort_values('Rating', ascending=False)
    amazon_ratings1 = amazon_ratings.head(5000)
    ratings_utility_matrix = amazon_ratings1.pivot_table(
        values='Rating', index='UserId', columns='ProductId', fill_value=0)
    X = ratings_utility_matrix.T
    X1 = X
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    i = latest_productId
    product_names = list(X.index)
    product_ID = product_names.index(i)
    product_ID
    correlation_product_ID = correlation_matrix[product_ID]
    correlation_product_ID.shape
    Recommend = list(X.index[correlation_product_ID > 0.90])
    Recommend.remove(i)
   # print(Recommend[0:9])
    userresponse = Recommend[0:20]

    product_descriptions = pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "wnewproductdis.csv"), index_col=0)
    product_descriptions.shape
    product_descriptions = product_descriptions.dropna()
    product_descriptions.shape
    product_descriptions.head()
    product_descriptions1 = product_descriptions.head(500)
    product_descriptions1["product_description"].head(10)
    vectorizer = TfidfVectorizer(stop_words='english')
    X1 = vectorizer.fit_transform(
        product_descriptions1["product_description"])
    X1
    X = X1
    kmeans = KMeans(n_clusters=10, init='k-means++')
    y_kmeans = kmeans.fit_predict(X)
    true_k = 10
    model = KMeans(n_clusters=true_k, init='k-means++',
                   max_iter=100, n_init=1)
    model.fit(X1)
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    Y = vectorizer.transform([keyWord])
    prediction = model.predict(Y)
    diskeyword = [keyWord]

    def print_cluster(i):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind])
            diskeyword.append(terms[ind])
    print_cluster(prediction[0])
    discresponse=getProductKey(diskeyword,keyWord)
    newindex=list(OrderedDict.fromkeys(indexresponse))
    newuser=list(OrderedDict.fromkeys(userresponse))
    newdisc=list(OrderedDict.fromkeys(discresponse))
    fullList=list(set(newuser+newdisc+newindex))
    response =list(OrderedDict.fromkeys(fullList))
    
    return JsonResponse(response, content_type='text/json', safe=False)


def userAndRating(request, year, latest_productId):
    df = pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "productrating.csv"), index_col=0)
    df = df[df.Year >= year]
    c = pd.DataFrame(df.groupby('ProductId')["Rating"].mean())
    c["Num_rat"] = df.groupby('ProductId')["Rating"].count()
    data = c.sort_values(['Num_rat', 'Rating'], ascending=False).head(20)
    res = pd.DataFrame.to_dict(data)
    indexresponse = res['Rating']
    # user user response

    amazon_ratings = pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "productrating.csv"), index_col=0)
    amazon_ratings = amazon_ratings.dropna()
    popular_products = pd.DataFrame(
        amazon_ratings.groupby('ProductId')['Rating'].count())
    most_popular = popular_products.sort_values('Rating', ascending=False)
    amazon_ratings1 = amazon_ratings.head(5000)
    ratings_utility_matrix = amazon_ratings1.pivot_table(
        values='Rating', index='UserId', columns='ProductId', fill_value=0)
    X = ratings_utility_matrix.T
    X1 = X
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    i = latest_productId
    product_names = list(X.index)
    product_ID = product_names.index(i)
    product_ID
    correlation_product_ID = correlation_matrix[product_ID]
    correlation_product_ID.shape
    Recommend = list(X.index[correlation_product_ID > 0.90])
    Recommend.remove(i)
   # print(Recommend[0:9])
    userresponse = Recommend[0:20]
    newindex=list(OrderedDict.fromkeys(indexresponse))
    newuser=list(OrderedDict.fromkeys(userresponse))
    fullList=list(set(newuser+newindex))
    response =list(OrderedDict.fromkeys(fullList))
    return JsonResponse(response, content_type='text/json', safe=False)
