
from collections import OrderedDict
from pathlib import Path
import os
import json
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sklearn
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse, response
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
plt.style.use("ggplot")
BASE_DIR = Path(__file__).resolve().parent.parent


def __init__(self):
    print('file loaded')


def index(request, year):
    df = pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "productrating.csv"), index_col=0)
    df = df[df.Year >= year]
    c = pd.DataFrame(df.groupby('ProductId')["Rating"].mean())
    c["Num_rat"] = df.groupby('ProductId')["Rating"].count()
    data = c.sort_values(['Num_rat', 'Rating'], ascending=False).head(20)
    res = pd.DataFrame.to_dict(data)
    response = res['Rating']
    return JsonResponse(response, content_type='text/json')

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


def recommendationByUser(request, latest_productId):
    if request.method == 'GET':
        try:
            amazon_ratings =  pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "productrating.csv"), index_col=0)
            amazon_ratings = amazon_ratings.dropna()
            popular_products = pd.DataFrame(
                amazon_ratings.groupby('ProductId')['Rating'].count())
            most_popular = popular_products.sort_values(
                'Rating', ascending=False)
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
            print(Recommend[0:9])
            if not Recommend:
                response = list(getRatingOnly().keys())
            else :
                response = Recommend
            return JsonResponse(response[:10], content_type='text/json', safe=False)

        except Exception as e:
            print(e)
            response = json.dumps([{'Error': e}])
    return HttpResponse(response, content_type='text/json',safe=False)


def recommendationByDis(request, keyWord):
    if request.method == 'GET':
        try:
            product_descriptions =  pd.read_csv(os.path.join(BASE_DIR, "myapp",
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
            true_k = 10
            model = KMeans(n_clusters=true_k, init='k-means++',
                           max_iter=100, n_init=1)
            model.fit(X1)
            print("Top terms per cluster:")
            order_centroids = model.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names()
            diskeyword = [keyWord]

            def print_cluster(i):
                print("Cluster %d:" % i),
                for ind in order_centroids[i, :10]:
                    print(' %s' % terms[ind])
                    diskeyword.append(terms[ind])
            print_cluster(prediction[0])
            response = getProductKey(diskeyword)
            newdisc=list(OrderedDict.fromkeys(response))
            return JsonResponse(newdisc[:15], content_type='text/json', safe=False)
        except Exception as e:
            print(e)
            return JsonResponse([], content_type='text/json', safe=False)


def getProductKey(mylist):
    productid = []
    dictobj = pd.read_csv(os.path.join(BASE_DIR, "myapp",
                     "wnewproductdis.csv"),
                          header=None, index_col=0, squeeze=True).to_dict()
    for item in mylist:
        for key, obj in dictobj.items():
            if item in obj:
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
    true_k = 10
    model = KMeans(n_clusters=true_k, init='k-means++',
                   max_iter=100, n_init=1)
    model.fit(X1)
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    diskeyword = [keyWord]

    def print_cluster(i):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind])
            diskeyword.append(terms[ind])
    print_cluster(prediction[0])
    discresponse=getProductKey(diskeyword)
    newindex=list(OrderedDict.fromkeys(indexresponse))
    newuser=list(OrderedDict.fromkeys(userresponse))
    newdisc=list(OrderedDict.fromkeys(discresponse))
    fullList=list(set(newindex+newuser+newdisc))
    response =list(OrderedDict.fromkeys(fullList))
    
    return JsonResponse(response[:15], content_type='text/json', safe=False)
