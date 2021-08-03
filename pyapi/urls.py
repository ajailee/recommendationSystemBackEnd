"""pyapi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('<int:year>/<str:latest_productId>',views.userAndRating,name="user and rating"),
    path('<int:year>/<str:latest_productId>/<str:keyWord>',views.all,name="recommendation based on all"),
    path('<int:year>',views.index,name="Recommendation Based On Rating"),
    path("recommendationByUser/<str:latest_productId>",views.recommendationByUser,name="Recommendation Based On User Purchase History"),
    path("recommendationByDis/<str:keyWord>/",views.recommendationByDis,name="Recommendation Based On Product Description"),
]
