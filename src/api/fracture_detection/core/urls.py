from django.contrib import admin
from django.urls import path
from django.contrib.auth.views import LogoutView
from core import views

urlpatterns = [
    path('', views.main_view, name='index'),  # Main page
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('result/', views.result_view, name='result'),
    path('index/', views.index, name='index'),
    path('error/', views.error_view, name='error'),
    path('api/process_image/', views.api, name='process_image_api'),  # JSON API
]
