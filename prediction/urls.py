from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^sign/', views.SignsView.as_view()),
    url(r'^dog/', views.DogView.as_view()),
    url(r'^emotion/', views.PredictionsView.as_view()),
]