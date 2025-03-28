from django.urls import path
import classification.views as classificationviews

urlpatterns = [
    path('livevideo', classificationviews.receive_video_url.as_view()),
    # path('video/<str:video_url>', userviews.receive_video_url.as_view()),
]