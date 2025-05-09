from django.urls import path
import analysis.views as analysisviews

urlpatterns = [
    path('kpis', analysisviews.kpis_in_dashboard.as_view()),
    # path('videoclassification', analysisviews.upload_video.as_view()),
    # path('video/<str:video_url>', userviews.receive_video_url.as_view()),
]