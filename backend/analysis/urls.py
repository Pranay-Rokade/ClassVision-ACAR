from django.urls import path
import analysis.views as analysisviews

urlpatterns = [
    path('kpis', analysisviews.kpis_in_dashboard.as_view()),
    path('activity-count', analysisviews.activity_count.as_view()),
    path('positive-negative-stats', analysisviews.PositiveNegativeStats.as_view()),
    path('activities-per-student', analysisviews.ActivitiesPerStudent.as_view()),
    path('percentage-of-actions', analysisviews.PercentageOfActions.as_view()),
]