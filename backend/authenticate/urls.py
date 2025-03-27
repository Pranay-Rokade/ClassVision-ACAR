from django.urls import path
import authenticate.views as userviews

urlpatterns = [
    path('register', userviews.register.as_view()),
    path('verify-email',userviews.verify_email.as_view()),
    path('<str:case>/resend-otp',userviews.resend_otp.as_view()), # takes email/login in case
    path('login', userviews.LoginView.as_view()),
    path('verify-token', userviews.CheckTokenValidity.as_view()),
    path('logout', userviews.LogoutView.as_view()),
]
