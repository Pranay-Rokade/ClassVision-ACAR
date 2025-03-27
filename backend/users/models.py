from django.db import models


# Create your models here.
class User(models.Model):
    USER_ROLES = [('admin','Admin'), ('user','Faculty')]

    fullname = models.CharField(max_length=255, unique=True)
    phone_number = models.CharField(max_length=15, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)
    role = models.CharField(max_length=5,choices=USER_ROLES, default='user') # if user then faculty else if admin then HOD or higher authority
    isemailverified = models.BooleanField(default=False)
    otp = models.CharField(max_length=5, null=True, blank=True)
    otp_expiry = models.DateTimeField(blank=True, null=True)


 
