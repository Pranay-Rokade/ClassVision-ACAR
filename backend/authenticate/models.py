from django.db import models

from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from authenticate.managers import UserManager

# Create your models here.
class User(AbstractBaseUser, PermissionsMixin):
    USER_ROLES = [('admin', 'Admin'),
    ('user', 'User'),]

    username = models.CharField(max_length=255, unique=True)
    fullname = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)
    role = models.CharField(max_length=5,choices=USER_ROLES, default='user') # Determines DB Access
    is_email_verified = models.BooleanField(default=False)
    phone_number = models.CharField(max_length=15, unique=True)
    otp = models.CharField(max_length=6, null=True, blank=True)
    otp_expiry = models.DateTimeField(blank=True, null=True)

    # required only for django (DB) admin
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=False)


    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email','phone_number']

    objects = UserManager()

    def __str__(self):
        return self.fullname

    def has_perm(self, perm, obj=None):
        return self.role == 'admin'
    def has_module_perms(self, app_label):
        return self.role == 'admin'
