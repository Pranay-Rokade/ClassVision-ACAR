from django.contrib.auth.models import BaseUserManager

class UserManager(BaseUserManager):
    def create_user(self, username, email, phone_number, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(username=username,email=email, phone_number=phone_number, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, phone_number, password=None, **extra_fields):
        email = self.normalize_email(email)
        user = self.create_user(
            username=username,email=email, phone_number=phone_number, **extra_fields
        )
        user.set_password(password)
        user.role = 'admin'
        # required only for django admin
        user.is_staff = True
        user.is_active = True
        
        user.save(using=self._db)
        return user

    def get_by_natural_key(self, username):
        return self.get(**{self.model.USERNAME_FIELD: username})