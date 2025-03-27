from django.shortcuts import render

# Create your views here.
import os
import re
import random
import datetime
from django.utils import timezone
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.core.mail import send_mail
from django.contrib.auth.hashers import make_password
from django.conf import settings
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from users.models import User
from django.contrib.auth import authenticate, login
from rest_framework.authtoken.models import Token
from users.decorators import role_required
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.exceptions import NotFound
from dotenv import load_dotenv

load_dotenv()


class AdminOnlyView(APIView):
    @role_required(['admin'])
    def get(self, request):
        return Response({'message': 'Welcome, Admin!'})
    

# OTP requirements for email verification 
OTP_DURATION = 5         # minutes
# body text
body_email_verification_otp = """
Hello {}, Your OTP for email verification on ClassVision is {}.
This OTP is valid for {} minutes. Do not share it with anyone else.
"""
#subject headers
subject_verify_email = "ClassVision - Email Verification"

# Send OTP to the user's email
def send_otp(username, email, subject, body):
    # OTP Generation
    otp = random.randint(10000, 99999)

    # Send OTP to email
    send_mail(
        subject,
        body.format(username , otp, OTP_DURATION),
        settings.DEFAULT_FROM_EMAIL,
        [email],
        fail_silently=False,
    )

    return otp



class register(APIView):
    def post(self, request):
        # Validation
        fullname = request.data.get('fullname')
        phone_number = request.data.get('phone_number')
        email = request.data.get('email')
        password = request.data.get('password')
        confirm_password = request.data.get('confirm_password')
        role = "user"
        otp_expiry = timezone.now() + datetime.timedelta(minutes=OTP_DURATION)

        if not all([fullname, email, phone_number, password, confirm_password]):
            return Response({'error': 'All fields are required.'}, status=400)

        if password != confirm_password:
            return Response({'error': 'Passwords do not match.'}, status=400)

        try:
            validate_email(email)
        except ValidationError:
            return Response({'error': 'Invalid email format.'}, status=400)

        if User.objects.filter(email=email).exists() or User.objects.filter(phone_number=phone_number).exists():
            return Response({'error': 'Email or phone number is already registered.'}, status=409)
        

        # Hash the password
        hashed_password = make_password(password)

        otp = send_otp(fullname, email, subject_verify_email, body_email_verification_otp)
        # Create a new user instance but do not activate it yet (needs OTP verification)

        user = User.objects.create(
            fullname=fullname,
            phone_number=phone_number,
            email=email,
            password=hashed_password,
            role=role,
            isemailverified=False,  # Set to False until OTP is verified
            otp=otp,
            otp_expiry=otp_expiry,
        )


        user.save()

        return Response({'message': 'User registered successfully. Please verify your Email address.'}, status=201)

    def get(self, request):
        return Response({'error': 'Invalid request method.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

class verify_email(APIView):
    def post(self, request):
        otp = request.data.get('otp')
        email = request.data.get('email')

        try:
            # Retrieve the user by email
            user = User.objects.get(email=email)

            # Check if the OTP matches
            if user.otp == otp:
                # check if the OTP has expired
                if user.otp_expiry and user.otp_expiry < timezone.now():
                    return Response({'error': 'OTP has expired.'}, status=400)

                # Activate the user account upon successful OTP verification
                user.isemailverified = True
                user.otp = None  # Clear the OTP once verified
                user.save()

                return Response({'success':'Email is verified.'}, status=200)
            else:
                return Response({'message':'OTP is incorrect'},status=status.HTTP_400_BAD_REQUEST)

        except User.DoesNotExist:
            return Response({'error': 'User with this email does not exist.'}, status=404)

    def get(self, request):
        return Response({'error': 'Invalid request method.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)
    


class resend_otp(APIView):
    def post(self, request, case):
        # sends otp again
        email = request.data.get('email')

        # # k-v mapping for email/login case
        # usecase = {'email':[subject_verify_email, body_email_verification_otp], 'login':[subject_login, body_login_otp]}

        try:
            user = User.objects.get(email=email)

            user.otp = send_otp(user.username, email, usecase[case][0], usecase[case][1])
            user.otp_expiry = timezone.now() + datetime.timedelta(minutes=OTP_DURATION)
            user.save()

        except User.DoesNotExist:
            return Response({'error': 'User with this email address does not exist.'}, status=404)

        return Response({'message':'OTP resent.'},status=200)

    def get(self, request):
        return Response({'error': 'Invalid request method.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)




class CheckTokenValidity(APIView):
    def post(self, request):
        email = request.data.get('email')
        token = request.data.get('token')

        if not email or not token:
            return Response("Email and token are required.", status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response("User with the provided email does not exist.", status=status.HTTP_404_NOT_FOUND)

        try:
            user_token = Token.objects.get(user=user)
        except Token.DoesNotExist:
            return Response("No token associated with this user.", status=status.HTTP_401_UNAUTHORIZED)

        if token == user_token.key:
            return Response("Token validated. User is Authenticated.", status=status.HTTP_200_OK)

        return Response("Invalid token.", status=status.HTTP_401_UNAUTHORIZED)
    
    def get(self, request):
        return Response({'error': 'Invalid request method.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# Login API-> allows the user to login once email verification via OTP is successful

class LoginView(APIView):
    def post(self, request):
        email = request.data.get('email')  # Get the email
        password = request.data.get('password')  # Get the password

        if not email or not password:
            return Response({'error': 'Email and password are required.'}, status=400)

        # Authenticate the user using email and password
        try:
            # Retrieve the user by email
            from users.models import User  # Adjust import path if needed
            user = User.objects.get(email=email)

            # Use the username for authentication
            user = authenticate(username=user.username, password=password)

            if user and user.is_active:
                # Generate a token for the authenticated user
                token, created = Token.objects.get_or_create(user=user)
                user.otp = send_otp(user.username, email, subject_login, body_login_otp)
                user.otp_expiry = timezone.now() + datetime.timedelta(minutes=OTP_DURATION)
                user.save()

                user_roles = user.user_roles
                return Response({'token': token.key, 'message': 'Login successful. Login OTP sent.', 'username':user.username,'phone_number':user.phone_number, 'role':user.role, 'user_roles':user_roles}, status=200)

            return Response({'error': 'Invalid credentials or email not verified.'}, status=401)

        except User.DoesNotExist:
            return Response({'error': 'User with this email does not exist.'}, status=404)

    def get(self, request):
        return Response({'error': 'Invalid request method.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated

    def post(self, request):
        try:
            # Get the token associated with the authenticated user
            user = request.user
            token = Token.objects.get(user=user)

            # delete their reports from file directory once the user logs out
            pattern = re.compile(rf"^{re.escape(user.username+'_')}.*")
    
            directory_path = os.path.join(settings.BASE_DIR,'reports')

            if os.path.exists(directory_path):
                for filename in os.listdir(directory_path):
                    if pattern.match(filename):
                        file_path = os.path.join(settings.BASE_DIR, 'reports', filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"Deleted: {user.username}'s data from file directory.")
                        else:
                            print(f"Skipped (not a file): {file_path}")

            if token:
                token.delete()
                return Response({'message': 'Successfully logged out and token destroyed.'}, status=200)

            raise AuthenticationFailed('Authentication credentials were not provided.')

        except Token.DoesNotExist:
            raise AuthenticationFailed('No token found or invalid token.')

    def get(self, request):
        return Response({'error': 'Invalid request method. Use POST to log out.'}, status=405)



class AddUserRoles(APIView):
    # Admin-only access, to be implemented

    def post(self, request):
        try:
            # JSON Input
            email = request.data.get('email')
            user = User.objects.get(email=email)

            # Parse JSON to get roles to be added
            roles_to_add = request.data.get('roles', [])

            for role in roles_to_add:
                if role not in user.user_roles:
                    user.user_roles.append(role)

            user.save()

            return Response({"message": "Roles added successfully", "roles": user.user_roles}, status=status.HTTP_200_OK)

        except User.DoesNotExist:
            raise NotFound(detail="User not found")
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
