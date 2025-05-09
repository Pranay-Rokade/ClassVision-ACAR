from django.shortcuts import render
from django.http import HttpResponse
from .models import ClassData
import pandas as pd
from rest_framework.views import APIView
from django.http import JsonResponse
import traceback

# Create your views here.

class_name = "Class A"

import pandas as pd

# 1. Total number of unique students
def total_number_of_students(df: pd.DataFrame) -> int:
    return df['Name'].nunique()

# 2. Total number of unique activities performed
def total_unique_activities(df: pd.DataFrame) -> int:
    return df['Action'].nunique()

# 3. Total number of unique positive activities performed
def total_unique_positive_activities(df: pd.DataFrame) -> int:
    positive_actions = ['Hand Raise', 'Reading Book', 'Sitting on Desk', 'Writing in Textbook']
    return df[df['Action'].isin(positive_actions)]['Action'].nunique()

# 4. Total number of unique negative activities performed
def total_unique_negative_activities(df: pd.DataFrame) -> int:
    negative_actions = ['Eating in Classroom', 'Sleeping', 'Using Phone']
    return df[df['Action'].isin(negative_actions)]['Action'].nunique()



class kpis_in_dashboard(APIView):
    def get(self, request):
        try:
            # Assume class_name is defined globally
            class_data = ClassData.objects.get(class_name=class_name)
            df = pd.read_csv(class_data.csv_file.path)

            response_data = {
                'total_students': total_number_of_students(df),
                'total_unique_activities': total_unique_activities(df),
                'positive_activities': total_unique_positive_activities(df),
                'negative_activities': total_unique_negative_activities(df),
            }

            return JsonResponse(response_data)

        except ClassData.DoesNotExist:
            return HttpResponse("No class with that name.", status=404)
        except Exception as e:
            traceback.print_exc()  # Logs the full traceback in console
            return HttpResponse(f"An error occurred: {str(e)}", status=500)
