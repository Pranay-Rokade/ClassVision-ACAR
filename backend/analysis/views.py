from django.shortcuts import render
from django.http import HttpResponse
from .models import ClassData
import pandas as pd
from rest_framework.views import APIView
from django.http import JsonResponse
import traceback

# Create your views here.

class_name = "Class A"
POSITIVE_ACTIONS = ['Hand Raise', 'Reading Book', 'Sitting on Desk', 'Writing in Textbook']
NEGATIVE_ACTIONS = ['Eating in Classroom', 'Sleeping', 'Using Phone']
TOTAL_ACTIONS = POSITIVE_ACTIONS + NEGATIVE_ACTIONS


# 1. Total number of unique students
def total_number_of_students(df: pd.DataFrame) -> int:
    return df['Name'].nunique()

# 2. Total number of unique activities performed
def total_unique_activities(df: pd.DataFrame) -> int:
    return df['Action'].nunique()

# 3. Total number of unique positive activities performed
def total_unique_positive_activities(df: pd.DataFrame) -> int:
    return df[df['Action'].isin(POSITIVE_ACTIONS)]['Action'].nunique()

# 4. Total number of unique negative activities performed
def total_unique_negative_activities(df: pd.DataFrame) -> int:
    return df[df['Action'].isin(NEGATIVE_ACTIONS)]['Action'].nunique()

# 5. count of occurrences of each activity
def activity_counts(df):
    # Count each activity in the dataframe
    activity_counter = df['Action'].value_counts()

    # Create a dictionary for only the activities in the list
    result = {activity: int(activity_counter.get(activity, 0)) for activity in TOTAL_ACTIONS}
    return result

# 6. Percentage of each action
def percentage_of_each_action(df):
    total = len(df)
    action_counts = df['Action'].value_counts()
    action_percentages = (action_counts / total * 100).round(2).to_dict()
    return action_percentages



class activity_count(APIView):
    def get(self, request):
        try:
            # Assume class_name is defined globally
            class_data = ClassData.objects.get(class_name=class_name)
            df = pd.read_csv(class_data.csv_file.path)

            new_keys = {"Reading Book":'Reading', "Sitting on Desk":'Sitting ', "Writing in Textbook":'Writing ',"Eating in Classroom":'Eating'}
            
            response_data = activity_counts(df)
            for key, value in new_keys.items():
                response_data[value] = response_data[key]

            for i in new_keys.keys():
                del response_data[i]

            return JsonResponse(response_data)

        except ClassData.DoesNotExist:
            return HttpResponse("No class with that name.", status=404)
        except Exception as e:
            traceback.print_exc()  # Logs the full traceback in console
            return HttpResponse(f"An error occurred: {str(e)}", status=500)


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


class PositiveNegativeStats(APIView):
    def get(self, request):
        try:
            class_data = ClassData.objects.get(class_name=class_name)
            df = pd.read_csv(class_data.csv_file.path)

            pos_count = df[df['Action'].isin(POSITIVE_ACTIONS)].shape[0]
            neg_count = df[df['Action'].isin(NEGATIVE_ACTIONS)].shape[0]

            return JsonResponse({
                "positive": pos_count,
                "negative": neg_count
            })

        except ClassData.DoesNotExist:
            traceback.print_exc()
            return HttpResponse("Class not found", status=404)
        except Exception as e:
            traceback.print_exc()
            return HttpResponse(f"Error: {str(e)}", status=500)
        

class ActivitiesPerStudent(APIView):
    def get(self, request):
        
        try:
            class_data = ClassData.objects.get(class_name=class_name)
            df = pd.read_csv(class_data.csv_file.path)

            activity_counts = df['Name'].value_counts().to_dict()

            return JsonResponse(activity_counts)

        except ClassData.DoesNotExist:
            traceback.print_exc()
            return HttpResponse("Class not found", status=404)
        except Exception as e:
            traceback.print_exc()
            return HttpResponse(f"Error: {str(e)}", status=500)
        

class PercentageOfActions(APIView):
    def get(self, request):
        try:
            class_data = ClassData.objects.get(class_name=class_name)
            df = pd.read_csv(class_data.csv_file.path)

            action_percentages = percentage_of_each_action(df)

            return JsonResponse(action_percentages)

        except ClassData.DoesNotExist:
            traceback.print_exc()
            return HttpResponse("Class not found", status=404)
        except Exception as e:
            traceback.print_exc()
            return HttpResponse(f"Error: {str(e)}", status=500)