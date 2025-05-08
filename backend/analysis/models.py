from django.db import models

# Create your models here.
class ClassData(models.Model):
    MODES = [('teacher', 'Teacher'), ('online', 'Online'), ('student', 'Student'), ('cheating', 'Cheating'), ('hybrid', 'Hybrid')]
    class_name = models.CharField(max_length=100)
    csv_file = models.FileField(upload_to='csv_uploads/')
    description = models.TextField(blank=True, null=True)
    class_time = models.DateTimeField(auto_now_add=True)
    mode = models.CharField(max_length=50, choices=MODES, default='student')

    def __str__(self):
        return f"{self.class_name} - {self.csv_file.name}"
