from django.urls import path
from .views import RunScriptView

urlpatterns = [
    path('run-script/', RunScriptView.as_view(), name='run_script'),
]