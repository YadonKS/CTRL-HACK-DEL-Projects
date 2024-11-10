from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views import View
import subprocess

class RunScriptView(View):
    def post(self, request, *args, **kwargs):
        try:
            # Define your script path here
            script_path = "../..TheGreatChain/start.py"
            
            # Run the script using subprocess
            result = subprocess.run(
                ["python3", script_path],  # Use "python" or "python3" based on your environment
                capture_output=True,
                text=True
            )
            
            # Check if the script ran successfully
            if result.returncode != 0:
                return JsonResponse({"error": result.stderr}, status=500)
            
            # Return the script output
            return JsonResponse({"output": result.stdout}, status=200)
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)