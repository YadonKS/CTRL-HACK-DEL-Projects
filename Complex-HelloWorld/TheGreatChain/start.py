import subprocess
import os

print("Starting in Python...")

# Call PowerShell script
current_directory = os.path.dirname(os.path.abspath(__file__))
powershell_script = os.path.join(current_directory, "script1.ps1")
subprocess.run(["powershell", "-File", powershell_script])