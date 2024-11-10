# Change to the "TheGreatChain" directory
$currentDirectory = Join-Path -Path (Get-Location) -ChildPath "Complex-HelloWorld\TheGreatChain"
Set-Location -Path $currentDirectory

# Get the full path of the Java file
$javaFile = Join-Path -Path $currentDirectory -ChildPath "script2.java"

Write-Output "In PowerShell, calling Java program..."
# Compile the Java file with the full path
javac $javaFile

# Run the compiled Java class
java -cp $currentDirectory script2