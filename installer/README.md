In order to build the installer, there are a few tools that need to be installed on Windows.

You need to install the .NET SDK:  https://dotnet.microsoft.com/en-us/download

You then need to install WiX v4

From the command line: ``dotnet tool install --global wix --version 5.0.0``


You must also install 7-Zip (for creating the multipart zip file): https://www.7-zip.org/

After which, to build the installer, run: ``python build.py``

The finished installer files will reside in: ``bin/Release``