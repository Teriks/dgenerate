pushd "%~dp0"


dotnet build dgenerate.wixproj --configuration Release

pushd obj\Release
"C:\Program Files\7-Zip\7z.exe" -v1500m a dgenerate_installer.zip dgenerate.msi cab1.cab cab2.cab cab3.cab
popd

popd

IF [%SAVED_VIRTUAL_ENV%] == [] GOTO skip_venv_restore
call %SAVED_VIRTUAL_ENV%\Scripts\activate.bat
:skip_venv_restore