pushd "%~dp0"

IF [%VIRTUAL_ENV%] == [] GOTO skip_venv_save
set SAVED_VIRTUAL_ENV=%VIRTUAL_ENV%
:skip_venv_save

rmdir venv /s /q
rmdir build /s /q
rmdir dist /s /q
rmdir obj /s /q
rmdir bin /s /q

python -m venv venv
call venv\Scripts\activate.bat

pushd ".."
set DGENERATE_FORCE_LOCKFILE_REQUIRES=1
pip install .[win-installer] --extra-index-url https://download.pytorch.org/whl/cu121/
popd

pyinstaller dgenerate.spec --clean

call venv\Scripts\deactivate.bat

pushd dist
"C:\Program Files\7-Zip\7z.exe" -v1500m a dgenerate_portable.zip dgenerate
popd

dotnet build dgenerate.wixproj --configuration Release

pushd bin\Release
set cab_files=
for /f "tokens=*" %%F in ('dir /b /a:-d "*.cab"') do call set cab_files=%%cab_files%% "%%F"

"C:\Program Files\7-Zip\7z.exe" -v1500m a dgenerate_installer.zip dgenerate.msi %cab_files%
popd

for /r "dist" %%x in (*.zip.*) do move "%%x" "bin\Release"

popd

IF [%SAVED_VIRTUAL_ENV%] == [] GOTO skip_venv_restore
call %SAVED_VIRTUAL_ENV%\Scripts\activate.bat
:skip_venv_restore