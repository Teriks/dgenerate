pushd "%~dp0"

IF [%VIRTUAL_ENV%] == [] GOTO skip_venv_save
SAVED_VIRTUAL_ENV=%VIRTUAL_ENV%
:skip_venv_save

rmdir venv /s /q
rmdir build /s /q
rmdir dist /s /q
rmdir obj /s /q

python -m venv venv
call venv\Scripts\activate.bat

pushd ".."

set DGENERATE_FORCE_LOCKFILE_REQUIRES=1

pip install --editable .[dev] --extra-index-url https://download.pytorch.org/whl/cu118/

popd

pyinstaller dgenerate.spec --clean

call venv\Scripts\deactivate.bat

dotnet build Installer.wixproj --configuration Release

pushd obj\Release
"C:\Program Files\7-Zip\7z.exe" -v1500m a dgenerate_installer.zip dgenerate.msi cab1.cab cab2.cab cab3.cab
popd

popd

IF [%SAVED_VIRTUAL_ENV%] == [] GOTO skip_venv_restore
call %SAVED_VIRTUAL_ENV%\Scripts\activate.bat
:skip_venv_restore