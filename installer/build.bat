pushd "%~dp0"

rmdir venv /s /q
rmdir build /s /q
rmdir dist /s /q
rmdir wixobj /s /q
del /s DgenerateComponents.wxs

python -m venv venv
call venv\Scripts\activate.bat

pushd ".."

set DGENERATE_FORCE_LOCKFILE_REQUIRES=1

pip install --editable .[dev] --extra-index-url https://download.pytorch.org/whl/cu118/

popd

pyinstaller dgenerate.spec --clean

call venv\Scripts\deactivate.bat

set PATH=%PATH%;C:\Program Files (x86)\WiX Toolset v3.11\bin

heat dir dist\dgenerate -o DgenerateComponents.wxs -scom -frag -srd -sreg -gg -cg DgenerateComponents -dr INSTALLFOLDER
candle Product.wix DgenerateComponents.wxs -arch x64 -out wixobj\ -ext WixUIExtension
light -b dist\dgenerate wixobj\Product.wixobj wixobj\DgenerateComponents.wixobj -out wixobj\dgenerate.msi -ext WixUIExtension

pushd wixobj
"C:\Program Files\7-Zip\7z.exe" -v1500m a dgenerate_installer.zip dgenerate.msi cab1.cab cab2.cab cab3.cab
popd

popd