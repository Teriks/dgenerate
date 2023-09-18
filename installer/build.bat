
set PATH=%PATH%;C:\Program Files (x86)\WiX Toolset v3.11\bin
rmdir build /s /q
rmdir dist /s /q
rmdir wixobj /s /q
del /s DgenerateComponents.wxs

pyinstaller dgenerate.spec --clean
heat dir dist\dgenerate -o DgenerateComponents.wxs -scom -frag -srd -sreg -gg -cg DgenerateComponents -dr INSTALLFOLDER
candle Product.wix DgenerateComponents.wxs -arch x64 -out wixobj\ -ext WixUIExtension
light -b dist\dgenerate wixobj\Product.wixobj wixobj\DgenerateComponents.wixobj -out wixobj\dgenerate.msi -ext WixUIExtension

: pushd wixobj
: "C:\Program Files\7-Zip\7z.exe" -v1500m a dgenerate_installer.zip dgenerate.msi cab1.cab cab2.cab cab3.cab
: popd