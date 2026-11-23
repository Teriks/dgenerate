# macOS: duplicate AVFoundation classes from cv2 and PyAV

Seen on a rented Apple Silicon Mac while running the console from a source
install at `/Users/m1/dgenerate` (macOS 27.2.1, Darwin 27.2.0, arm64,
Python 3.14 venv).

```
objc[17184]: Class AVFFrameReceiver is implemented in both
  .../site-packages/cv2/.dylibs/libavdevice.63.1.102.dylib
  and
  .../site-packages/av/.dylibs/libavdevice.62.3.103.dylib.
This may cause spurious casting failures and mysterious crashes.
One of the duplicates must be removed or renamed.

objc[17184]: Class AVFAudioReceiver is implemented in both
  (same two dylibs).
```

## What is happening

`opencv-python` and `av` (PyAV) each ship a private FFmpeg. Both copies of
`libavdevice` contain the AVFoundation capture device, and both register the
Objective-C classes `AVFFrameReceiver` and `AVFAudioReceiver`.

dyld maps each dylib when that package is imported and registers those class
names in the process-wide Objective-C runtime. The runtime allows one class
per name. The second registration is rejected and this message is printed.
The two copies here are not the same build: OpenCV has libavdevice 63.1.102
and PyAV has 62.3.103.

Those classes are only the camera and microphone receivers. Decoding a file
with PyAV, and image work through OpenCV, do not message them. The console
still runs. The crash the message describes happens if both libraries open an
AVFoundation capture device in the same process and one of them messages the
other library's class.

## What does not fix it

Importing one package before the other does not help. Both dylibs still load,
and the classes are registered inside `dlopen`, before the import returns.

Nothing can rename or unregister a class that is already registered from a
mapped dylib. There is no Python API for that. Setting
`DYLD_INSERT_LIBRARIES` from inside the process is too late; dyld only honors
it at process start. The file on disk can still be rewritten before either
library is loaded.

`opencv-python-headless` does not help. The headless wheel still bundles
`libavdevice`. It only drops the highgui backends.

Replacing one `libavdevice` file with a symlink to the other is unsafe on
this install. The sonames differ (63.1.102 vs 62.3.103). That shortcut is
only plausible when both wheels ship the identical FFmpeg build, and even
then it is an unsupported edit that pip overwrites on the next install.

## What removes it

`dgenerate` rewrites OpenCV's `libavdevice` before either library is imported.
`AVFFrameReceiver` and `AVFAudioReceiver` become `CV2FrameReceiver` and
`CV2AudioReceiver`. The new names are the same length, so the rest of the
library stays aligned, and the dylib is ad-hoc re-signed. PyAV keeps the
original names. dgenerate decodes video with PyAV and never opens a camera
or microphone through OpenCV.

pip does not run code after it installs a wheel, so this runs at startup
(`dgenerate/cv2avnames.py`), including `pip install dgenerate`. The network
installer and `make_dev_env.py` apply the same edit at install time. Once the
names are changed, later startups leave the file alone. If the dylib cannot
be written or re-signed, it is left unchanged and the warning remains.

A shared FFmpeg, as in the conda-forge builds of `av` and `opencv`, would
also remove it. The pip wheels each ship a private copy.
