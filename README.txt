TAP SOFTWARE - INSTALLATION GUIDE
=================================

What you need
-------------
- Windows 10 or 11
- Python 3.11 (the version must match the one in installer.bat, line "set PYVER")
- An internet connection for the first installation
  (not needed if this package contains a "wheels" folder)
- Bluetooth on the PC, to connect the TapStrap
- About 10 GB of free disk space

Install (once)
--------------
1. Unzip the package to a folder you can write to, for example
   C:\TapSoftware\  (do NOT use "Program Files").
2. If Python is not installed: download it from
   https://www.python.org/downloads/windows/
   and tick "Add Python to PATH" during installation.
3. Double-click "installer.bat".
   It creates a private environment in the "venv" folder and installs the
   exact library versions the app was tested with, including the TapStrap SDK.
   This can take several minutes. Wait until you see "Installation finished".

Start the app
-------------
1. Double-click "lancer.bat". A black window opens: leave it open while you work.
2. The browser opens on the participant page after a few seconds.
   - Participant page: http://localhost:5000
   - Examiner page:    http://localhost:5000/examiner
3. To stop the app, close the black window.

Where your data is stored
-------------------------
Recordings and results are written in this same folder:
  data\          raw recordings
  data_clean\    cleaned data and versions
  classifier_store\   trained models
These folders are created automatically on first use. The package contains
no dataset. Back them up before updating or moving the app.

Update to a new version
-----------------------
1. Keep the folders data, data_clean, gesture_sets and classifier_store.
2. Copy the new files over the old ones (overwrite) and run installer.bat again.

Uninstall
---------
Delete the folder. Nothing else was installed on the PC (except Python).

Troubleshooting
---------------
"Python 3.11 was not found"
  Install that exact Python version (see above), then run installer.bat again.

The installation fails while installing a library
  Check the internet connection and run installer.bat again. If it fails again,
  send the last 20 lines of the black window to the person who gave you the app.

The page does not open
  Another program may be using port 5000. Close it, or restart the PC.
  Then run lancer.bat again and open http://localhost:5000 yourself.

Windows asks to allow the app through the firewall
  Allow it on private networks. This is needed if the participant screen is
  another device.

The TapStrap is not found
  Check that Bluetooth is on, the TapStrap is charged and in range, and that no
  other program (phone app, other computer) is connected to it.
