# CHANGELOG - YOLO Person Detector
*By Manuel Heredia*

All notable changes and additions to this project are documented in this file

## v1.0 - *Initial release*

### Documentation
- README.md - Added a setup guide including how to install Python and the required libraries, as well as how to run the program through the terminal
- requirements.txt - Added a list of the project's required Python libraries, currently only containing ultralytics.
- CHANGELOG.md - Added documentation of version 1.0

### Program
- Main menu - Added a recurring menu with 4 options: Run person detection, Read instructions, Show credits, and Exit. Also implemented input validation (so that the only valid inputs are 1, 2, 3, or 4, and other inputs don't crash the program) and a loop (so this menu shows up after every task until the user selects Exit, and confirms).

- Instructions and credits - Added in-program instructions on how to use the program and warnings on limitations like supported formats, model compatibility, detection accuracy and resource usage. Also added in-program credits acknowledging the libraries used and providing an email to contact.

- Person detection - Added the Ultralytics YOLO library to perform object detection using both pretrained and custom models (in either .pt or .onnx format). The 5 pretrained models are Nano, Small, Medium, Large, and Extra Large. Implemented a custom confidence threshold system so the user can select any number from 0 to 1 to use as a threshold. Configured the program to detect objects in class 0, the person class in all 5 pretrained models. Made the program be able to analyse  individual files or entire folders. By default it analyses what is within the input folder, but can use any other folder the user provides a path to, and search subfolders for supported image formats (.jpg, .jpeg, .png, .bmp, .webp). Incorporated automatic counting and reporting of detected people, and automatic saving of the annotated images to the output folder, with unique filenames to prevent them from being overwritten.



