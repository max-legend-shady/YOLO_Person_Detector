from ultralytics import YOLO
from pathlib import Path

project_folder = Path(__file__).resolve().parent
output_folder = project_folder / "output"

def introduction():
    print("\n--- WELCOME TO THE YOLO PERSON DETECTION PROGRAM ---")
    print("\nCreated by Manuel Heredia")


def show_menu():
    # Show main menu options to the user and prompt for a choice
    print("\n--- MENU ---")
    print("1. Detect people in images")
    print("2. Read the instructions")
    print("3. Show credits")
    print("4. Exit")
    try:
        menu_choice = int(input("Select an option (1, 2, 3, or 4): "))
    except ValueError:
        print("You have not entered a number.")
        return None
    return menu_choice


def choose_model():
    while True:    
        # Prompt the user to select a model or enter a custom path
        print("\nSelect the model you would like to use for detection.")
        print("Leave blank to use the default model, Nano (yolo26n.pt).")
        print("Available models: \n1. Nano (yolo26n.pt)\n2. Small (yolo26s.pt)\n3. Medium (yolo26m.pt)\n4. Large (yolo26l.pt)\n5. Extra Large (yolo26x.pt)")
        print("6. Custom (use a model file stored on your system)")
        print("7. Back to main menu")
        model_choice = input("What model would you like to use? ").strip().lower()
        if model_choice in ("yolo26n.pt", "nano", "1", "n", ""):
            print("Model selected: Nano")
            model_path = "yolo26n.pt"
        elif model_choice in ("yolo26s.pt", "small", "2", "s"):
            print("Model selected: Small")
            model_path = "yolo26s.pt"
        elif model_choice in ("yolo26m.pt", "medium", "3", "m"):
            print("Model selected: Medium")
            model_path = "yolo26m.pt"
        elif model_choice in ("yolo26l.pt", "large", "4", "l"):
            print("Model selected: Large")
            model_path = "yolo26l.pt"
        elif model_choice in ("yolo26x.pt", "extra large", "5", "x"):
            print("Model selected: Extra Large")
            model_path = "yolo26x.pt"
        elif model_choice in ("custom", "6", "c"):
            print("Model selected: Custom")
            model_path = input("Enter the path to your custom model: ").strip().strip('&').strip().strip("'").strip('"').strip()
            if not model_path:
                print("No model path provided. Please check the path and try again.")
                continue
            print("Checking the specified path for a model...")
            custom_model_path = Path(model_path)
            if not custom_model_path.is_absolute():
                custom_model_path = project_folder / custom_model_path
            if not custom_model_path.is_file():
                print("Could not find the model. Please check the path and try again.")
                continue
            if custom_model_path.suffix.lower() not in (".pt", ".onnx"):
                print("Unsupported model format. The supported model formats are '.pt' and '.onnx'. Please check the path and try again.")
                continue
            model_path = str(custom_model_path)
        elif model_choice in ("back", "7", "b"):
            print("Returning to the main menu...")
            return None
        else:
            print("Invalid input. Please select a valid option or leave blank for the default model.")
            continue
        print("Model file found.")
        print("Loading the selected model...")
        try:
            model = YOLO(model_path)
            print("Model loaded successfully.")
            model_name = Path(model_path).stem
            return model, model_name
        except Exception as error:
            print(f"Could not load the selected model: {error}")
            continue


def select_confidence():
    while True:
        try:
            confidence = float(input("Enter the confidence threshold for the detection (0-1): "))
            if 0 <= confidence <= 1:
                return confidence
            else:
                print("Please enter a number between 0 and 1")
        except ValueError:
            print("Invalid input. Please enter a numerical value.")


def choose_images():
    while True:
        # Prompt the user to input the path to the image or folder to analyse
        image_input = input("\nEnter the path to the image or folder you want to analyse (or leave blank to use the 'input' folder): ").strip().strip('&').strip().strip("'").strip('"').strip()
        image_input = image_input or "input"
        # Make relative paths start from the Python script's folder
        selected_image_path = Path(image_input)
        if not selected_image_path.is_absolute():
            selected_image_path = project_folder / selected_image_path
        # Ensure the path exists and is a file in a supported format or folder
        print(f"Checking the selected path ({selected_image_path}) for the image(s)...")
        if not selected_image_path.exists():
            print("The selected path does not exist. Please check the path and try again.")
            continue
        # If the selected path is a single image in a supported format, return it as a list.
        if selected_image_path.is_file():
            if selected_image_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                return [selected_image_path]
            else:
                print("Unsupported image format. The supported formats are '.jpg', '.jpeg', '.png', '.bmp' and '.webp'")
                continue
        # If the selected path is a folder, search for all supported images within it and return them as a list.
        elif selected_image_path.is_dir():
            images = []
            for file in selected_image_path.rglob("*"):
                if file.is_file() and file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    images.append(file)
            if len(images) > 0:
                print(f"Found {len(images)} images within the folder {selected_image_path}.")
                return images
            else:
                print(f"No supported images found in {selected_image_path}.")
                continue
        else:
            print("The path is neither a file nor a folder. Please check the path and try again.")


def create_output_path(image, model_name, confidence):
    number = 0
    filename = f"{image.stem}_{model_name}_{confidence:g}_annotated"
    # If the filename already exists, increase its number by 1 until a unique filename is found
    while True:
        if number == 0 and not (output_folder / f"{filename}{image.suffix}").exists():
            output_path = output_folder / f"{filename}{image.suffix}"
            return output_path
        else:
            number += 1
            output_path = output_folder / f"{filename}_{number}{image.suffix}"
            if not output_path.exists():
                return output_path


def detect_people():
    # Choose the model for the detection
    model_selection = choose_model()
    # If the user selected "7. Back", this sends them back to the main menu
    if model_selection is None:
        return
    model, model_name = model_selection
    # Choose the confidence threshold for the detection
    confidence = select_confidence()
    # Choose the image(s) to analyse
    images = choose_images()
    # Create the output folder if it doesn't already exist
    output_folder.mkdir(exist_ok=True)
    # Process each image individually
    for image in images:
        print(f"\nRunning detection on: {image.name}")
        # Run the detection
        results = model.predict(source=str(image), classes=[0], conf=confidence)
        # Fetch the detection result for the current image
        result = results[0]
        # Count the people
        person_count = len(result.boxes)
        print(f"Detection completed for {image.name}.")
        print(f"People detected in {image.name}: {person_count}")
        # Create a unique output filename
        output_path = create_output_path(image, model_name, confidence)
        # Save the annotated image
        result.save(filename=str(output_path))
        print(f"Annotated image ({image.name}) saved to: {output_path}")
    print("\nAll images processed successfully.")
    print(f"Annotated images saved in {output_folder}.")


def read_instructions():
    # Display the instructions for using the program
    print("\n--- INSTRUCTIONS ---")
    print("-STEP 1: Preparation-")
    print("Before running the program, ensure you have Python and the required dependencies installed. \nYou can install the Ultralytics package by running the command 'pip install ultralytics' in your terminal.")
    print("Ensure the 'input' folder exists in the same directory as this script, or create it if it doesn't exist.")
    print("Place the images you would like to analyse in the 'input' folder, or copy the path to an image or folder elsewhere on your system.")
    print("-STEP 2: Selecting a model-")
    print("The program will ask for a YOLO model to use for detection. \nYou can leave the selection blank to use the default model, Nano (yolo26n.pt). \nAlternatively, you can select Small, Medium, Large, or Extra Large. \nYou can also choose a custom model, which you will have to provide the path to.")
    print("-STEP 3: Selecting a confidence threshold-")
    print("The program will ask for a confidence threshold to use. \nThe confidence is an estimate of how sure the model is of its own prediction, and by setting it as a threshold only detections more confident than the number you choose will be counted. \nA higher threshold generally provides higher precision and lower recall, and a lower threshold generally provides lower precision and higher recall.")
    print("-STEP 4: Selecting the images to analyse-")
    print("The program will ask for a path to an image or folder to analyse. \nYou can leave the input blank to use whatever is within the 'input' folder. \nDragging and dropping (copying over) a folder or image also works when pasting the image path. \nYou can also enter a relative path, which will be searched from the folder containing the python program. \nThe program also checks for subfolders within the inputted folder, and analyses all images it finds.")
    print("-STEP 5: Running the detection-")
    print("The program will run the detection on the selected images, and return the number of people detected in each image. \nThis may take some time, depending on the number of images and the complexity of the detection model.")
    print("-STEP 6: Interpreting the results-")
    print("The program will save the annotated images (with bounding boxes) in the 'output' folder (which will be created if it doesn't exist), with unique filenames. Output filenames include the original image name, the model name, the confidence threshold, and a unique number to distinguish different detections of the same image. \nYou can check the output folder to see the results of the detection.")
    # Display a warning message about the program's performance and accuracy
    print("\n---WARNINGS---")
    print("The program can only handle image formats '.jpg', '.jpeg', '.png', '.webp', '.bmp'. Ensure your image files are in one of these extensions.")
    print("The program can only handle model formats '.pt' and '.onnx'. Ensure your model file is in one of these extensions.")
    print("The program may need to download a pre-trained model the first time it is run, which may take some time depending on your internet connection.")
    print("The program may take some time to process the images, depending on the number of images and the complexity of the detection model.")
    print("The program may also use a significant amount of memory, depending on the size of the images and the detection model used.")
    print("The program assumes the custom model used is a YOLO model with class '0' being the 'person' class. If this is not the case, the program may not work as expected.")
    print("The program is not guaranteed to be accurate, and may produce false positives or false negatives. Double-check crucial results manually.")
    # Finish the instructions with a closing message
    print("\nEnjoy using the YOLO Person Detection Program!")


def show_credits():
    # Display the credits for the program
    print("\n--- CREDITS ---")
    print("Program written by Manuel Heredia")
    print("Using packages: \nUltralytics \npathlib")
    # Thank the providers of the YOLO models and the ultralytics library
    print("\nSpecial thanks to the open-source community for providing the YOLO models and the ultralytics library.")
    # Offer contact information for feedback or questions about the program
    print("\nThis program was made as a personal project to learn more about computer vision and object detection.")
    print("If you have any questions or feedback, please feel free to reach out to me at the following email: manu.heredia.estrada@gmail.com")
    # Encourage the user to enjoy using the program
    print("\nEnjoy using the YOLO Person Detection Program!")



# The main function that runs the program
def main():
    # Show the introduction message
    introduction()
    # Start the main loop to show the menu and handle user choices
    while True:
        choice = show_menu()
        if choice == 1:
            # If the user selects option 1, start detecting people.
            detect_people()
        elif choice == 2:
            # If the user selects option 2, show the instructions.
            read_instructions()
        elif choice == 3:
            # If the user selects option 3, show the credits.
            show_credits()
        elif choice == 4:
            # If the user selects option 4, ask for confirmation before exiting the program.
            confirmation = input("Are you sure you want to exit? (enter 'yes'/'y' to confirm): ").strip().lower()
            if confirmation in ("yes", "y"):
                print("Thank you for using the YOLO Person Detection Program. Goodbye!")
                print("Exiting the program...")
                break
            else:
                print("Returning to the main menu...")    
        else:
            # If the user enters an invalid choice, display an error message and prompt them to select a valid option.
            print("Invalid input. Please select a valid option (1, 2, 3, or 4).")

# If the script is run directly, call the main function.
if __name__ == "__main__":
    main()