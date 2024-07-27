import os


def list_files_in_directory(directory):
    try:
        # Get a list of all files and directories in the specified directory
        files_and_directories = os.listdir(directory)

        # Filter out directories, only keep files
        files = [f for f in files_and_directories if os.path.isfile(os.path.join(directory, f))]

        return files
    except Exception as e:
        return str(e)


def write_files_to_text_file(files, output_file):
    try:
        with open(output_file, 'w') as file:
            for f in files:
                file.write(f + '\n')
        print(f"List of files has been written to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage:
directory_path = 'data/Anticipating-Accident/Dachcam_dataset/obj_data/testing'  # You can change this to any directory you want to list files from
output_file = 'file_list.txt'  # The name of the output text file

files = list_files_in_directory(directory_path)
write_files_to_text_file(files, output_file)
