import re
import os
import shutil

# Run this file to set up the tiny-imagenet-200 dataset. Before running, make sure that the tiny-imagenet-200 is extracted
# to this working directory. The script copy each validation image in the dataset and sort them in folders for each class,
# like the structure in the train subdirectory. The script will delete the val_annotations.txt and val/images subdirectory
# when completed.

if __name__ == "__main__":
    annotation_file = open("tiny-imagenet-200/val/val_annotations.txt", "r")
    annotation_file_lines = annotation_file.readlines()
    annotation_file.close()

    for line in annotation_file_lines:
        # Regex search capturing class name and image name in two capture groups
        search = re.search(r"(val_\w+.JPEG)\t+(n\d{8})", line)

        if not isinstance(search, re.Match):
            raise Exception(f"Line not matching regex. Line: {line}")

        file_name = search.group(1)
        class_name = search.group(2)

        print(f"{file_name} -> {class_name}")
        if not os.path.exists(f"tiny-imagenet-200/val/{class_name}"):
            os.mkdir(f"tiny-imagenet-200/val/{class_name}")

        shutil.copyfile(f"tiny-imagenet-200/val/images/{file_name}", f"tiny-imagenet-200/val/{class_name}/{file_name}")

    should_delete = input("Delete tiny-imagenet-200/val/image folder and val_annotations.txt? (y/N) ").lower() == 'y'
    if should_delete:
        print("Deleting...")
        shutil.rmtree("tiny-imagenet-200/val/images")
        os.remove("tiny-imagenet-200/val/val_annotations.txt")
    else: 
        print("Not deleting...")