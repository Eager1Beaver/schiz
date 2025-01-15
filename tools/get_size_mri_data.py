import os

def getSize(filename):
    st = os.stat(filename)
    return st.st_size

def main():
    # iterate over "anat" folder for every subject in "data" folder 
    # for both dataset and get size of each file 
    # with extension ".nii.gz", get a sum of all sizes
    total_size = 0

    for root, dirs, files in os.walk("data"):
        # check if the folder is "anat" folder
        if "anat" not in root:
            continue
        for file in files:
            if file.endswith(".nii.gz"):
                try:
                    total_size += getSize(os.path.join(root, file))
                except OSError as e:
                    print(f"Error accessing file {file}: {e}")

    print(f'Total size (MB) = {total_size/1024/1024}')            

if __name__ == '__main__':
    main()