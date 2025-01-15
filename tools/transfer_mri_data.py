import os
import shutil

# Recursive folder copy function
def copy_folder_contents(src, dst):
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)
        if os.path.isdir(src_item):
            copy_folder_contents(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

def main():
    #count = 0 # for testing
    # Loop through the folders
    for root_dir in ["schizconnect_COBRE_images_22613/COBRE", "schizconnect_MCICShare_images_22613/MCICShare"]:
        root_path = os.path.join(source_dir, root_dir)
        #print(f"Processing root directory: {root_path}")

        if not os.path.exists(root_path):
            print(f"Root path does not exist: {root_path}")
            continue

        for subject in os.listdir(root_path):
            subject_path = os.path.join(root_path, subject)

            if os.path.isdir(subject_path):
                for ses in os.listdir(subject_path):
                    ses_path = os.path.join(subject_path, ses)

                    if os.path.isdir(ses_path):
                        anat_path = os.path.join(ses_path, "anat")

                        if os.path.exists(anat_path):
                            dest_path = os.path.join(destination_dir, root_dir, subject, ses, "anat")
                            print(f"Copying from: {anat_path}")
                            print(f"Copying to: {dest_path}")
                            copy_folder_contents(anat_path, dest_path)
                            #count += 1
                            #if count == 3:
                            #    break
                        #break
                #break
        #break    

    print("Copy operation completed!")

    ### takes 8.5 minutes to copy 20 GB of data from WSL do Windows HDD (external)

if __name__ == '__main__':
    # Define the source and destination directories
    cwd = os.getcwd()
    source_dir = cwd + '/data' #
    destination_dir = "/mnt/e/1DATA" # Explicit Windows path for HDD
    
    main(source_dir, destination_dir)