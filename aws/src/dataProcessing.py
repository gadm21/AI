

from utils import *





def main():
    file_names = os.listdir(small_train_dir)[:3]
    file_names = [os.path.join(small_train_dir, file_name) for file_name in file_names]
    
    
    status =upload_file(bucket_name, file_names[0], 'input')
    print(status)

if __name__ == '__main__': 
    
    main()