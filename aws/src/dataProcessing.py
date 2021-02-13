

from utils import *





def main():
    file_names = os.listdir(test_dir)
    file_names = [os.path.join(small_train_dir, file_name) for file_name in file_names]
    
    manifest = read_manifest(manifest_file)
    replace_from = 'sagemaker-us-east-1-229639953574/input'
    replace_to = bucket_prefix
    manifest = manifest.replace(replace_from, replace_to)
    with open(manifest_file, 'wt') as  file : 
        file.write(manifest)
        
    print("done")


def main2():

    files = os.listdir(test_dir)
    manifest = read_manifest(manifest_file)
    manifest = manifest.split('\n')
    new_manifest = [] 

    replace_from = 'sagemaker-remars/datasets/na-bees/500/'
    replace_to = val_bucket_prefix
    print(files[0])
    print(manifest[0])
    print(len(manifest))
    for man in manifest:
        for file in files :
            if file in man :
                print(file)
                new_man = man.replace(replace_from, replace_to)
                new_manifest.append(new_man)
                break
    print(len(new_manifest))
    new_manifest = '\n'.join(new_manifest)
    with open(manifest_file, 'wt') as file :
        file.write(new_manifest)
    
    print("done")


def main3():

    manifest = read_manifest(manifest_file)
    replace_from = 'randomfirstbucket/Input/train'
    replace_to = train_bucket_prefix
    manifest = manifest.replace(replace_from, replace_to)
    with open(manifest_file, 'wt') as  file : 
        file.write(manifest)
        
    print("done")

if __name__ == '__main__': 
    
    main2()