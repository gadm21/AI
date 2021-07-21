




from utils import * 





def augment_yolo_data():
    data_dir = 'yolo_data2'
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    image_paths = [os.path.join(images_dir, image) for image in sorted(os.listdir(images_dir))]
    label_paths = [os.path.join(labels_dir, label) for label in sorted(os.listdir(labels_dir))]

    new_images_dir = os.path.join('augmented_yolo_data2', 'images')
    new_labels_dir = os.path.join('augmented_yolo_data2', 'labels')
    
    transform = my_augmentation()
    class_labels = ['sperm']
    counter = 0
    for image_path, label_path in zip(image_paths, label_paths):
        counter += 1

        image = cv_utils.read_image(image_path)
        boxes = get_yolo_labels(label_path)

        cv2.imwrite(os.path.join(new_images_dir, str(counter))+'.jpg', image)
        write_yolo_labels(os.path.join(new_labels_dir, str(counter)+'.txt'), boxes)

        for _ in range(10):
            counter += 1
            transformed = transform(image = image, bboxes = boxes, class_labels = class_labels*len(boxes))
            cv2.imwrite(os.path.join(new_images_dir, str(counter))+'.jpg', transformed['image'])
            write_yolo_labels(os.path.join(new_labels_dir, str(counter)+'.txt'), transformed['bboxes'])

        


    # t_image = transform(image = image)['image']
    # print(t_image.keys())
    # cv_utils.show([image, t_image])


    



if __name__ == "__main__":

    augment_yolo_data()