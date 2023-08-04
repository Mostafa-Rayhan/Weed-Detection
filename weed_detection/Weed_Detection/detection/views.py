from django.shortcuts import render
from django.views import View
import tensorflow as tf

from .result import output

from django.core.files.storage import FileSystemStorage
from django.conf import settings
from io import BytesIO
from PIL import Image
from django.core.files import File
import os

# from joblib import load
# model = load('./savedModels/')

# Create your views here.

# actual_class2,predicted_class2,confidence2 = output(test_ds)

# print(actual_class2)

IMAGE_SIZE = 256
BATCH_SIZE = 42
CHANNELS = 3

def get_dataset2(ds ,train_split=0.0,val_split=0.0 , test_split=1.0,shuffle=True, shuffle_size =1000):
  assert(train_split+val_split+test_split)==1
  ds_size = len(ds)
  if shuffle:
    ds = ds.shuffle(shuffle_size,seed=12)

  train_size = int(train_split*ds_size)
  val_size = int (val_split*ds_size)

  train_ds = ds.take(train_size)
  val_ds = ds.skip(train_size).take(val_size)
  test_ds = ds.skip(train_size).skip(val_size)

  return test_ds


# dataset_given = tf.keras.preprocessing.image_dataset_from_directory(
#     "C:\\Users\\DCL\\Desktop\\polash\\Weed-Detection\\weed_detection\\Weed_Detection\\static\\given",
#     seed=1,
#     shuffle=True,
#     image_size =(IMAGE_SIZE,IMAGE_SIZE),
#     batch_size =BATCH_SIZE
#   )
  
def data_prepro(dataset_given):
  
  # dataset_given = tf.keras.preprocessing.image_dataset_from_directory(
  #   "C:\\Users\\DCL\\Desktop\\polash\\Weed-Detection\\weed_detection\\Weed_Detection\\static\\given",
  #   seed=1,
  #   shuffle=True,
  #   image_size =(IMAGE_SIZE,IMAGE_SIZE),
  #   batch_size =BATCH_SIZE
  #   )
  
  test_ds = get_dataset2(dataset_given)

  return test_ds

# test_ds = data_prepro()

# test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE)

# class_names2 =dataset_given.class_names
# class_names2


def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact_us.html')

def image_compress_save(image, img_name):
    im = Image.open(image)
    im = im.convert('RGB')
    im_io = BytesIO()
    im.save(im_io, 'JPEG', quality=60)
    compressed_image = File(im_io, name=img_name)
    print(settings.STATIC_URL)
    print(settings.STATIC_ROOT)
    print(settings.STATICFILES_DIRS[0])
    FileSystemStorage(location=os.path.join(settings.STATICFILES_DIRS[0], 'given','user')).save(img_name,
                                                                                                    compressed_image)


def prediction(request):

    if request.method == 'GET':
            return render(request, 'Prediction.html')
    
    if request.method == 'POST':


        image_compress_save(request.FILES['upload_image'],'img_example.jpg')


        dataset_given = tf.keras.preprocessing.image_dataset_from_directory(
            "C:\\Users\\DCL\\Desktop\\polash\\Weed-Detection\\weed_detection\\Weed_Detection\\static\\given",
            seed=1,
            shuffle=True,
            image_size =(IMAGE_SIZE,IMAGE_SIZE),
            batch_size =BATCH_SIZE
            
          )
        
        test_ds = data_prepro(dataset_given)
        test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE)
        class_names2 =dataset_given.class_names
        class_names2
        actual_class2,predicted_class2,confidence2 = output(test_ds,class_names2)

        context = {

            'actual' : actual_class2,
            'predict' : predicted_class2,
            'conf' : confidence2
        }

        return render(request, 'Prediction.html', context)
    
    return render(request, 'Prediction.html')


