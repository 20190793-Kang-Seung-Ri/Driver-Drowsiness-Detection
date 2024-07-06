import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,Input,Flatten,Dense,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data Augumentation
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau

# 교육 및 유효성 검사 데이터를 로드합니다.

tf.test.is_gpu_available()

# 머신 러닝 모델 아키텍처를 정의합니다.

batchsize=8

# 적절한 손실 함수와 최적화기를 사용하여 모델을 컴파일합니다.

train_datagen= ImageDataGenerator(rescale=1./255, rotation_range=0.2,shear_range=0.2,
    zoom_range=0.2,width_shift_range=0.2,
    height_shift_range=0.2, validation_split=0.2)

train_data= train_datagen.flow_from_directory(r'C:\_AppleBanana\dataset',
                                target_size=(80,80),batch_size=batchsize,class_mode='categorical',subset='training' )

validation_data= train_datagen.flow_from_directory(r'C:\_AppleBanana\dataset',
                                target_size=(80,80),batch_size=batchsize,class_mode='categorical', subset='validation')

# 교육 데이터에 대해 모델을 교육합니다.

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(r'C:\_AppleBanana\dataset',
                                target_size=(80,80),batch_size=batchsize,class_mode='categorical')

# 유효성 검사 데이터에서 모델을 평가합니다.

bmodel = InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(80,80,3)))
hmodel = bmodel.output
hmodel = Flatten()(hmodel)
hmodel = Dense(64, activation='relu')(hmodel)
hmodel = Dropout(0.5)(hmodel)
hmodel = Dense(2,activation= 'softmax')(hmodel)

model = Model(inputs=bmodel.input, outputs= hmodel)
for layer in bmodel.layers:
    layer.trainable = False

# 훈련된 모델과 평가 메트릭을 저장합니다.

model.summary()

checkpoint = ModelCheckpoint(r'C:\_AppleBanana\승리_장난감\modeling\model_weights.keras',
                            monitor='val_loss',save_best_only=True,verbose=3)


earlystop = EarlyStopping(monitor = 'val_loss', patience=7, verbose= 3, restore_best_weights=True)

learning_rate = ReduceLROnPlateau(monitor= 'val_loss', patience=3, verbose= 3, )

callbacks=[checkpoint,earlystop,learning_rate]

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_data,steps_per_epoch=train_data.samples//batchsize,
                   validation_data=validation_data,
                   validation_steps=validation_data.samples//batchsize,
                   callbacks=callbacks,
                    epochs=5)

# 모델 평가

acc_tr, loss_tr = model.evaluate_generator(train_data)
print(acc_tr)
print(loss_tr)

acc_vr, loss_vr = model.evaluate_generator(validation_data)
print(acc_vr)
print(loss_vr)

acc_test, loss_test = model.evaluate_generator(test_data)
print(acc_tr)
print(loss_tr)