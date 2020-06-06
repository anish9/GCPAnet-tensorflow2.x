
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Nadam,Adam,SGD
from layers import *
from conf import *
from utils import schedule,losses,weights
import os
from conf import *
from dataloader import train_,val_,ts,vs

sched = LearningRateScheduler(schedule)

logdir = config_map["tbl"]
epochs = config_map["epochs"]


file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

MOD = GCPA()
MOD.compile(optimizer="Nadam",loss=losses,loss_weights=weights,metrics=["acc"])
call_list = [TensorBoard(logdir),ModelCheckpoint("save_ckp.h5",monitor="val_loss",save_weights_only=True,verbose=1,mode="min"),
             EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=9),sched]

print("Training_initialized ...")
print("--------------------------")

MOD.fit(train_,steps_per_epoch=ts,epochs=epochs,callbacks=call_list,validation_data=val_,validation_steps=vs)
