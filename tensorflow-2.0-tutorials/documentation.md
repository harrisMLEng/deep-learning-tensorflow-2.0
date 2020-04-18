# Tensorflow Documentation

## Callbacks
  
### ModelCheckpoint

```keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)```

Save the model after every epoch.

filepath can contain named formatting options, which will be filled with the values of epoch and keys in logs (passed in on_epoch_end).

For example: if filepath is weights.{epoch:02d}-{val_loss:.2f}.hdf5, then the model checkpoints will be saved with the epoch number and the validation loss in the filename.

#### Arguments

- filepath: string, path to save the model file.
- monitor: quantity to monitor.
- verbose: verbosity mode, 0 or 1.
- <strong>save_best_only: if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
- save_weights_only: if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved - -(model.save(filepath)).</strong>
- mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the - maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto -mode, the direction is automatically inferred from the name of the monitored quantity.
- period: Interval (number of epochs) between checkpoints.

#### Example 
```checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5")
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])```

### EarlyStopping

```keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)```

#### Arguments

- monitor: quantity to be monitored.
- min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
- <strong>patience: number of epochs that produced the monitored quantity with no improvement after which training will be stopped. Validation quantities may not be produced for every epoch, if the validation frequency (model.fit(validation_freq=5)) is greater than one. </strong>
- verbose: verbosity mode.
- mode: one of {auto, min, max}. In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.
- baseline: Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline.
- restore_best_weights: whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.

#### Example 

```early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100,
validation_data=(X_valid, y_valid),
callbacks=[checkpoint_cb, early_stopping_cb])```

	
