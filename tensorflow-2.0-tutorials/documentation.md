# Tensorflow-Keras Documentation
<i> Reference http://keras.io </i>
<i> Reference https://www.tensorflow.org/api_docs/python/tf </i>
## Callbacks
  
### ModelCheckpoint

```
keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

Save the model after every epoch.

filepath can contain named formatting options, which will be filled with the values of epoch and keys in logs (passed in on_epoch_end).

For example: if filepath is weights.{epoch:02d}-{val_loss:.2f}.hdf5, then the model checkpoints will be saved with the epoch number and the validation loss in the filename.

#### Arguments

- filepath: string, path to save the model file.
- monitor: quantity to monitor.
- verbose: verbosity mode, 0 or 1.
- <strong>save_best_only: if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.</strong>
- save_weights_only: if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved - -(model.save(filepath)).
- mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the - maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto -mode, the direction is automatically inferred from the name of the monitored quantity.
- period: Interval (number of epochs) between checkpoints.

#### Example 

```
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5")
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])
```

### EarlyStopping

```
keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
```

#### Arguments

- monitor: quantity to be monitored.
- min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
- <strong>patience: number of epochs that produced the monitored quantity with no improvement after which training will be stopped. Validation quantities may not be produced for every epoch, if the validation frequency (model.fit(validation_freq=5)) is greater than one. </strong>
- verbose: verbosity mode.
- mode: one of {auto, min, max}. In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.
- baseline: Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline.
- <strong>restore_best_weights: whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.</strong>

#### Example 

```
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100,
validation_data=(X_valid, y_valid),
callbacks=[checkpoint_cb, early_stopping_cb])
```


### Tensorboard 

```
tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None, **kwargs
)
```

#### Arguments:
- log_dir: the path of the directory where to save the log files to be parsed by TensorBoard.
- histogram_freq: frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations.
- write_graph: whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
- write_images: whether to write model weights to visualize as image in TensorBoard.
- update_freq: 'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 1000, the callback will write the metrics and losses to TensorBoard every 1000 batches. Note that writing too frequently to TensorBoard can slow down your training.
- profile_batch: Profile the batch to sample compute characteristics. By default, it will profile the second batch. Set profile_batch=0 to disable profiling. Must run in TensorFlow eager mode.
- embeddings_freq: frequency (in epochs) at which embedding layers will be visualized. If set to 0, embeddings won't be visualized.
- embeddings_metadata: a dictionary which maps layer name to a file name in which metadata for this embedding layer is saved. See the details about metadata files format. In case if the same metadata file is used for all embedding layers, string can be passed.


#### Example 

```
import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
import time
run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22'

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
validation_data=(X_valid, y_valid),
callbacks=[tensorboard_cb])
```	

## tf.keras

### tf.keras.utils.get_file

- Downloads a file from a URL if it not already in the cache.

```
tf.keras.utils.get_file(
    fname, origin, untar=False, md5_hash=None, file_hash=None,
    cache_subdir='datasets', hash_algorithm='auto', extract=False,
    archive_format='auto', cache_dir=None
)
```

By default the file at the url origin is downloaded to the cache_dir ~/.keras, placed in the cache_subdir datasets, and given the filename fname. The final location of a file example.txt would therefore be ~/.keras/datasets/example.txt.

Files in tar, tar.gz, tar.bz, and zip formats can also be extracted. Passing a hash will verify the file after download. The command line programs shasum and sha256sum can compute the hash.

Arguments:
- fname: Name of the file. If an absolute path /path/to/file.txt is specified the file will be saved at that location.
- origin: Original URL of the file.
- untar: Deprecated in favor of 'extract'. boolean, whether the file should be decompressed
- md5_hash: Deprecated in favor of 'file_hash'. md5 hash of the file for verification
- file_hash: The expected hash string of the file after download. The sha256 and md5 hash algorithms are both supported.
- cache_subdir: Subdirectory under the Keras cache dir where the file is saved. If an absolute path /path/to/folder is specified the file will be saved at that location.
- hash_algorithm: Select the hash algorithm to verify the file. options are 'md5', 'sha256', and 'auto'. The default 'auto' detects the hash algorithm in use.
- extract: True tries extracting the file as an Archive, like tar or zip.
- archive_format: Archive format to try for extracting the file. Options are 'auto', 'tar', 'zip', and None. 'tar' includes tar, tar.gz, and tar.bz files. The default 'auto' is ['tar', 'zip']. None or an empty list will return no matches found.
- cache_dir: Location to store cached files, when None it defaults to the Keras Directory.

- Returns:
Path to the downloaded file


### tf.keras.preprocessing.image.ImageDataGenerator

- Generate batches of tensor image data with real-time data augmentation.

```
tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
    height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,
    vertical_flip=False, rescale=None, preprocessing_function=None,
    data_format=None, validation_split=0.0, dtype=None
)
```

### DirectoryIterator

- Iterator capable of reading images from a directory on disk.
- Returned from ImageGenerator
- x, y = next() returns the next batch of numpy arrays and the labels of the batch
 
Arguments:
- directory: Path to the directory to read images from. Each subdirectory in this directory will be considered to contain images from one class, or alternatively you could specify class subdirectories via the classes argument.
- image_data_generator: Instance of ImageDataGenerator to use for random transformations and normalization.
- target_size: tuple of integers, dimensions to resize input images to.
- color_mode: One of "rgb", "rgba", "grayscale". Color mode to read images.
- classes: Optional list of strings, names of subdirectories containing images from each class (e.g. ["dogs", "cats"]). It will be computed automatically if not set.
- class_mode: Mode for yielding the targets: "binary": binary targets (if there are only two classes), "categorical": categorical targets, "sparse": integer targets, "input": targets are images identical to input images (mainly used to work with autoencoders), None: no targets get yielded (only input images are yielded).
- batch_size: Integer, size of a batch.
- shuffle: Boolean, whether to shuffle the data between epochs.
- seed: Random seed for data shuffling.
- data_format: String, one of channels_first, channels_last.
- save_to_dir: Optional directory where to save the pictures being yielded, in a viewable format. This is useful for visualizing the random transformations being applied, for debugging purposes.
- save_prefix: String prefix to use for saving sample images (if save_to_dir is set).
- save_format: Format to use for saving sample images (if save_to_dir is set).
- subset: Subset of data ("training" or "validation") if validation_split is set in ImageDataGenerator.
- interpolation: Interpolation method used to resample the image if the target size is different from that of the loaded image. Supported methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3 or newer is installed, "lanczos" is also supported. If PIL version 3.4.0 or newer is installed, "box" and "hamming" are also supported. By default, "nearest" is used.
- dtype: Dtype to use for generated arrays.
Attributes:
- filepaths: List of absolute paths to image files
- labels: Class labels of every observation
- sample_weight

### tf.keras.models.Sequential

### fit 

```
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
    use_multiprocessing=False, **kwargs
)
```

Fits the model on data yielded batch-by-batch by a Python generator. (deprecated)

### fit_generator


```
fit_generator(
    generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None,
    validation_data=None, validation_steps=None, validation_freq=1,
    class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False,
    shuffle=True, initial_epoch=0
)
```

### compile

```
compile(
    optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
    sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
    distribute=None, **kwargs
)
```

- Configures the model for training.

Arguments:
- optimizer: String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
- loss: String (name of objective function), objective function or tf.keras.losses.Loss instance. See tf.keras.losses. An objective function is any callable with the signature scalar_loss = fn(y_true, y_pred). If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses.
- metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use metrics=['accuracy']. To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}. You can also pass a list (len = len(outputs)) of lists of metrics such as metrics=[['accuracy'], ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']].
- loss_weights: Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value that will be minimized by the model will then be the weighted sum of all individual losses, weighted by the loss_weights coefficients. If a list, it is expected to have a 1:1 mapping to the model's outputs. If a tensor, it is expected to map output names (strings) to scalar coefficients.
sample_weight_mode: If you need to do timestep-wise sample weighting (2D weights), set this to "temporal". None defaults to sample-wise weights (1D). If the model has multiple outputs, you can use a different sample_weight_mode on each output by passing a dictionary or a list of modes.
- weighted_metrics: List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
- target_tensors: By default, Keras will create placeholders for the model's target, which will be fed with the target data during training. If instead you would like to use your own target tensors (in turn, Keras will not expect external Numpy data for these targets at training time), you can specify them via the target_tensors argument. It can be a single tensor (for a single-output model), a list of tensors, or a dict mapping output names to target tensors.
- distribute: NOT SUPPORTED IN TF 2.0, please create and compile the model under distribution strategy scope instead of passing it to compile.
- **kwargs: Any additional arguments.


