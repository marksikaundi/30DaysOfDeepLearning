Let's go through the process of fine-tuning a pre-trained model on your custom dataset and then evaluating its performance. We'll use a popular deep learning framework like TensorFlow/Keras or PyTorch. For this example, I'll use TensorFlow/Keras, but the steps are similar for PyTorch.

### Step 1: Set Up Your Environment

Make sure you have TensorFlow installed. You can install it using pip if you haven't already:

```bash
pip install tensorflow
```

### Step 2: Load a Pre-trained Model

We'll use a pre-trained model from TensorFlow's Keras applications. For this example, let's use the MobileNetV2 model, which is a lightweight model suitable for fine-tuning.

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Freeze the base model
base_model.trainable = False
```

### Step 3: Prepare Your Custom Dataset

Assume you have your custom dataset organized in directories. We'll use TensorFlow's `ImageDataGenerator` to load and preprocess the images.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
train_dir = 'path/to/train'
validation_dir = 'path/to/validation'

# Create ImageDataGenerator instances
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

### Step 4: Add Custom Layers

Add custom layers on top of the pre-trained base model to adapt it to your specific task.

```python
# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)
```

### Step 5: Compile the Model

Compile the model with an appropriate optimizer, loss function, and metrics.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Step 6: Train the Model

Train the model using the training and validation data generators.

```python
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)
```

### Step 7: Evaluate the Model

Evaluate the performance of the fine-tuned model on the validation set.

```python
# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')
```

### Step 8: Fine-Tune the Model (Optional)

If you want to further improve the model's performance, you can unfreeze some layers of the base model and fine-tune them.

```python
# Unfreeze some layers of the base model
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training the model
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)
```

### Step 9: Evaluate the Fine-Tuned Model

Evaluate the performance of the fine-tuned model again.

```python
# Evaluate the fine-tuned model
loss_fine, accuracy_fine = model.evaluate(validation_generator)
print(f'Fine-Tuned Validation Loss: {loss_fine}')
print(f'Fine-Tuned Validation Accuracy: {accuracy_fine}')
```

That's it! You've successfully fine-tuned a pre-trained model on your custom dataset and evaluated its performance.
