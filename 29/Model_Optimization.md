Understand model optimization techniques and how to apply them to improve your model's efficiency while maintaining performance. Let's break this down into key sections:

1. Model Optimization Overview
```python
# Common optimization techniques:
# 1. Quantization - Reducing numerical precision
# 2. Pruning - Removing unnecessary weights
# 3. Knowledge Distillation - Training smaller models from larger ones
# 4. Architecture Optimization - Designing efficient model structures
```

2. Quantization Example (using TensorFlow)
```python
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('your_model.h5')

# Convert to TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: Specify target data type
converter.target_spec.supported_types = [tf.float16]

# Convert the model
quantized_tflite_model = converter.convert()

# Save the quantized model
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)
```

3. Pruning Example (using TensorFlow Model Optimization Toolkit)
```python
import tensorflow_model_optimization as tfmot

# Define pruning schedule
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

# Apply pruning to model
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    **pruning_params
)

# Compile pruned model
model_for_pruning.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train pruned model
model_for_pruning.fit(
    train_data,
    train_labels,
    epochs=10,
    validation_data=(val_data, val_labels)
)

# Strip pruning wrapper
final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
```

4. Evaluating Optimization Impact
```python
def evaluate_model_performance(original_model, optimized_model, test_data):
    # Compare inference speed
    import time

    def measure_inference_time(model, data):
        start_time = time.time()
        model.predict(data)
        end_time = time.time()
        return end_time - start_time

    original_time = measure_inference_time(original_model, test_data)
    optimized_time = measure_inference_time(optimized_model, test_data)

    # Compare model sizes
    import os
    original_size = os.path.getsize('original_model.h5')
    optimized_size = os.path.getsize('optimized_model.h5')

    # Compare accuracy
    original_accuracy = original_model.evaluate(test_data)[1]
    optimized_accuracy = optimized_model.evaluate(test_data)[1]

    print(f"Original Model:\n"
          f"Inference Time: {original_time:.4f}s\n"
          f"Model Size: {original_size/1024:.2f}KB\n"
          f"Accuracy: {original_accuracy:.4f}\n")

    print(f"Optimized Model:\n"
          f"Inference Time: {optimized_time:.4f}s\n"
          f"Model Size: {optimized_size/1024:.2f}KB\n"
          f"Accuracy: {optimized_accuracy:.4f}")
```

5. Complete Optimization Pipeline
```python
def optimize_model(model, train_data, val_data, test_data):
    # 1. Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()

    # 2. Pruning
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=1000
        )
    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        **pruning_params
    )

    # 3. Train pruned model
    pruned_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    pruned_model.fit(
        train_data,
        epochs=5,
        validation_data=val_data
    )

    # 4. Evaluate and compare
    evaluate_model_performance(model, pruned_model, test_data)

    return pruned_model, quantized_model
```

Best Practices for Model Optimization:

1. Start with a baseline model performance measurement
2. Apply optimizations incrementally and measure impact
3. Consider the trade-off between model size, speed, and accuracy
4. Test optimized models thoroughly on target hardware
5. Document performance changes for each optimization technique

Remember:
- Not all optimization techniques are suitable for every model
- Test on your target deployment platform
- Monitor accuracy degradation carefully
- Consider your specific requirements (size vs. speed vs. accuracy)

This code provides a foundation for model optimization. Adjust parameters and techniques based on your specific needs and constraints.
